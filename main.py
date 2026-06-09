from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import os
import threading
import stripe
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(title="InstaFace Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FACECHECK_API_TOKEN = os.environ.get("FACECHECK_API_TOKEN", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

PLATFORM_MAP = {
    "instagram.com": "instagram",
    "facebook.com": "facebook",
    "linkedin.com": "linkedin",
    "tiktok.com": "tiktok",
    "youtube.com": "youtube",
    "twitter.com": "twitter",
    "x.com": "twitter",
}

NEWS_DOMAINS = [
    "bbc.", "cnn.", "nytimes.", "reuters.", "forbes.", "bloomberg.",
    "theguardian.", "washingtonpost.", "latimes.", "eluniversal.",
    "milenio.", "excelsior.", "reforma.", "elfinanciero.", "expansion.",
    "infobae.", "proceso.", "eleconomista.", "jornada.", "univision.",
    "telemundo.", "azteca.", "televisa.", "apnews.", "huffpost.",
    "businessinsider.", "techcrunch.", "wired.", "medium.", "substack.",
]

PACKAGES = {
    "1":  {"tokens": 1,  "price": 350,  "name": "InstaFace - 1 Search"},
    "10": {"tokens": 10, "price": 3000, "name": "InstaFace - 10 Searches"},
    "25": {"tokens": 25, "price": 6250, "name": "InstaFace - 25 Searches"},
}

face_app = None
face_app_loading = False


def load_face_model():
    global face_app, face_app_loading
    face_app_loading = True
    try:
        import numpy as np
        import cv2
        from insightface.app import FaceAnalysis
        fa = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=0, det_size=(640, 640))
        face_app = fa
        print("InsightFace loaded successfully")
    except Exception as e:
        print(f"InsightFace unavailable: {e}")
        face_app = None
    finally:
        face_app_loading = False


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_face_model, daemon=True)
    thread.start()
    print("InsightFace loading in background...")


def detect_platform(url: str) -> str:
    for domain, platform in PLATFORM_MAP.items():
        if domain in url:
            return platform
    for news_domain in NEWS_DOMAINS:
        if news_domain in url:
            return "news"
    return "web"


class SearchResult(BaseModel):
    platform: str
    imageUrl: str
    profileUrl: str
    matchPercentage: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    biometricEnabled: bool


class CheckoutRequest(BaseModel):
    package: str
    user_id: str
    success_url: str
    cancel_url: str


async def facecheck_search(image_bytes: bytes, api_token: str) -> list:
    site = "https://facecheck.id"
    headers = {"accept": "application/json", "Authorization": api_token}

    async with httpx.AsyncClient(timeout=30) as client:
        files = {"images": ("photo.jpg", image_bytes, "image/jpeg")}
        resp = await client.post(f"{site}/api/upload_pic", headers=headers, files=files)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"FaceCheck upload error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()

    if data.get("error"):
        raise HTTPException(status_code=502, detail=f"FaceCheck error: {data['error']}")

    id_search = data["id_search"]
    print(f"FaceCheck upload OK, id_search={id_search}")

    json_data = {"id_search": id_search, "with_progress": True, "status_only": False, "demo": False}

    for attempt in range(90):
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{site}/api/search", headers=headers, json=json_data)
            data = resp.json()

        if data.get("error"):
            raise HTTPException(status_code=502, detail=f"FaceCheck search error: {data['error']}")

        if data.get("output"):
            items = data["output"]["items"]
            print(f"FaceCheck search complete: {len(items)} results")
            return items

        print(f"FaceCheck progress: {data.get('progress', 0)}% (attempt {attempt + 1})")
        await asyncio.sleep(1)

    print("FaceCheck search timed out after 90s")
    return []


@app.get("/")
def root():
    return {"status": "ok", "biometric": face_app is not None, "loading": face_app_loading}


@app.get("/health")
def health():
    return {"status": "healthy", "biometric": face_app is not None, "loading": face_app_loading}


@app.get("/stripe-config")
def stripe_config():
    return {"publishable_key": STRIPE_PUBLISHABLE_KEY}


@app.post("/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    pkg = PACKAGES.get(req.package)
    if not pkg:
        raise HTTPException(status_code=400, detail="Invalid package. Use '1', '10', or '25'.")

    stripe.api_key = STRIPE_SECRET_KEY

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": pkg["price"],
                "product_data": {"name": pkg["name"]},
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=req.success_url + "?payment=success",
        cancel_url=req.cancel_url,
        metadata={"user_id": req.user_id, "tokens": str(pkg["tokens"])},
    )

    return {"url": session.url}


@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        stripe.api_key = STRIPE_SECRET_KEY
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"].get("user_id")
        tokens = int(session["metadata"].get("tokens", 0))

        if user_id and tokens and SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                from supabase import create_client
                sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
                result = sb.table("user_tokens").select("tokens").eq("user_id", user_id).execute()
                if result.data:
                    current = result.data[0]["tokens"]
                    sb.table("user_tokens").update({"tokens": current + tokens}).eq("user_id", user_id).execute()
                else:
                    sb.table("user_tokens").insert({"user_id": user_id, "tokens": tokens}).execute()
                print(f"Added {tokens} tokens to user {user_id}")
            except Exception as e:
                print(f"Supabase update error: {e}")

    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...)):
    if not FACECHECK_API_TOKEN:
        raise HTTPException(status_code=500, detail="FACECHECK_API_TOKEN not configured")

    image_bytes = await file.read()
    items = await facecheck_search(image_bytes, FACECHECK_API_TOKEN)

    results = []
    seen = set()

    for item in items:
        score = item.get("score", 0)
        url = item.get("url", "")
        base64_img = item.get("base64", "")

        if not url or url in seen:
            continue
        if score < 75:
            continue

        seen.add(url)
        platform = detect_platform(url)
        clean_b64 = base64_img.split(" ")[-1] if base64_img else ""
        image_url = f"data:image/webp;base64,{clean_b64}" if clean_b64 else ""

        results.append(SearchResult(
            platform=platform,
            imageUrl=image_url,
            profileUrl=url,
            matchPercentage=int(score),
        ))

    results.sort(key=lambda x: x.matchPercentage, reverse=True)
    return SearchResponse(results=results, biometricEnabled=face_app is not None)
