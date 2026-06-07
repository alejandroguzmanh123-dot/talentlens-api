from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import os
import threading
from typing import List
from pydantic import BaseModel

app = FastAPI(title="TalentLens Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FACECHECK_API_TOKEN = os.environ.get("FACECHECK_API_TOKEN", "")

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


async def facecheck_search(image_bytes: bytes, api_token: str) -> list:
    site = "https://facecheck.id"
    headers = {"accept": "application/json", "Authorization": api_token}

    async with httpx.AsyncClient(timeout=30) as client:
        files = {"images": ("photo.jpg", image_bytes, "image/jpeg")}
        resp = await client.post(f"{site}/api/upload_pic", headers=headers, files=files)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"FaceCheck upload error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()

    if data.get("error"):
        raise HTTPException(status_code=502, detail=f"FaceCheck error: {data['error']}")

    id_search = data["id_search"]
    print(f"FaceCheck upload OK, id_search={id_search}")

    json_data = {
        "id_search": id_search,
        "with_progress": True,
        "status_only": False,
        "demo": False,
    }

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

        if score < 50:
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
