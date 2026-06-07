from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import threading
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(title="TalentLens Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERP_API_KEY = os.environ.get("SERP_API_KEY", "")

PLATFORM_MAP = {
    "instagram.com": "instagram",
    "facebook.com": "facebook",
    "linkedin.com": "linkedin",
    "tiktok.com": "tiktok",
    "youtube.com": "youtube",
    "twitter.com": "twitter",
    "x.com": "twitter",
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


def get_embedding(image_bytes: bytes):
    if face_app is None:
        return None
    try:
        import numpy as np
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = face_app.get(img)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return largest.normed_embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def cosine_similarity(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SearchResult(BaseModel):
    platform: str
    imageUrl: str
    profileUrl: str
    matchPercentage: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    biometricEnabled: bool


def detect_platform(url: str) -> Optional[str]:
    for domain, platform in PLATFORM_MAP.items():
        if domain in url:
            return platform
    return None


@app.get("/")
def root():
    return {"status": "ok", "biometric": face_app is not None, "loading": face_app_loading}


@app.get("/health")
def health():
    return {"status": "healthy", "biometric": face_app is not None, "loading": face_app_loading}


@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...)):
    if not SERP_API_KEY:
        raise HTTPException(status_code=500, detail="SERP_API_KEY not configured")

    image_bytes = await file.read()
    query_embedding = get_embedding(image_bytes)
    biometric_enabled = query_embedding is not None

    async with httpx.AsyncClient(timeout=30) as client:
        serp_resp = await client.post(
            "https://serpapi.com/search",
            params={"engine": "google_lens", "api_key": SERP_API_KEY},
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
        )
        if serp_resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"SerpAPI error {serp_resp.status_code}: {serp_resp.text[:300] if serp_resp.text else 'empty response'}")
        serp_data = serp_resp.json()

    visual_matches = serp_data.get("visual_matches", [])
    results = []

    async with httpx.AsyncClient(timeout=15) as client:
        for match in visual_matches[:40]:
            source_url = match.get("link", "")
            image_url = match.get("thumbnail", match.get("image", ""))
            platform = detect_platform(source_url)
            if not platform or not image_url:
                continue

            if biometric_enabled:
                try:
                    img_resp = await client.get(image_url, follow_redirects=True)
                    if img_resp.status_code == 200:
                        result_embedding = get_embedding(img_resp.content)
                        if result_embedding is not None:
                            similarity = cosine_similarity(query_embedding, result_embedding)
                            match_pct = int(max(0, min(100, similarity * 100)))
                        else:
                            continue
                    else:
                        continue
                except Exception:
                    continue
            else:
                position = match.get("position", 30)
                match_pct = max(65, int(98 - (position * 1.5)))

            if match_pct >= 65:
                results.append(SearchResult(
                    platform=platform,
                    imageUrl=image_url,
                    profileUrl=source_url,
                    matchPercentage=match_pct,
                ))

    seen = set()
    unique_results = []
    for r in sorted(results, key=lambda x: x.matchPercentage, reverse=True):
        if r.profileUrl not in seen:
            seen.add(r.profileUrl)
            unique_results.append(r)

    return SearchResponse(results=unique_results, biometricEnabled=biometric_enabled)
