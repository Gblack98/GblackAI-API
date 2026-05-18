import asyncio
import hashlib
import json
from contextlib import asynccontextmanager

import cloudinary
import google.api_core.exceptions
import google.generativeai as genai
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import (
    CACHE_TTL,
    CLOUDINARY_API_KEY_ENV,
    CLOUDINARY_API_SECRET,
    CLOUDINARY_CLOUD_NAME,
    GEMINI_API_KEYS,
    GEMINI_MODEL,
    RATE_LIMIT,
)
from app.key_manager import KeyManager
from app.models import AIAnalysisResponse, AnalysisType
from app.prompts import DRONE_PROMPT, PLANT_PEST_PROMPT, SATELLITE_PROMPT
from app.services.cloudinary import crop_and_upload
from app.services.gemini import call_gemini

key_manager = KeyManager(GEMINI_API_KEYS)

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY_ENV,
    api_secret=CLOUDINARY_API_SECRET,
)

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

PROMPTS = {
    AnalysisType.PLANT_PEST: PLANT_PEST_PROMPT,
    AnalysisType.SATELLITE_REMOTE_SENSING: SATELLITE_PROMPT,
    AnalysisType.DRONE_ANALYSIS: DRONE_PROMPT,
}

SUPPORTED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/tiff"}
CROP_SUPPORTED_TYPES = {"image/jpeg", "image/jpg", "image/png"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend(), prefix="gblackai-cache")
    print("Cache ready. GblackAI v12.0.0 started.")
    yield


app = FastAPI(
    title="GblackAI - Unified Analysis Microservice",
    description="API v12.0 — AI-powered crop analysis (Plant, Satellite, Drone). Powered by Gemini 3 Flash Preview.",
    version="12.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def unified_key_builder(func, namespace: str = "", *, request: Request, response: Response, **kwargs):
    analysis_type_str = str(kwargs.get("analysis_type", "unknown"))
    file: UploadFile = kwargs["file"]
    content = file.file.read()
    file.file.seek(0)
    file_hash = hashlib.sha256(content).hexdigest()
    return f"{namespace}:{analysis_type_str}:{file_hash}"


@app.post(
    "/api/v12/analyze",
    response_model=AIAnalysisResponse,
    summary="Unified analysis v12 — Plant / Satellite / Drone",
    tags=["GblackAI v12"],
)
@limiter.limit(RATE_LIMIT)
@cache(namespace="gblackai-v12", expire=CACHE_TTL, key_builder=unified_key_builder)
async def analyze(
    request: Request,
    response: Response,
    analysis_type: AnalysisType = Form(
        ...,
        description="Analysis type: PLANT_PEST, SATELLITE_REMOTE_SENSING, or DRONE_ANALYSIS.",
    ),
    file: UploadFile = File(
        ...,
        description="Image file: JPEG, JPG, PNG, or TIFF.",
    ),
):
    """
    Full pipeline:
    1. Select the expert prompt based on analysis type.
    2. Call Gemini 3 Flash with automatic key rotation.
    3. Crop detected zones and upload to Cloudinary (parallel).
    4. Return Pydantic-validated JSON.
    """
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format. Accepted formats: {', '.join(SUPPORTED_TYPES)}.",
        )

    image_bytes = await file.read()
    prompt = PROMPTS[analysis_type]
    image_part = {"mime_type": file.content_type, "data": image_bytes}
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

    try:
        raw_text = await call_gemini(prompt, image_part, generation_config, key_manager)
        analysis_data = json.loads(raw_text)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="The AI model returned a non-JSON response.",
        )
    except google.api_core.exceptions.GoogleAPICallError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Google API error: {e.message}",
        )

    if analysis_data.get("detections") and file.content_type in CROP_SUPPORTED_TYPES:
        loop = asyncio.get_event_loop()
        analysis_data["detections"] = await loop.run_in_executor(
            None, crop_and_upload, image_bytes, analysis_data["detections"], "gblackai_v12"
        )

    return analysis_data


@app.get("/", include_in_schema=False)
def health():
    return {"status": "ok", "service": "GblackAI v12.0.0", "model": GEMINI_MODEL}
