import os
import json
import hashlib
import io
import itertools
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import google.api_core.exceptions

import cloudinary
import cloudinary.uploader
from PIL import Image

load_dotenv()


class KeyManager:
    def __init__(self, keys: List[str]):
        if not keys or all(k == '' for k in keys):
            raise ValueError("Gemini API key list cannot be empty.")
        self.keys = keys
        self._key_iterator = itertools.cycle(keys)
        self._current_key = next(self._key_iterator)
        print(f"KeyManager ready with {len(self.keys)} key(s).")

    def get_current_key(self) -> str:
        return self._current_key

    def switch_to_next_key(self) -> str:
        self._current_key = next(self._key_iterator)
        print(f"Quota exceeded. Switching to next key: ...{self._current_key[-4:]}")
        return self._current_key


gemini_api_keys_str = os.getenv("GEMINI_API_KEYS")
if not gemini_api_keys_str:
    raise ValueError("GEMINI_API_KEYS environment variable is missing.")
gemini_api_keys = [key.strip() for key in gemini_api_keys_str.split(',') if key.strip()]
key_manager = KeyManager(gemini_api_keys)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

limiter = Limiter(key_func=get_remote_address, default_limits=["15/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Cache ready. GblackAI v8.5 started.")
    yield


app = FastAPI(
    title="GblackAI - AI Analysis Microservice",
    description="API v8.5. Reference version — universal prompt, Gemini key rotation.",
    version="8.5.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class BoundingBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)

class SolutionDetail(BaseModel):
    solution: str
    details: str
    source: Optional[str] = None

class RecommendationsGroup(BaseModel):
    biological: List[SolutionDetail]
    chemical: List[SolutionDetail]
    cultural: List[SolutionDetail]

class DetailedInfo(BaseModel):
    description: str
    impact: str
    recommendations: RecommendationsGroup
    knowledgeBaseTags: List[str]

class Detection(BaseModel):
    className: str
    confidenceScore: float
    severity: SeverityLevel
    boundingBox: BoundingBox
    details: DetailedInfo
    croppedImageUrl: Optional[str] = Field(None)

class AnalysisSubject(BaseModel):
    subjectType: str
    description: str
    confidence: float

class AIAnalysisResponse(BaseModel):
    subject: AnalysisSubject
    detections: List[Detection]


UNIVERSAL_PROMPT = """
You are 'GblackAI-Core', a world-class agricultural image analysis engine.
Your only task is to receive an image and return a complete expert analysis.

Identify the subject ('PLANT', 'PEST', or 'UNKNOWN'), each detected problem,
its severity ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'), and provide grouped recommendations.

Respond ONLY with JSON using this schema:
{
  "subject": { "subjectType": "string", "description": "string", "confidence": "float" },
  "detections": [
    {
      "className": "string",
      "confidenceScore": "float",
      "severity": "string",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string",
        "impact": "string",
        "recommendations": {
          "biological": [ { "solution": "string", "details": "string", "source": "string|null" } ],
          "chemical":   [ { "solution": "string", "details": "string", "source": "string|null" } ],
          "cultural":   [ { "solution": "string", "details": "string", "source": "string|null" } ]
        },
        "knowledgeBaseTags": ["string"]
      }
    }
  ]
}

LANGUAGE: All text responses in FRENCH.
"""


async def generate_gemini_analysis_with_key_rotation(image_part: dict, config: genai.types.GenerationConfig):
    initial_key = key_manager.get_current_key()
    for _ in range(len(key_manager.keys)):
        try:
            current_key = key_manager.get_current_key()
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel('gemini-3-flash-preview')
            response = await model.generate_content_async(
                [UNIVERSAL_PROMPT, image_part],
                generation_config=config,
                request_options={'timeout': 120}
            )
            return response
        except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.PermissionDenied) as e:
            print(f"Quota error for key ...{current_key[-4:]}: {e}")
            key_manager.switch_to_next_key()
            if key_manager.get_current_key() == initial_key:
                raise HTTPException(status_code=429, detail="All Gemini API keys have exceeded their quota.")
    raise HTTPException(status_code=503, detail="AI analysis failed after rotating all available keys.")


def image_key_builder(func, namespace: str = "", *, request: Request, response: Response, **kwargs):
    file_content = kwargs["file"].file.read()
    kwargs["file"].file.seek(0)
    file_hash = hashlib.sha256(file_content).hexdigest()
    return f"{namespace}:{file_hash}"


@app.post(
    "/api/v8/analyze-image",
    response_model=AIAnalysisResponse,
    summary="Universal AI analysis v8.5 (reference version)",
    tags=["GblackAI v8.5 (reference)"],
)
@limiter.limit("15/minute")
@cache(namespace="gblackai-v8", expire=86400, key_builder=image_key_builder)
async def analyze_image_endpoint(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="Image file (JPEG or PNG)."),
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Unsupported format. Use JPEG or PNG.")

    image_bytes = await file.read()

    try:
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        gemini_response = await generate_gemini_analysis_with_key_rotation(image_part, generation_config)
        analysis_data = json.loads(gemini_response.text)

        if analysis_data.get("detections"):
            original_image = Image.open(io.BytesIO(image_bytes))
            width, height = original_image.size
            for detection in analysis_data["detections"]:
                detection.setdefault("croppedImageUrl", None)
                bbox = detection.get("boundingBox")
                if not bbox:
                    continue
                try:
                    coords = (
                        int(bbox["x_min"] * width), int(bbox["y_min"] * height),
                        int(bbox["x_max"] * width), int(bbox["y_max"] * height),
                    )
                    if coords[0] >= coords[2] or coords[1] >= coords[3]:
                        continue
                    cropped = original_image.crop(coords)
                    buffer = io.BytesIO()
                    cropped.save(buffer, format="PNG")
                    buffer.seek(0)
                    result = cloudinary.uploader.upload(buffer, folder="gblackai_detections")
                    detection["croppedImageUrl"] = result.get("secure_url")
                except Exception as e:
                    print(f"Crop warning for '{detection.get('className', '?')}': {e}")

        return analysis_data

    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="AI model returned a non-JSON response.")
    except google.api_core.exceptions.GoogleAPICallError as e:
        raise HTTPException(status_code=503, detail=f"Google API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/", include_in_schema=False)
def read_root():
    return {"status": "ok", "service": "GblackAI v8.5 (reference)", "model": "gemini-3-flash-preview"}
