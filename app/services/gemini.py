import logging

import google.api_core.exceptions
import google.generativeai as genai
from fastapi import HTTPException

from app.config import GEMINI_MODEL
from app.key_manager import KeyManager

logger = logging.getLogger(__name__)


async def call_gemini(
    prompt: str,
    image_part: dict | None,
    config: genai.types.GenerationConfig,
    key_manager: KeyManager,
) -> str:
    """
    Calls the Gemini API with automatic key rotation on quota errors.
    Returns the raw response text (JSON string).

    Raises:
        HTTPException 429: all keys have hit their quota.
        HTTPException 503: failed after full key rotation.
    """
    initial_key = key_manager.get_current_key()

    for _ in range(len(key_manager.keys)):
        current_key = key_manager.get_current_key()
        try:
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            contents = [prompt, image_part] if image_part else [prompt]
            response = await model.generate_content_async(
                contents,
                generation_config=config,
                request_options={"timeout": 120},
            )
            return response.text

        except (
            google.api_core.exceptions.ResourceExhausted,
            google.api_core.exceptions.PermissionDenied,
        ) as e:
            logger.warning(f"Quota/permission error on key ...{current_key[-4:]}: {e}")
            await key_manager.rotate()
            if key_manager.get_current_key() == initial_key:
                logger.error("All Gemini API keys have exceeded their quota.")
                raise HTTPException(
                    status_code=429,
                    detail="All Gemini API keys are unavailable or over quota.",
                )

    raise HTTPException(
        status_code=503,
        detail="AI analysis failed after rotating all available keys.",
    )
