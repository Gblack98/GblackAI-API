"""
GblackAI API v12 tests.
Gemini and Cloudinary calls are mocked — no real keys needed.
"""
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image_bytes(fmt: str = "JPEG") -> bytes:
    """Generate a 100x100 RGB image in memory."""
    img = Image.new("RGB", (100, 100), color=(80, 160, 80))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


MOCK_RESPONSE = {
    "subject": {
        "subjectType": "PLANT",
        "description": "Maize plant (Zea mays)",
        "confidence": 0.96,
    },
    "detections": [
        {
            "className": "Common rust",
            "confidenceScore": 0.87,
            "severity": "HIGH",
            "boundingBox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.6, "y_max": 0.6},
            "croppedImageUrl": None,
            "details": {
                "description": "Orange spots on leaves.",
                "impact": "Yield will decrease.",
                "recommendations": {
                    "biological": [],
                    "chemical": [{"solution": "Fungicide", "details": "Apply early.", "source": None}],
                    "cultural": [],
                },
                "knowledgeBaseTags": ["rust", "maize", "fungus"],
            },
        }
    ],
}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self):
        res = client.get("/")
        assert res.status_code == 200

    def test_response_contains_service_info(self):
        res = client.get("/")
        data = res.json()
        assert "GblackAI" in data["service"]
        assert data["status"] == "ok"
        assert "model" in data


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_unsupported_mime_type_returns_415(self):
        res = client.post(
            "/api/v12/analyze",
            data={"analysis_type": "PLANT_PEST"},
            files={"file": ("test.gif", b"GIF89a", "image/gif")},
        )
        assert res.status_code == 415

    def test_invalid_analysis_type_returns_422(self):
        res = client.post(
            "/api/v12/analyze",
            data={"analysis_type": "UNKNOWN_TYPE"},
            files={"file": ("test.jpg", make_image_bytes(), "image/jpeg")},
        )
        assert res.status_code == 422

    def test_missing_file_returns_422(self):
        res = client.post(
            "/api/v12/analyze",
            data={"analysis_type": "PLANT_PEST"},
        )
        assert res.status_code == 422

    def test_missing_analysis_type_returns_422(self):
        res = client.post(
            "/api/v12/analyze",
            files={"file": ("test.jpg", make_image_bytes(), "image/jpeg")},
        )
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Successful analysis
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    @patch("main.call_gemini", new_callable=AsyncMock)
    @patch("main.crop_and_upload")
    def test_plant_pest_analysis_returns_200(self, mock_crop, mock_gemini):
        mock_gemini.return_value = json.dumps(MOCK_RESPONSE)
        mock_crop.return_value = MOCK_RESPONSE["detections"]

        res = client.post(
            "/api/v12/analyze",
            data={"analysis_type": "PLANT_PEST"},
            files={"file": ("plant.jpg", make_image_bytes(), "image/jpeg")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["subject"]["subjectType"] == "PLANT"
        assert len(data["detections"]) == 1
        assert data["detections"][0]["severity"] == "HIGH"

    @patch("main.call_gemini", new_callable=AsyncMock)
    @patch("main.crop_and_upload")
    def test_satellite_analysis_returns_200(self, mock_crop, mock_gemini):
        satellite_resp = {
            **MOCK_RESPONSE,
            "subject": {**MOCK_RESPONSE["subject"], "subjectType": "SATELLITE_PLOT"},
        }
        mock_gemini.return_value = json.dumps(satellite_resp)
        mock_crop.return_value = satellite_resp["detections"]

        res = client.post(
            "/api/v12/analyze",
            data={"analysis_type": "SATELLITE_REMOTE_SENSING"},
            files={"file": ("sat.png", make_image_bytes("PNG"), "image/png")},
        )
        assert res.status_code == 200

    @patch("main.call_gemini", new_callable=AsyncMock)
    @patch("main.crop_and_upload")
    def test_drone_analysis_returns_200(self, mock_crop, mock_gemini):
        drone_resp = {
            **MOCK_RESPONSE,
            "subject": {**MOCK_RESPONSE["subject"], "subjectType": "DRONE_PLOT"},
        }
        mock_gemini.return_value = json.dumps(drone_resp)
        mock_crop.return_value = drone_resp["detections"]

        res = client.post(
            "/api/v12/analyze",
            data={"analysis_type": "DRONE_ANALYSIS"},
            files={"file": ("drone.jpg", make_image_bytes(), "image/jpeg")},
        )
        assert res.status_code == 200

    @patch("main.call_gemini", new_callable=AsyncMock)
    def test_invalid_json_from_gemini_returns_502(self, mock_gemini):
        mock_gemini.return_value = "not valid json"

        res = client.post(
            "/api/v12/analyze",
            data={"analysis_type": "PLANT_PEST"},
            files={"file": ("plant.jpg", make_image_bytes(), "image/jpeg")},
        )
        assert res.status_code == 502


# ---------------------------------------------------------------------------
# KeyManager unit tests
# ---------------------------------------------------------------------------

class TestKeyManager:
    def test_initial_key_is_set(self):
        from app.key_manager import KeyManager
        km = KeyManager(["key_a", "key_b"])
        assert km.get_current_key() == "key_a"

    def test_empty_keys_raises_error(self):
        from app.key_manager import KeyManager
        with pytest.raises(ValueError):
            KeyManager([])

    @pytest.mark.asyncio
    async def test_rotate_cycles_keys(self):
        from app.key_manager import KeyManager
        km = KeyManager(["key_a", "key_b"])
        await km.rotate()
        assert km.get_current_key() == "key_b"
        await km.rotate()
        assert km.get_current_key() == "key_a"
