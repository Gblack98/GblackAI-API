import os

# Set fake env vars before any app import
os.environ.setdefault("GEMINI_API_KEYS", "test_key_1,test_key_2")
os.environ.setdefault("CLOUDINARY_CLOUD_NAME", "test_cloud")
os.environ.setdefault("CLOUDINARY_API_KEY", "test_api_key")
os.environ.setdefault("CLOUDINARY_API_SECRET", "test_api_secret")
