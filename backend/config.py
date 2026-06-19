"""
config.py — Central configuration for the Factory Compliance System.
Loads from .env file. Copy .env.example to .env and fill in your values.
"""
from pydantic_settings import BaseSettings # type: ignore
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # --- API Keys ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- MongoDB ---
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "factory_compliance"
    REPORTS_COLLECTION: str = "violation_reports"

    # --- Paths ---
    POLICY_PDF_PATH: str = str(BASE_DIR / "../compliance_policy.pdf")
    VIDEO_INPUT_DIR: str = str(BASE_DIR / "../data/videos")
    REPORTS_OUTPUT_DIR: str = str(BASE_DIR / "../outputs/reports")

    # --- YOLO ---
    # Use 'yolov8n.pt' for speed, 'yolov8m.pt' for better accuracy.
    # Replace with path to your fine-tuned model once trained.
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.40
    FRAME_SAMPLE_INTERVAL: int = 10  # Process every Nth frame for speed

    # --- Detection Zones ---
    # Walkway zone: fraction of frame width [x_min, x_max].
    # Adjust after inspecting your actual camera footage.
    WALKWAY_X_MIN_FRAC: float = 0.10
    WALKWAY_X_MAX_FRAC: float = 0.55
    WALKWAY_Y_MIN_FRAC: float = 0.0
    WALKWAY_Y_MAX_FRAC: float = 1.0

    # Forklift overload threshold
    FORKLIFT_BLOCK_THRESHOLD: int = 3  # 3+ blocks = violation (per Section 6)

    class Config:
        env_file = str(BASE_DIR / "../../.env")
        env_file_encoding = "utf-8"


settings = Settings()
