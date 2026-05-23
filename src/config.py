import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API ---
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# --- MODELS ---
VISION_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIM: int = 3072

# --- Frame extraction ---
FRAME_INTEREVAL_SECONDS: int = 30
JPEG_QUALITY: int = 85 # 0-100, higher is better quality but larger file size
VIDEO_FORMAT: str = "bestvideo[height<=720][ext=mp4]/best[height<=720]" # Use best video format with height <= 720p in MP4 format, fallback to best format if not available

# --- Vector store ---
COLLECTION_NAME: str = "video_frames"
TOP_K: int = 3 # retrieves the 3 most relevant frames from the vector store for each query

# --- Paths ---
FRAMES_DIR: Path = Path("frames")
VIDEO_DIR: Path = Path("videos")