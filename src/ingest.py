"""
It runs once per video and builds the entire searchable index.
"""

import os
from pathlib import Path
import time
import cv2
import yt_dlp
from google import genai
from google.genai import types
from openai import OpenAI
import config
import logging
import base64
logger = logging.getLogger(__name__)


os.environ["CHROMA_TELEMETRY"] = "false" # Disable Chroma telemetry -> ChromaDB’s usage/analytics reporting
import chromadb

def download_video(url: str) -> Path:
    """
    Download a youtube video and return the local file path.
    """
    config.VIDEO_DIR.mkdir(exist_ok=True)

    ydl_opts = {
        'format': config.VIDEO_FORMAT,
        'outtmpl': str(config.VIDEO_DIR / '%(id)s.%(ext)s'), # Output filename template - writes files into config.VIDEO_DIR with a name like Ksdf3A1.mp4 (video id + extension).
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = config.VIDEO_DIR / f"{info_dict['id']}.{info_dict.get('ext', 'mp4')}"
        return video_path
    
def extract_frame(video_path: Path) -> list[dict]:
    """
    Extract one JPEG frame every FRAME_INTERVAL seconds from the video.
    """

    # Initialize video capture object from the video file
    cap = cv2.VideoCapture(str(video_path))

    # Verify that the video file was opened successfully
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Make sure the frame output directory exists before writing any files
    config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate video duration in seconds, handling division by zero if fps is 0
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video duration: {duration:.2f} seconds at FPS: {fps:.2f}")

     # Ensure at least 1 frame is extracted
    frame_interval = max(1, int(fps * config.FRAME_INTEREVAL_SECONDS))  

    extracted = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # if frame_idx % frame_interval == 0: when the current frame index is a multiple of 
        # frame_interval, treat it as an extraction point
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps # compute the time (in seconds) of the extracted frame
            frame_path = config.FRAMES_DIR / f"frame_{frame_idx:06d}.jpg"
            success = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            if not success:
                raise RuntimeError(f"Failed to write frame image: {frame_path}")
            extracted.append({
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "path": frame_path,
            })

        frame_idx += 1
    
    cap.release()
    logger.info(f"Extracted {len(extracted)} frames from video.")
    return extracted


def describe_frame(client, frame_path: Path, model_name: str, provider: str) -> str:
    """
    Send a frame to Gemini or OpenAI Vision and get a detailed text description.
    """

    with open(frame_path, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "Describe this video frame in detail.\n"
        "Include:\n"
        "- any text visible on screen such as whiteboards,\n"
        "- slides, terminals, or code,\n"
        "- any diagrams, charts, or visual elements,\n"
        "- what is happening in the scene,\n"
        "- any tools or interfaces visible.\n"
        "Be specific and thorough. Write 3-5 sentences."
    )

    if provider == "gemini":
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ]
        )
        return response.text.strip()
    elif provider == "openai":
        image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown provider: {provider}")

def embed_text(client, text: str, provider: str = "gemini") -> list[float]:
    """
    Convert a text description into an embedding vector.
    Uses Gemini's embedding model or OpenAI's embedding API.
    """
    if provider == "gemini":
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=config.EMBEDDING_DIM,
            )
        )
        return response.embeddings[0].values
    elif provider == "openai":
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        return response.data[0].embedding
    else:
        raise ValueError(f"Unknown provider: {provider}")

def _format_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to MM:SS display format.
     smallhelper that turns 90.0 into "01:30" for showing timestamps in the UI."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def ingest(url: str, vision_model: str | None = None, provider: str = "gemini", api_key: str | None = None) -> tuple:
    """
    Full ingestion pipeline:
    download video -> extract frames -> describe frames with Vision API -> embed descriptions -> store in ChromaDB

    Args:
        url: YouTube video URL
        vision_model: Model name for vision API (e.g., 'gemini-2.5-flash' or 'gpt-4o')
        provider: 'gemini' or 'openai'
        api_key: API key for the chosen provider

    return (client, chroma_collection, provider) for reuse in querying
    """

    vision_model = vision_model or config.VISION_MODEL
    
    if provider == "gemini":
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or config.GEMINI_API_KEY)
    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key required for OpenAI provider")
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # --- Step 1: Download video ---
    logger.info("Downloading video...")
    video_path = download_video(url)
    logger.info(f"Video downloaded to: {video_path}")

    # --- Step 2: Extract frames ---
    logger.info(f"\n Extracting frames (1 per {config.FRAME_INTEREVAL_SECONDS}s)...")
    frames = extract_frame(video_path)

    # --- Step 3, 4, 5 : Describe -> Embed -> Store (one frame at a time) ---
    logger.info("\nDescribing and indexing frames...")

    db = chromadb.Client()
    collection = db.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}, # Use cosine similarity for vector search, which is common for text embeddings
    )

    for i, frame in enumerate(frames):
        timeestamp_str = _format_timestamp(frame["timestamp"])
        logger.info(f"[{i+1}/{len(frames)}] t ={timeestamp_str} - Describing frame {frame['frame_index']}...", end=" ", flush=True)
        description = describe_frame(client, frame["path"], vision_model, provider)
        vector = embed_text(client, description, provider)

        collection.add(
            ids=[f"frame_{frame['frame_index']:06d}"], # Unique ID for this frame, e.g. "frame_000150"
            embeddings=[vector], # The embedding vector for this frame's description
            metadatas=[{
                "timestamp": frame["timestamp"],
                "timestamp_str": timeestamp_str,
                "frame_path": str(frame["path"]),
                "frame_index": frame["frame_index"],
                "description": description,
            }],
        )

        logger.info("done.")
        # Free tier limit: 15 RPM across all Gemini calls.
        # Each frame costs 2 calls (describe + embed).
        # Sleeping 2s per frame keeps us well under the limit.
        if i < len(frames) - 1:
            time.sleep(2)

    logger.info(f"\n Indexed {len(frames)} frames into ChromaDB")
    return client, collection, provider