# 1-video-rag

Video RAG ingests a YouTube video, extracts periodic frames, describes them with a vision model, stores the descriptions in ChromaDB, and answers user questions using the most relevant frames.

## What is included

- `src/run.py`: CLI interface that ingests a YouTube URL and opens a text Q&A loop.
- `src/streamlit_app.py`: Streamlit app for ingesting a video and asking questions through a browser UI.
- `src/ingest.py`: downloads video with `yt-dlp`, extracts frames with OpenCV, generates vision descriptions, embeds text, and stores vectors in ChromaDB.
- `src/query.py`: embeds user questions, retrieves top frames from ChromaDB, and asks the vision model to answer using the selected images.
- `src/config.py`: configuration for API keys, models, frame extraction, and storage paths.

## Requirements

- Python 3.14 or later
- Dependencies listed in `pyproject.toml`
  - `chromadb`
  - `google-genai`
  - `opencv-python-headless`
  - `openai`
  - `python-dotenv`
  - `streamlit`
  - `yt-dlp`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install chromadb google-genai opencv-python-headless openai python-dotenv streamlit yt-dlp
```

3. Provide your Gemini API key in an environment variable or `.env` file:

```bash
export GEMINI_API_KEY="your_key"
```

## Usage

### CLI

Run the ingestion and interactive question loop from the terminal:

```bash
python src/run.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### Streamlit

Run the browser app:

```bash
streamlit run src/streamlit_app.py
```

Then paste your API key and YouTube URL, ingest the video, and ask questions.

## Notes

- Ingestion downloads videos to `videos/` and saves frames to `frames/`.
- The default vision provider is Gemini (`gemini-2.5-flash`) and default embedding model is `gemini-embedding-001`.
- The system retrieves the top 3 relevant frames for each question.



