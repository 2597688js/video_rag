"""
The Q & A engine that answers questions about the video content. It uses the vector database to find relevant frames and their descriptions, and then generates an answer using Gemini Pro or OpenAI.
"""
import os
import logging
from pathlib import Path    
from google import genai
from google.genai import types
from openai import OpenAI
import config   
import base64
logger = logging.getLogger(__name__)

os.environ["CHROMA_TELEMETRY"] = "false" # Disable Chroma telemetry -> ChromaDB’s usage/analytics reporting
import chromadb

def embed_query(client, query: str, provider: str = "gemini") -> list[float]:
    """
    Convert the user query into an embedding vector using Gemini or OpenAI.
    """
    if provider == "gemini":
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=config.EMBEDDING_DIM,
            ),
        )
        return response.embeddings[0].values
    elif provider == "openai":
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=query,
        )
        return response.data[0].embedding
    else:
        raise ValueError(f"Unknown provider: {provider}")

def retrieve_frames(
        collection: chromadb.Collection,
        query_vector: list[float],
) -> list[dict]:
    """
    Search ChromaDB for the frames most semantically similar to the question."""
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=config.TOP_K,
        include=["documents", "metadatas", "distances"], # Return the metadata we stored for each frame, which includes the description and path
    )

    frames = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        frames.append({
            "description": doc, # The Gemini Vision description of the frame, which we stored as the "document" in ChromaDB
            "timestamp_str": meta["timestamp_str"], # The timestamp of the frame in MM:SS format, stored in the metadata
            "frame_path": meta["frame_path"], # The path to the frame image file, stored in the metadata
            "similarity": 1 - dist, # The distance between the query vector and the frame's embedding vector - a measure of relevance
        })

    # Re-sort by timestamp before sending to Gemini so the model sees them in chronological order, helping it understand the video’s progression and give a more coherent answer.
    frames.sort(key=lambda x: x["timestamp_str"]) # Sort the frames by their timestamp in the video for better readability in the UI
    return frames

def answer_query(client, query: str, frames: list[dict], model_name: str | None = None, provider: str = "gemini") -> str:
    """
    Send the user query and the relevant frames to Gemini or OpenAI and get a visual answer.
    """

    contents = []
    context_prompt = (
        f"You are answering a question about a video. "
        f"You have been given {len(frames)} frames retrieved "
        f"from the video "
        f"that are most relevant to the question. "
        f"Each frame is labeled with its timestamp. "
        f"Answer the question based on what you can see in these frames."
    )

    if provider == "gemini":
        contents.append(types.Part.from_text(text=context_prompt))
        for frame in frames:
            contents.append(
                types.Part.from_text(text=f"\n[Frame at {frame['timestamp_str']}]")
            )
            with open(frame["frame_path"], "rb") as f:
                img_bytes = f.read()
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )
        contents.append(types.Part.from_text(text=f"\n Question: {query}"))

        response = client.models.generate_content(
            model=model_name or config.VISION_MODEL,
            contents=contents,
        )
        return response.text.strip()
    
    elif provider == "openai":
        message_content = [
            {"type": "text", "text": context_prompt}
        ]
        for frame in frames:
            message_content.append(
                {"type": "text", "text": f"\n[Frame at {frame['timestamp_str']}]"}
            )
            with open(frame["frame_path"], "rb") as f:
                img_bytes = f.read()
            image_base64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )
        message_content.append(
            {"type": "text", "text": f"\nQuestion: {query}"}
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": message_content}
            ],
        )
        return response.choices[0].message.content.strip()
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def ask(
    client,
    collection: chromadb.Collection,
    question: str,
    model_name: str | None = None,
    provider: str = "gemini",
) -> str:
    """
    Single entry point for the CLI: embed question -> retrieve frames -> answer.
    Hides the three-step pipeline behind one clean call.
    """
    query_vector = embed_query(client, question, provider)
    frames = retrieve_frames(collection, query_vector)

    if not frames:
        return "No relevant frames found for that question."

    return answer_query(client, question, frames, model_name, provider)




