import os
import streamlit as st
import config
from ingest import ingest
from query import ask

st.set_page_config(
    page_title="Video RAG - Ask Questions About a YouTube Video",
    page_icon="🎬",
    layout="centered",
)

st.title("Video RAG - Ask Questions About a YouTube Video")
st.markdown(
    "Paste your Gemini or OpenAI API key and a YouTube URL, ingest the video, then ask questions about what appears in the video."
)
st.info("This app is currently best for short videos, probably less than 10 minutes.")

if "client" not in st.session_state:
    st.session_state.client = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "provider" not in st.session_state:
    st.session_state.provider = "gemini"
if "model_choice" not in st.session_state:
    st.session_state.model_choice = config.VISION_MODEL
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

st.subheader("Step 1: Select Provider")
provider = st.selectbox(
    "Choose AI provider",
    ["gemini", "openai"],
    format_func=lambda x: "Google Gemini" if x == "gemini" else "OpenAI",
    key="provider_select"
)

st.subheader("Step 2: Provide API key and YouTube URL")

with st.form(key="video_form"):
    youtube_url = st.text_input("YouTube video URL")
    
    if provider == "gemini":
        api_key = st.text_input("Gemini API key", type="password", key="gemini_key")
        model_choice = st.selectbox(
            "Gemini model for vision and QA",
            [
                "gemini-2.5-flash"
            ],
            index=0,
            key="gemini_model"
        )
    else:  # openai
        api_key = st.text_input("OpenAI API key", type="password", key="openai_key")
        model_choice = st.selectbox(
            "OpenAI model for vision and QA",
            [
                "gpt-5"
            ],
            index=0,
            key="openai_model"
        )
    
    ingest_button = st.form_submit_button("Ingest video and build index")

if ingest_button:
    if not youtube_url:
        st.error("Please enter a YouTube URL before ingesting.")
    elif not api_key:
        st.error(f"Please enter your {provider.upper()} API key.")
    else:
        try:
            st.session_state.provider = provider
            st.session_state.model_choice = model_choice
            with st.spinner("Downloading, extracting, describing, and indexing the video. This may take some time..."):
                client, collection, returned_provider = ingest(
                    youtube_url,
                    model_choice,
                    provider,
                    api_key
                )
            st.session_state.client = client
            st.session_state.collection = collection
            st.session_state.video_url = youtube_url
            st.success("Ingestion complete! You can now ask questions about the video.")
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

if st.session_state.client and st.session_state.collection:
    st.markdown("---")
    st.subheader("Step 3: Ask a question")

    with st.form(key="question_form"):
        question = st.text_input("Your question about the video")
        ask_button = st.form_submit_button("Ask")

    if ask_button:
        if not question:
            st.error("Please type a question before asking.")
        else:
            try:
                with st.spinner("Asking AI about the video..."):
                    answer = ask(
                        st.session_state.client,
                        st.session_state.collection,
                        question,
                        st.session_state.model_choice,
                        st.session_state.provider,
                    )
                st.session_state.last_question = question
                st.session_state.last_answer = answer
                st.success("Answer received.")
            except Exception as exc:
                st.error(f"Question failed: {exc}")

    if st.session_state.last_question:
        st.markdown("### Conversation")
        st.write(f"**Question:** {st.session_state.last_question}")
        st.write(f"**Answer:** {st.session_state.last_answer}")
else:
    st.warning("Ingest a video first to enable question answering.")

st.markdown("---")
st.caption("Note: Short videos (under 10 minutes) are recommended for faster ingestion and lower cost.")
