import streamlit as st
import yt_dlp
import requests
from transformers import pipeline
import tempfile

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="YouTube Video Summarizer 🎬",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 YouTube Video Summarizer")
st.write("Paste any YouTube URL and get a concise summary!")

# -------------------------------
# Load summarizer
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

summarizer = load_summarizer()

# -------------------------------
# Helper functions
# -------------------------------
def download_audio(youtube_url):
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_file.name,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return temp_file.name

def transcribe_with_api(audio_file):
    """
    Uses OpenAI Whisper API to transcribe audio without installing Whisper locally.
    Requires OPENAI_API_KEY set as environment variable.
    """
    import os
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(audio_file, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript['text']

def summarize_text(text):
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

# -------------------------------
# Streamlit UI
# -------------------------------
url = st.text_input("🔗 Enter YouTube URL here:")

if st.button("📝 Summarize"):
    if url:
        try:
            with st.spinner("⏳ Downloading audio..."):
                audio_file = download_audio(url)

            with st.spinner("⏳ Transcribing audio via API..."):
                transcript_text = transcribe_with_api(audio_file)

            with st.spinner("⏳ Summarizing transcript..."):
                summary_text = summarize_text(transcript_text)

            st.markdown("### 📝 Summary")
            st.success(summary_text)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
    else:
        st.warning("⚠️ Please enter a valid YouTube URL.")
