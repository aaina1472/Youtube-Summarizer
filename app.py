import streamlit as st
import os
import yt_dlp
import whisper
from transformers import pipeline

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="YouTube Video Summarizer 🎬",
    page_icon="🎬",
    layout="wide",
)

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 YouTube Video Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Paste any YouTube video link and get a concise summary in seconds!</p>", unsafe_allow_html=True)
st.write("---")

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")  # smaller model for speed

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

whisper_model = load_whisper()
summarizer = load_summarizer()

# -------------------------------
# Helper functions
# -------------------------------
def download_audio(youtube_url, filename="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return filename

def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result["text"]

def chunk_text(text, max_chunk=1000):
    return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

def recursive_summarize(text):
    chunks = chunk_text(text, max_chunk=2000)
    summaries = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
        progress_bar.progress((i + 1) / len(chunks))
    combined_summary = " ".join(summaries)
    if len(combined_summary) > 2000:
        return recursive_summarize(combined_summary)
    return combined_summary

# -------------------------------
# Streamlit UI
# -------------------------------
url = st.text_input("🔗 Enter YouTube URL here:")

if url:
    try:
        with st.spinner("⏳ Downloading audio..."):
            audio_file = download_audio(url)
        with st.spinner("⏳ Transcribing audio..."):
            transcript_text = transcribe_audio(audio_file)
        with st.spinner("⏳ Summarizing transcript..."):
            summary_text = recursive_summarize(transcript_text)
        st.markdown("### 📝 Summary")
        st.success(summary_text)
        st.balloons()
        # Remove audio file after processing
        os.remove(audio_file)
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
