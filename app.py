import streamlit as st
import os
import yt_dlp
import whisper
from transformers import pipeline
import imageio_ffmpeg

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
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

whisper_model = load_whisper()
summarizer = load_summarizer()

# -------------------------------
# Helper functions
# -------------------------------
def download_audio(youtube_url, filename="audio.mp3"):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
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
        'ffmpeg_location': ffmpeg_path
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return filename

def transcribe_audio(audio_file):
    return whisper_model.transcribe(audio_file)["text"]

def chunk_text(text, max_chunk=2000):
    """Split text into chunks safely without breaking sentences abruptly"""
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_chunks(chunks):
    summaries = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
        progress_bar.progress((i + 1) / len(chunks))
    return " ".join(summaries)

def format_summary_pointwise(summary_text):
    points = summary_text.split(". ")
    formatted = "\n".join([f"• {point.strip()}" for point in points if point.strip()])
    return formatted

# -------------------------------
# Streamlit UI
# -------------------------------
url = st.text_input("🔗 Enter YouTube URL here:")

if st.button("📝 Summarize"):
    if url:
        try:
            with st.spinner("⏳ Downloading audio..."):
                audio_file = download_audio(url)

            with st.spinner("⏳ Transcribing audio..."):
                transcript_text = transcribe_audio(audio_file)

            with st.spinner("⏳ Splitting transcript into chunks..."):
                chunks = chunk_text(transcript_text)

            with st.spinner("⏳ Summarizing transcript..."):
                summary_text = summarize_chunks(chunks)

            formatted_summary = format_summary_pointwise(summary_text)

            st.markdown("### 📝 Summary (Point-wise)")
            st.success(formatted_summary)
            st.balloons()

            os.remove(audio_file)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
    else:
        st.warning("⚠️ Please enter a valid YouTube URL.")
