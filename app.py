import os
import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel
from transformers import pipeline

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="YouTube Video Summarizer 🎬",
    page_icon="🎬",
    layout="wide",
)

# -------------------------------
# Custom CSS for styling
# -------------------------------
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main-title {
            text-align: center;
            color: #ff4b4b;
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 0.2em;
        }
        .sub-text {
            text-align: center;
            font-size: 1.1em;
            color: #555;
            margin-bottom: 1em;
        }
        .note-text {
            text-align: center;
            font-size: 0.95em;
            color: #ff4b4b;
            margin-bottom: 1.5em;
            font-weight: 600;
        }
        .stTextInput>div>div>input {
            border-radius: 12px;
            border: 2px solid #ff4b4b;
            padding: 0.5em;
        }
        .summarize-btn button {
            background-color: #ff4b4b;
            color: white;
            font-weight: 600;
            padding: 0.6em 1.2em;
            border-radius: 10px;
            border: none;
            font-size: 1em;
        }
        .summary-card {
            background-color: white;
            padding: 1.8em;
            border-radius: 15px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            margin-top: 2em;
        }
        .summary-header {
            color: #ff4b4b;
            font-weight: 700;
            font-size: 1.5em;
            margin-bottom: 0.7em;
        }
        .summary-text {
            font-size: 1.15em;
            line-height: 1.6em;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 class='main-title'>🎬 YouTube Video Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Paste a YouTube link and get a short, clean summary instantly.</p>", unsafe_allow_html=True)
st.markdown("<p class='note-text'>⚠️ For best performance, use videos up to ~5 minutes. Longer videos may cause memory issues on Streamlit Cloud.</p>", unsafe_allow_html=True)

# -------------------------------
# Session state
# -------------------------------
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel("base")  # switch to "tiny" if memory is limited

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

whisper_model = load_whisper()
summarizer = load_summarizer()

# -------------------------------
# Helper functions
# -------------------------------
def download_audio(youtube_url, filename="audio.webm"):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": filename,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return filename

def transcribe_audio(audio_file):
    segments, info = whisper_model.transcribe(audio_file)
    text = " ".join([segment.text for segment in segments])
    return text

def chunk_text(text, max_chunk=1000):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chunk:
            cur += (s + " ")
        else:
            chunks.append(cur.strip())
            cur = s + " "
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def recursive_summarize(text):
    chunks = chunk_text(text, max_chunk=2000)
    summaries = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
        progress_bar.progress((i + 1) / len(chunks))
    combined_summary = " ".join(summaries)
    if len(combined_summary) > 3000:
        chunks2 = chunk_text(combined_summary, max_chunk=2000)
        summaries2 = []
        for c in chunks2:
            summaries2.append(summarizer(c, max_length=150, min_length=50, do_sample=False)[0]['summary_text'])
        combined_summary = " ".join(summaries2)
    return combined_summary

def format_summary_pointwise(summary_text):
    import re
    points = re.split(r'(?<=[.!?]) +', summary_text)
    formatted = "\n".join([f"• {point.strip()}" for point in points if point.strip()])
    return formatted

# -------------------------------
# Streamlit UI logic
# -------------------------------
url = st.text_input("🔗 Enter YouTube URL here:")

if st.button("📝 Summarize", key="summarize"):
    if url:
        try:
            st.session_state.last_summary = ""
            if st.session_state.last_audio:
                try:
                    os.remove(st.session_state.last_audio)
                except Exception:
                    pass
                st.session_state.last_audio = None

            with st.spinner("⏳ Downloading audio..."):
                audio_file = download_audio(url)
                st.session_state.last_audio = audio_file

            with st.spinner("⏳ Transcribing audio..."):
                transcript_text = transcribe_audio(audio_file)

            with st.spinner("⏳ Summarizing transcript..."):
                summary_text = recursive_summarize(transcript_text)
                formatted_summary = format_summary_pointwise(summary_text)

            # Prepend "In this video, "
            if formatted_summary:
                final_summary = "In this video, " + formatted_summary[0].lower() + formatted_summary[1:]
            else:
                final_summary = "Summary could not be generated."

            # Display nicely in a card
            st.markdown(f"""
                <div class='summary-card'>
                    <div class='summary-header'>📝 Video Summary</div>
                    <div class='summary-text'>{final_summary}</div>
                </div>
            """, unsafe_allow_html=True)

            st.balloons()

            # Cleanup audio
            if st.session_state.last_audio:
                try:
                    os.remove(st.session_state.last_audio)
                except Exception:
                    pass
                st.session_state.last_audio = None

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
    else:
        st.warning("⚠️ Please enter a valid YouTube URL.")
