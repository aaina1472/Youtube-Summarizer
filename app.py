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

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 YouTube Video Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Paste any YouTube video link and get a concise summary!</p>", unsafe_allow_html=True)
st.write("---")


st.markdown(
    "⚠️ **Note:** For best performance, use videos up to ~5 minutes. Longer videos may exceed memory limits on Streamlit Cloud.", 
    unsafe_allow_html=True
)

# -------------------------------
# Session state: store last summary and audio file
# -------------------------------
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""   # stores the most recent formatted summary
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None  # stores path to last downloaded audio

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel("base")  # faster-whisper local model

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

def format_summary_pointwise(summary_text):
    import re
    # Split by sentence endings (., !, ?)
    points = re.split(r'(?<=[.!?]) +', summary_text)
    formatted = "\n".join([f"• {point.strip()}" for point in points if point.strip()])
    return formatted


# -------------------------------
# Streamlit UI
# -------------------------------
url = st.text_input("🔗 Enter YouTube URL here:")

if st.button("📝 Summarize"):
    if url:
        try:
            # Clear previous summary and delete previous audio file if present
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

            # Save only the latest summary to session state
            st.session_state.last_summary = formatted_summary

            # Display only the latest summary
            st.markdown("### 📝 Summary (Point-wise)")
            st.success(st.session_state.last_summary)
            st.balloons()

            # Clean up audio file after processing
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

# If a summary exists in session state (from previous run), show it (this ensures a single persistent result)
if st.session_state.last_summary:
    st.success(st.session_state.last_summary)
