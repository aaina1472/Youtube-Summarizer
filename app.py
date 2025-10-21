import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
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
# Load summarizer
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # CPU
    )

summarizer = load_summarizer()

# -------------------------------
# Helper functions
# -------------------------------
def get_transcript(video_url):
    """Fetch transcript (manual or auto-generated) for a YouTube video."""
    try:
        # Extract video ID
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1].split("?")[0]
        else:
            st.error("❌ Invalid YouTube URL format.")
            return None

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_transcript(['en']).fetch()
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(['en']).fetch()

        transcript_text = " ".join([t['text'] for t in transcript])
        return transcript_text

    except TranscriptsDisabled:
        st.error("❌ This video does not have transcripts enabled.")
        return None
    except NoTranscriptFound:
        st.error("❌ No transcript found for this video in English.")
        return None
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
        return None

def chunk_text(text, max_chunk=1000):
    """Split text into chunks."""
    return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

def recursive_summarize(text):
    """Summarize text recursively for long transcripts."""
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
