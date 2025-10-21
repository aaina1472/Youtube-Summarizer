import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

# -------------------------------
# Helper Functions
# -------------------------------

@st.cache_resource
def load_summarizer():
    # Load Hugging Face summarization pipeline
    return pipeline("summarization")

summarizer = load_summarizer()

def get_transcript(video_url):
    """Extract transcript text from YouTube URL."""
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript_list])
        return transcript_text
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def chunk_text(text, max_chunk=1000):
    """Split text into chunks to avoid model limits."""
    return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

def generate_summary(text):
    """Generate summary for text using chunking."""
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🎬")
st.title("🎬 YouTube Video Summarizer")
st.write("Paste any YouTube video link to get a concise summary.")

# Input YouTube URL
url = st.text_input("YouTube URL")

if url:
    with st.spinner("Fetching transcript and generating summary..."):
        transcript_text = get_transcript(url)
        if transcript_text:
            summary_text = generate_summary(transcript_text)
            st.subheader("Summary")
            st.write(summary_text)
