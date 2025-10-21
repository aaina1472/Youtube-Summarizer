import os
import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel
from transformers import pipeline
import openai   # new: for LLM refinement

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
    # safe sentence-aware chunking
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
    # avoid deep recursion: if combined too long, summarize again but keep one level limit
    if len(combined_summary) > 3000:
        chunks2 = chunk_text(combined_summary, max_chunk=2000)
        summaries2 = []
        for i, c in enumerate(chunks2):
            summaries2.append(summarizer(c, max_length=150, min_length=50, do_sample=False)[0]['summary_text'])
        combined_summary = " ".join(summaries2)
    return combined_summary

def format_summary_pointwise(summary_text):
    import re
    # Split by sentence endings (., !, ?)
    points = re.split(r'(?<=[.!?]) +', summary_text)
    formatted = "\n".join([f"• {point.strip()}" for point in points if point.strip()])
    return formatted

# -------------------------------
# LLM refinement (OpenAI) with fallback
# -------------------------------
def refine_with_llm(bulleted_summary, transcript=None):
    """
    Send a short prompt to OpenAI to refine the summary into clear, factual,
    concise bullet points. If OpenAI fails (no key/quota), return the original.
    """
    # Get API key from env / Streamlit secrets
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        # Try Streamlit secrets (if running on Streamlit Cloud)
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            openai_api_key = None

    if not openai_api_key:
        # No key available -> fallback
        return bulleted_summary

    openai.api_key = openai_api_key

    system_prompt = (
        "You are a concise summarization assistant. "
        "Given a rough summary and optionally the original transcript, produce a short, accurate, "
        "bullet-point summary (each bullet 1-2 short sentences). Prioritize clarity and factual accuracy. "
        "If any point is uncertain or speculative, omit it. Number the bullets if helpful."
    )

    user_prompt = f"Rough summary:\n{bulleted_summary}\n\n"
    if transcript:
        # give a short hint we have transcript available if needed
        user_prompt += "Full transcript is available if you need to check facts. "
    user_prompt += "\nPlease return the improved summary as bullet points only."

    try:
        # Use Chat Completions (compatible with openai package)
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # replace with a model available to you; fallback handled on error
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.2,
        )
        refined = resp["choices"][0]["message"]["content"].strip()
        return refined, None
    except Exception as e:
        # catch rate-limit, quota, model mismatch, or other issues -> fallback
        return bulleted_summary, f"LLM refine failed: {e}"

# -------------------------------
# Streamlit UI
# -------------------------------
url = st.text_input("🔗 Enter YouTube URL here:")

if st.button("📝 Summarize"):
    if url:
        try:
            # Clear previous summary and audio if present
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

            # LLM refine step (optional, falls back if no key/quota)
            with st.spinner("🤖 Enhancing summary with LLM..."):
                refined_summary, llm_error = refine_with_llm(formatted_summary, transcript=transcript_text)

            # Prefer refined summary if it was successful (i.e., differs and no error message)
            final_summary = refined_summary if (llm_error is None) else formatted_summary

            # Save and display
            st.session_state.last_summary = final_summary
            st.markdown("### 📝 Final Summary (Point-wise)")
            st.success(final_summary)

            if llm_error:
                st.warning(f"LLM refine fallback: {llm_error}")

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

# Persist last summary so page refresh doesn't lose it
if st.session_state.last_summary:
    st.success(st.session_state.last_summary)
