import os
import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel
from transformers import pipeline
import openai

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
            background-color: #f5f7fa;
        }
        .main-title {
            text-align: center;
            color: #ff4b4b;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 0.2em;
        }
        .sub-text {
            text-align: center;
            font-size: 1.1em;
            color: #555;
            margin-bottom: 2em;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 2px solid #ff4b4b;
        }
        .summary-card {
            background-color: white;
            padding: 1.5em;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-top: 1.5em;
        }
        .summary-header {
            color: #ff4b4b;
            font-weight: 600;
            font-size: 1.4em;
            margin-bottom: 0.5em;
        }
        .summary-text {
            font-size: 1.1em;
            line-height: 1.6em;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 class='main-title'>🎬 YouTube Video Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Paste any YouTube video link and get a concise, clean summary!</p>", unsafe_allow_html=True)

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
    return WhisperModel("base")  # or "tiny" if memory limited

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

whisper_model = load_whisper()
summarizer = load_summarizer()

# -------------------------------
# Helper functions (download, transcribe, chunk, summarize)
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
        for i, c in enumerate(chunks2):
            summaries2.append(summarizer(c, max_length=150, min_length=50, do_sample=False)[0]['summary_text'])
        combined_summary = " ".join(summaries2)
    return combined_summary

def format_summary_pointwise(summary_text):
    import re
    points = re.split(r'(?<=[.!?]) +', summary_text)
    formatted = "\n".join([f"• {point.strip()}" for point in points if point.strip()])
    return formatted

# -------------------------------
# LLM refinement (optional)
# -------------------------------
def refine_with_llm(bulleted_summary, transcript=None):
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not openai_api_key:
        return bulleted_summary, None
    openai.api_key = openai_api_key
    system_prompt = (
        "You are a concise summarization assistant. "
        "Given a rough summary, produce short, factual bullet points."
    )
    user_prompt = f"Rough summary:\n{bulleted_summary}\n\nPlease improve it clearly."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
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
        return bulleted_summary, f"LLM refine failed: {e}"

# -------------------------------
# Streamlit UI logic
# -------------------------------
url = st.text_input("🔗 Enter YouTube URL here:")

if st.button("📝 Summarize"):
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

            with st.spinner("🤖 Enhancing summary with LLM..."):
                refined_summary, llm_error = refine_with_llm(formatted_summary, transcript=transcript_text)

            # Prefer refined if no error
            final_summary = refined_summary if llm_error is None else formatted_summary

            # Prepend "In this video, "
            if final_summary:
                final_summary = "In this video, " + final_summary[0].lower() + final_summary[1:]

            # Display nicely in a card
            st.markdown(f"""
                <div class='summary-card'>
                    <div class='summary-header'>📝 Video Summary</div>
                    <div class='summary-text'>{final_summary}</div>
                </div>
            """, unsafe_allow_html=True)

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
