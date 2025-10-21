import streamlit as st
import os
import yt_dlp
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
import imageio_ffmpeg  # to get ffmpeg binary automatically

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
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

@st.cache_resource
def load_wav2vec_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer, model

summarizer = load_summarizer()
tokenizer, wav2vec_model = load_wav2vec_model()

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
    # Load audio and convert to 16kHz
    audio, rate = librosa.load(audio_file, sr=16000)
    input_values = tokenizer(audio, return_tensors="pt").input_values
    with torch.no_grad():
        logits = wav2vec_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

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

            with st.spinner("⏳ Summarizing transcript..."):
                summary_text = recursive_summarize(transcript_text)

            formatted_summary = format_summary_pointwise(summary_text)

            st.markdown("### 📝 Summary (Point-wise)")
            st.success(formatted_summary)
            st.balloons()

            # Remove audio file after processing
            os.remove(audio_file)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
    else:
        st.warning("⚠️ Please enter a valid YouTube URL.")
