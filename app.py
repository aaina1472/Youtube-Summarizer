import streamlit as st

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
# Input
# -------------------------------
url = st.text_input("🔗 Enter YouTube URL here:")

if url:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from transformers import pipeline

        # -------------------------------
        # Load summarizer
        # -------------------------------
        @st.cache_resource
        def load_summarizer():
            return pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=-1
            )

        summarizer = load_summarizer()

        # -------------------------------
        # Helper functions
        # -------------------------------
        def get_transcript(video_url):
            video_id = video_url.split("v=")[-1].split("&")[0]
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([t['text'] for t in transcript_list])

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
        # Run
        # -------------------------------
        with st.spinner("⏳ Fetching transcript and generating summary..."):
            transcript_text = get_transcript(url)
            summary_text = recursive_summarize(transcript_text)
            st.markdown("### 📝 Summary")
            st.success(summary_text)
            st.balloons()

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
        st.info("Make sure the URL is a valid YouTube video link.")
