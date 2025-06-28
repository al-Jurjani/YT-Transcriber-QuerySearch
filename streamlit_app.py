# Bismillah
# Started on: 01-01-1447 - 2023-10-01

import streamlit as st
from processing import process_youtube_video, search_query_in_transcript

st.title("YouTube Timestamp Finder")

url = st.text_input("Enter YouTube Video URL:")
query = st.text_input("Enter your query (e.g., Docker installation):")

if st.button("Process Video") and url:
    with st.spinner("Processing video. Please wait..."):
        transcript_file, chunks = process_youtube_video(url)
        st.success("Transcript generated successfully.")
        st.download_button("Download Transcript", transcript_file.read(), file_name="transcript.txt")

if st.button("Search Query") and query:
    with st.spinner("Searching query in transcript..."):
        results = search_query_in_transcript(query)
        if not results:
            st.warning("No similar chunks found in the video.")
        else:
            st.subheader("Matching Timestamps")
            for res in results:
                st.markdown(f"[{res['timestamp']}] {res['text']}")

st.write("Enter a YouTube video URL and a query to find matching timestamps in the transcript.")
st.write("Make sure to process the video first before searching for queries.")