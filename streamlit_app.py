# Bismillah
# Started on: 01-01-1447 - 2023-10-01

import streamlit as st
from processing import process_youtube_video, search_query_in_transcript

st.title("YouTube Timestamp Finder")

url = st.text_input("Enter YouTube Video URL:")
query = st.text_input("Enter your query (e.g., Docker installation):")

if st.button("Process Video") and url:
    with st.spinner("Processing video. Please wait..."):
        transcript_file, video_title, chunks, index = process_youtube_video(url)
        st.success("Transcript generated successfully.")

        # Save the file reference in session state so it persists
        st.session_state.transcript_file = transcript_file
        st.session_state.video_title = video_title
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.video_processed = True

# Show buttons after video has been processed
if st.session_state.get("video_processed", False):

    st.download_button(
        "Download Transcript",
        st.session_state.transcript_file.read(),
        file_name=f"{st.session_state.video_title}___transcript.txt"
    )

    if st.button("Search Query") and query:
        with st.spinner("Searching query in transcript..."):
            results = search_query_in_transcript(query, st.session_state.index, st.session_state.chunks)
            if not results:
                st.warning("No similar chunks found in the video.")
            else:
                st.subheader("Matching Timestamps")
                for res in results:
                    st.markdown(f"[{res['timestamp']}] {res['text']}")
            # st.download_button("Download Transcript", transcript_file.read(), file_name=f"{video_title}___transcript.txt")

st.write("Enter a YouTube video URL and a query to find matching timestamps in the transcript.")
st.write("Make sure to process the video first before searching for queries.")