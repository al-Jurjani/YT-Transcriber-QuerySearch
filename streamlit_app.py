# Bismillah
# Started on: 01-01-1447 - 2023-10-01

import streamlit as st
from processing import process_youtube_video, search_query_in_transcript

st.title("YouTube Timestamp Finder")

url = st.text_input("Enter YouTube Video URL:")
query = st.text_input("Enter your query (e.g., Docker installation):")

if st.button("Process Video") and url:
    with st.spinner("Processing video. Please wait..."):
        transcript_file, video_title, chunks, index, embeddings = process_youtube_video(url)
        st.success("Transcript generated successfully.")

        # Save the file reference in session state so it persists
        st.session_state.transcript_file = transcript_file
        st.session_state.video_title = video_title
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.embeddings = embeddings
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
            results = search_query_in_transcript(query, st.session_state.index, st.session_state.chunks, st.session_state.embeddings)
            if not results:
                st.warning("No similar chunks found in the video.")
            else:
                st.subheader("Matching Timestamps with Semantic Scores")
                
                # Display results with scores
                for i, res in enumerate(results, 1):
                    with st.expander(f"Match {i} - [{res['timestamp']}]"):
                        st.markdown(f"**Text:** {res['text']}")
                        
                        # Create columns for metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Cosine Similarity", f"{res['cosine_similarity']:.3f}")
                        with col2:
                            st.metric("L2 Distance", f"{res['l2_distance']:.3f}")
                        
                        # Progress bar for similarity
                        # st.progress(res['similarity_score'])
                
                # Summary table
                st.subheader("Results Summary")
                summary_data = []
                for i, res in enumerate(results, 1):
                    summary_data.append({
                        "Rank": i,
                        "Timestamp": res['timestamp'],
                        # "Similarity": f"{res['similarity_score']:.3f}",
                        "Cosine Sim": f"{res['cosine_similarity']:.3f}",
                        "L2 Distance": f"{res['l2_distance']:.3f}",
                        "Text Preview": res['text'][:100] + "..." if len(res['text']) > 100 else res['text']
                    })
                
                st.dataframe(summary_data, use_container_width=True)

st.write("Enter a YouTube video URL and a query to find matching timestamps in the transcript.")
st.write("Make sure to process the video first before searching for queries.")