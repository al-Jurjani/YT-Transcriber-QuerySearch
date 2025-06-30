# YouTube Video Transcriber and Semantic Search

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-orange.svg)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow.svg)](https://huggingface.co/BAAI/bge-small-en)

This project allows you to download and transcribe any YouTube video and then perform a semantic search to find specific moments within the video based on a natural language query. It uses `yt-dlp` for audio extraction, `openai-whisper` for high-quality transcription, and a `FAISS`-powered vector index for fast, intelligent searching.

The entire interface is built with Streamlit, making it easy to run and interact with.


## Features

-   **YouTube Video Processing**: Simply provide a URL to download and process a video.
-   **Accurate Transcription**: Utilizes OpenAI's `whisper` model to generate a full text transcript with timestamps.
-   **Semantic Query Search**: Instead of simple keyword matching, it understands the *meaning* of your query (e.g., "how to set up the database?") to find the most relevant segments.
-   **Timestamped Results**: The search results point you directly to the relevant moments in the video.
-   **Downloadable Transcript**: Save the full generated transcript as a `.txt` file.
-   **Web-Based UI**: A simple and intuitive interface powered by Streamlit.

## How It Works

The application follows a multi-step process to deliver its functionality:

1.  **Audio Extraction**: When a user provides a YouTube URL, `yt-dlp` downloads the best available audio stream.
2.  **Transcription**: The downloaded audio is fed into the `whisper` model, which transcribes the speech into text and extracts segments with `start` and `end` timestamps.
3.  **Chunking**: The transcript is broken down into smaller, overlapping text chunks using `langchain` to ensure semantic completeness for embedding.
4.  **Embedding**: Each chunk is converted into a numerical vector (an embedding) using the `BAAI/bge-small-en` model from Hugging Face. These embeddings capture the semantic meaning of the text.
5.  **Indexing**: All embeddings are stored in a `FAISS` index. FAISS (Facebook AI Similarity Search) allows for incredibly fast and efficient similarity searches over millions of vectors.
6.  **Querying**: When you enter a search query, it's also converted into an embedding. FAISS then compares this query embedding to all the indexed chunk embeddings to find the ones that are most similar.
7.  **Displaying Results**: The application displays the text from the top matching chunks along with their original timestamps.

## Technologies Used

-   **Backend/Processing**: Python
-   **Web Framework**: Streamlit
-   **Audio Downloader**: `yt-dlp`
-   **Audio Transcription**: `openai-whisper`
-   **Text Processing**: `langchain`
-   **NLP/Embeddings**: `transformers`, `torch`, `sentencepiece`
-   **Vector Search**: `faiss-cpu`

## Installation and Usage

Follow these steps to run the project locally.

### Prerequisites

-   Python 3.8 or higher.
-   **FFmpeg**: Whisper requires FFmpeg to be installed on your system.
    -   **Windows**: Download from the [official site](https://ffmpeg.org/download.html), unzip it, and add the `bin` directory to your system's PATH environment variable. The `processing.py` script has a hardcoded path for this which you may need to update.
    -   **macOS**: `brew install ffmpeg`
    -   **Linux**: `sudo apt update && sudo apt install ffmpeg`

### 1. Clone the Repository

```bash
git clone https://github.com/al-Jurjani/YT-Transcriber-QuerySearch.git
cd YT-Transcriber-QuerySearch
```

### 2. Running with Docker

If you don't want to set up Python and dependencies locally, you can run the entire app using Docker.

### 🔧 Step 1: Build the Docker Image

From the root of the project directory (where the `Dockerfile` is located):

```bash
docker build -t yt_transcriber .
```

### ▶️ Step 2: Run the App

```bash
docker run -p 8501:8501 yt_transcriber
```

Then open your browser and go to:

```
http://localhost:8501
```

> 💡 **Note:** The app may take a few minutes to start on the first run because it loads machine learning models.

Alternatively, you can download the .tar file from the repo, then load it like this:
```bash
docker load -i yt_tqs_zuhair.tar
```

Then you can run the code like this
```bash
docker run -p 8501:8501 yt_tqs_zuhair
```
