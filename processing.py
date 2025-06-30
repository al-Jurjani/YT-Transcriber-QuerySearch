# # Bismillah
# # Started on: 01-01-1447 - 2023-10-01

import os
import tempfile
import torch
import whisper
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import shutil
from yt_dlp import YoutubeDL
import datetime

# Globals
CHUNKS = []
EMBEDDINGS = []
INDEX = None
MODEL_NAME = "BAAI/bge-small-en"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Helper Functions
def download_audio_with_ytdlp(url, output_path):
    cmd = [
        "yt-dlp",
        url,
        "-f", "bestaudio",
        "--output", output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise Exception(f"yt-dlp failed: {result.stderr.decode()}")

def get_embeddings(texts):
    with torch.no_grad():
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # outputs = model(**tokens)
        outputs = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        embeddings = outputs.last_hidden_state[:, 0, :].numpy() # last_hidden_state is the output of the model
    return embeddings.astype("float32")

# embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    # outputs.last_hidden_state is a 3D tensor:
    # shape = (batch_size, seq_len, hidden_size)
    # e.g., (5, 128, 384) for 5 texts, 128 tokens each, and embedding size 384.

    # [:, 0, :]: For each input in the batch, take only the first token's embedding ([CLS] token).

    # So we get shape: (batch_size, hidden_size) — perfect for FAISS or similarity search.

    # .numpy(): Converts it from PyTorch tensor → NumPy array.

def seconds_to_hhmmss(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# def l2_to_similarity_score(l2_distance):
#     """Convert L2 distance to a similarity score between 0-1 (1 being most similar)"""
#     # Using exponential decay to convert distance to similarity
#     # You can adjust the scale factor (0.5) based on your preference
#     return np.exp(-l2_distance * 0.5)

def cosine_similarity_score(query_embedding, chunk_embedding):
    """Calculate cosine similarity between query and chunk embeddings"""
    # Reshape to 2D arrays for sklearn
    query_emb = query_embedding.reshape(1, -1)
    chunk_emb = chunk_embedding.reshape(1, -1)
    return cosine_similarity(query_emb, chunk_emb)[0][0]

# Main Functions
def process_youtube_video(url):
    # Step 1: Download Audio using yt-dlp (raw format, no ffmpeg)
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.%(ext)s")
    final_audio_path = None
    print("Downloading audio...")
    download_audio_with_ytdlp(url, audio_path)
    print("Audio downloaded successfully!")

    # Locate downloaded file
    for file in os.listdir(temp_dir):
        if file.startswith("audio") and file.endswith(('.webm', '.m4a', '.mp3')):
            final_audio_path = os.path.join(temp_dir, file)
            break

    if final_audio_path is None:
        raise FileNotFoundError("Audio file not found after yt-dlp download.")
    
    with YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
    VIDEO_TITLE = info.get('title', 'video').replace(" ", "_").replace("/", "_")
    TRANSCRIPT_FILE = f"{VIDEO_TITLE}___transcript.txt"

    # Inject ffmpeg into PATH for Whisper
    # Uncomment the following two lines if you need to set a specific ffmpeg path
    # Comment out the following two lines if using Docker or if ffmpeg is already in PATH
    # ffmpeg_bin = r"Z:\\ffmpeg\\ffmpeg-7.1.1-essentials_build\\bin"
    # os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]

    # Confirm ffmpeg is visible to subprocesses
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("FFmpeg not found! Check your PATH or verify ffmpeg.exe exists in the bin folder.")

    # Step 2: Transcribe using Whisper
    print("Transcribing audio...")
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(final_audio_path)
    transcript = result["segments"]
    print("Transcription completed!")

    # Save transcript to file
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        for seg in transcript:
            timestamp = seconds_to_hhmmss(seg['start'])
            f.write(f"{timestamp} --- {seg['text']}\n")

    # Step 3: Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    for seg in transcript:
        chunks = text_splitter.split_text(seg['text'])
        for chunk in chunks:
            timestamp = seconds_to_hhmmss(seg['start'])
            CHUNKS.append({
                "text": chunk,
                "timestamp": timestamp
            })

    # Convert all chunks to embeddings
    all_texts = [chunk['text'] for chunk in CHUNKS]
    embeddings = get_embeddings(all_texts)

    # Create FAISS index
    INDEX = faiss.IndexFlatL2(embeddings.shape[1])
    INDEX.add(embeddings)

    # Save embeddings globally
    global EMBEDDINGS
    EMBEDDINGS = embeddings

    return open(TRANSCRIPT_FILE, "rb"), VIDEO_TITLE, CHUNKS, INDEX, EMBEDDINGS


def search_query_in_transcript(query, index, chunks, embeddings, top_k=5):
    query_vector = get_embeddings([query])[0].reshape(1, -1)
    D, I = index.search(query_vector, top_k) # calculates the distances and indices of the top_k nearest neighbors based on L2 distance

    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(chunks):
            l2_distance = D[0][i]  # L2 distance from FAISS
            # similarity_score = l2_to_similarity_score(l2_distance)  # Convert to 0-1 scale
            cosine_sim = cosine_similarity_score(query_vector[0], embeddings[idx])  # Cosine similarity
            
            result = {
                "text": chunks[idx]['text'],
                "timestamp": chunks[idx]['timestamp'],
                "l2_distance": float(l2_distance),
                # "similarity_score": float(similarity_score),
                "cosine_similarity": float(cosine_sim)
            }
            results.append(result)
    
    return results