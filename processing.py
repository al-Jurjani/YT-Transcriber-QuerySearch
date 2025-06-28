# # Bismillah
# # Started on: 01-01-1447 - 2023-10-01

# import os
# import tempfile
# import torch
# from pytube import YouTube
# from whisper import load_model
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity
# import faiss
# import numpy as np

# # Globals
# TRANSCRIPT_FILE = "transcript.txt"
# CHUNKS = []
# EMBEDDINGS = []
# INDEX = None
# MODEL_NAME = "BAAI/bge-small-en"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)


# def process_youtube_video(url):
#     # Step 1: Download Audio
#     # yt = YouTube(url)
#     # stream = yt.streams.filter(only_audio=True).first()

#     try:
#         yt = YouTube(url)
#         VIDEO_TITLE = yt.title.replace(" ", "_").replace("/", "_")  # Clean title for file naming
#         TRANSCRIPT_FILE = f"{VIDEO_TITLE}___transcript.txt"
#         stream = yt.streams.filter(only_audio=True).first()
#         if stream is None:
#             raise Exception("No downloadable audio stream found.")
#     except Exception as e:
#         raise Exception(f"Failed to load or download video: {e}")


#     temp_dir = tempfile.mkdtemp()
#     audio_path = stream.download(output_path=temp_dir, filename="audio.mp3")

#     # Step 2: Transcribe using Whisper
#     whisper_model = load_model("base")  # or 'medium' if you want more accuracy
#     print("Transcribing audio...")
#     result = whisper_model.transcribe(audio_path)
#     print("Transcription completed!")
#     transcript = result["segments"]  # each segment has 'start', 'end', 'text'

#     # Save transcript to file
#     with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
#         for seg in transcript:
#             f.write(f"[{seg['start']:.2f}] {seg['text']}\n")

#     # Step 3: Chunking
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
#     for seg in transcript:
#         chunks = text_splitter.split_text(seg['text'])
#         for chunk in chunks:
#             CHUNKS.append({
#                 "text": chunk,
#                 "timestamp": f"{seg['start']:.2f}"
#             })

#     # Step 4: Embedding and FAISS Indexing
#     embeddings = get_embeddings([chunk['text'] for chunk in CHUNKS])
#     global INDEX
#     INDEX = faiss.IndexFlatL2(embeddings.shape[1])
#     INDEX.add(embeddings)
#     global EMBEDDINGS
#     EMBEDDINGS = embeddings

#     return open(TRANSCRIPT_FILE, "rb"), CHUNKS


# def get_embeddings(texts):
#     with torch.no_grad():
#         tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
#         outputs = model(**tokens)
#         embeddings = outputs.last_hidden_state[:, 0, :].numpy()
#     return embeddings.astype("float32")


# def search_query_in_transcript(query, top_k=5):
#     query_emb = get_embeddings([query])[0].reshape(1, -1)
#     D, I = INDEX.search(query_emb, top_k)

#     results = []
#     for idx in I[0]:
#         if idx < len(CHUNKS):
#             results.append(CHUNKS[idx])

#     return results

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
# TRANSCRIPT_FILE = "transcript.txt"
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

    # # Step 2: Transcribe using Whisper
    # whisper_model = whisper.load_model("base")
    # result = whisper_model.transcribe(final_audio_path)
    # transcript = result["segments"]

    # whisper_model = whisper.load_model("base")
    # result = whisper_model.transcribe(final_audio_path)
    # transcript = result["segments"]

    # Inject ffmpeg into PATH for Whisper
    ffmpeg_bin = r"Z:\\ffmpeg\\ffmpeg-7.1.1-essentials_build\\bin"
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]

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

    # Step 4: Embedding and FAISS Indexing
    # embeddings = get_embeddings([chunk['text'] for chunk in CHUNKS])
    # global INDEX
    # INDEX = faiss.IndexFlatL2(embeddings.shape[1])
    # INDEX.add(embeddings)
    # global EMBEDDINGS
    # EMBEDDINGS = embeddings

    # Convert all chunks to embeddings
    all_texts = [chunk['text'] for chunk in CHUNKS]
    embeddings = get_embeddings(all_texts)

    # Create FAISS index
    INDEX = faiss.IndexFlatL2(embeddings.shape[1])
    INDEX.add(embeddings)

    # Save embeddings globally
    global EMBEDDINGS
    EMBEDDINGS = embeddings

    return open(TRANSCRIPT_FILE, "rb"), VIDEO_TITLE, CHUNKS, INDEX


def search_query_in_transcript(query, index, chunks, top_k=5):
    query_vector = get_embeddings([query])[0].reshape(1, -1)
    D, I = index.search(query_vector, top_k)

    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    return results