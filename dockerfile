# Use Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]

# # Use Python 3.10 slim image for smaller size and faster downloads
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies (including ffmpeg for whisper)
# # Combine RUN commands to reduce layers and build time
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     git \
#     wget \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# # Copy requirements first for better Docker layer caching
# COPY requirements.txt .

# # Install Python dependencies
# # Use --no-cache-dir to reduce image size
# # Use pip install with CPU-specific torch to avoid GPU dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application files
# COPY processing.py .
# COPY streamlit_app.py .

# # Create directories for temporary files
# RUN mkdir -p /tmp/whisper_cache

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV STREAMLIT_SERVER_PORT=8501
# ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# # Expose the port Streamlit runs on
# EXPOSE 8501

# # Health check to ensure the app is running
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:8501/_stcore/health || exit 1

# # Run the Streamlit app
# CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]