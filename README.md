# 🎙️ Offline AI Video Transcription Tool

A high-performance, private, locally running desktop/web application that transcribes video files into timestamped text and subtitles without sending any data to the cloud.

Built with **Faster-Whisper (CTranslate2)**, **FFmpeg**, and **Streamlit**.

---

## 🚀 Key Features

- **100% Local & Offline**: Audio and video never leave your workstation. No external API keys or cloud dependencies.
- **High-Speed Inference**: Uses `faster-whisper` and CTranslate2 with **INT8 quantization**, running up to 4x faster than standard OpenAI Whisper with reduced memory usage.
- **Comprehensive Video Format Support**: Ingests `.mp4`, `.mov`, `.mkv`, `.avi`, and `.webm`.
- **Standard 16 kHz Mono Audio Pipeline**: Employs FFmpeg (`-vn -acodec pcm_s16le -ar 16000 -ac 1`) for clean speech extraction.
- **Timestamped Transcripts**: Formatted with `[HH:MM:SS]` intervals.
- **Multi-Format Exporters**:
  - 📄 **Plain Text (`.txt`)**: Clean, human-readable transcript with metadata and timestamps.
  - 📝 **Microsoft Word (`.docx`)**: Formatted document with styled time badges and metadata summary table.
  - 🎬 **Subtitles (`.srt`)**: Standard SubRip subtitles ready to import into Premiere, DaVinci Resolve, VLC, or YouTube.
- **Hardware Acceleration**: Automatic detection of NVIDIA CUDA GPU acceleration, with CPU fallback.
- **Live Preview & Segment Streaming**: Real-time progress bar and segment updates during transcription.

---

## 📁 Project Structure

```
d:\Video-Transcription-Tool\
├── app.py                 # Streamlit UI & interactive workflow
├── requirements.txt       # Python package dependencies
├── README.md              # Documentation & setup guide
└── core/
    ├── __init__.py        # Package exports
    ├── audio.py           # FFmpeg audio conversion to 16 kHz mono WAV
    ├── transcriber.py     # Faster-Whisper / CTranslate2 INT8 speech engine
    └── exporters.py       # Exporters for .txt, .docx, and .srt with timestamp formatting
```

---

## 🛠️ Prerequisites & Setup

### 1. Python Environment
Python 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit) is recommended.

Open PowerShell or your terminal in the project directory:

```powershell
cd d:\Video-Transcription-Tool
```

Create and activate a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# (Or on Command Prompt: .\venv\Scripts\activate.bat)
# (Or on Linux/macOS: source venv/bin/activate)
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Note on FFmpeg**: `requirements.txt` includes `imageio-ffmpeg`, which automatically bundles a standalone FFmpeg binary. You do **not** need to manually install FFmpeg or edit system PATH variables. If you already have system FFmpeg installed, the application detects and prioritizes it automatically.

---

## 🏃 Running the Application

Launch the local web UI:

```powershell
streamlit run app.py
```

Streamlit will automatically open your default browser at:
```
http://localhost:8501
```

---

## 📖 Usage Guide

1. **Configure Settings (Sidebar)**:
   - **Model Size**: Choose `base` (default) or `small` for fast, accurate everyday transcription. For specialized audio or challenging accents, choose `medium` or `large-v3`.
   - **Quantization**: Keep `int8` for maximum speed and minimal memory footprint.
   - **Task Mode**: Choose *Transcribe* (keep source language) or *Translate to English*.
   - **Voice Activity Filter (VAD)**: Keep enabled to skip silent sections and background noise.
2. **Upload Video**:
   - Drag and drop any `.mp4`, `.mov`, `.mkv`, `.avi`, or `.webm` file.
3. **Transcribe**:
   - Click **🚀 Extract Audio & Transcribe**.
   - Monitor the progress bar and real-time segment updates.
4. **Export Results**:
   - Download `.txt`, `.docx`, or `.srt` with a single click.

---

## 🔒 True Air-Gapped / 100% Offline Guide

By default, when `faster-whisper` loads a model for the very first time, it downloads the model weights from HuggingFace to your local cache (`~/.cache/huggingface/hub`).

If you need to deploy this tool in a **completely air-gapped environment** (no internet access ever):

### Step 1: Pre-download model weights on a connected machine
Run this one-line Python command:

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('base', download_root='./models/base')"
```

### Step 2: Transfer the directory to your offline computer
Copy the `models/base` directory to the target offline machine.

### Step 3: Point the application to the local folder
In `app.py` or code, pass the path directly:
```python
engine = WhisperTranscriber(model_size_or_path="./models/base")
```

---

## 💻 Hardware Acceleration (GPU vs. CPU)

- **CPU**: Runs out-of-the-box on all modern multi-core processors using CTranslate2's optimized AVX/AVX2/AVX-512 kernels.
- **NVIDIA GPU (CUDA)**: If an NVIDIA GPU with CUDA drivers is detected, `faster-whisper` leverages CUDA and cuDNN automatically. Set *Inference Device* to `cuda` or `auto`.
