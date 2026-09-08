"""
Audio extraction and processing module using FFmpeg.
Converts video inputs (MP4, MOV, MKV, AVI, WEBM) to 16 kHz mono WAV for Whisper.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def get_ffmpeg_executable() -> str:
    """
    Locates the FFmpeg binary executable.
    Checks system PATH first, then falls back to imageio-ffmpeg bundled binary.

    Returns:
        str: Absolute path to the FFmpeg executable.

    Raises:
        RuntimeError: If FFmpeg cannot be located anywhere.
    """
    # 1. Check system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # 2. Check imageio-ffmpeg package
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.isfile(ffmpeg_exe):
            return ffmpeg_exe
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg executable was not found. Please install FFmpeg and add it to your PATH, "
        "or install imageio-ffmpeg via: pip install imageio-ffmpeg"
    )


def get_media_duration(file_path: str, ffmpeg_exe: Optional[str] = None) -> float:
    """
    Retrieves the duration in seconds of an audio or video file using FFmpeg.

    Args:
        file_path (str): Path to the media file.
        ffmpeg_exe (Optional[str]): Optional path to FFmpeg executable.

    Returns:
        float: Duration in seconds, or 0.0 if not detectable.
    """
    if not os.path.exists(file_path):
        return 0.0

    if ffmpeg_exe is None:
        try:
            ffmpeg_exe = get_ffmpeg_executable()
        except RuntimeError:
            return 0.0

    cmd = [ffmpeg_exe, "-i", file_path]
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # Search for Duration: 00:01:23.45 in stderr
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", process.stderr)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    return 0.0


def extract_audio(
    video_path: str,
    output_wav_path: Optional[str] = None,
    sample_rate: int = 16000
) -> Tuple[str, float]:
    """
    Extracts audio from a video file and converts it to a 16 kHz mono WAV file,
    the optimal format for Whisper speech recognition models.

    Args:
        video_path (str): Path to the source video file.
        output_wav_path (Optional[str]): Destination WAV file path. If None, a temp file is created.
        sample_rate (int): Audio sampling rate in Hz (default 16000).

    Returns:
        Tuple[str, float]: Tuple containing (output_wav_path, duration_in_seconds).

    Raises:
        FileNotFoundError: If the source video file does not exist.
        ValueError: If the file extension is not among supported video formats.
        RuntimeError: If FFmpeg execution fails.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ext = Path(video_path).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format: '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )

    ffmpeg_exe = get_ffmpeg_executable()

    if output_wav_path is None:
        # Create a named temporary WAV file
        temp_dir = tempfile.gettempdir()
        base_name = Path(video_path).stem
        output_wav_path = os.path.join(temp_dir, f"{base_name}_extracted_16k.wav")

    # Command: ffmpeg -y -i <input> -vn -acodec pcm_s16le -ar 16000 -ac 1 <output>
    # -vn: disable video recording
    # -acodec pcm_s16le: 16-bit uncompressed PCM
    # -ar 16000: 16kHz sampling rate
    # -ac 1: 1 channel (mono)
    cmd = [
        ffmpeg_exe,
        "-y",               # Overwrite output file if exists
        "-i", video_path,   # Input file
        "-vn",              # Strip video stream
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_wav_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed (exit code {result.returncode}):\n{result.stderr}"
        )

    duration = get_media_duration(output_wav_path, ffmpeg_exe=ffmpeg_exe)
    return output_wav_path, duration
