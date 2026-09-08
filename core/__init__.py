"""
Core module for offline video transcription tool.
Integrates FFmpeg audio extraction, faster-whisper AI transcription, and export utilities.
"""

from .audio import extract_audio, get_ffmpeg_executable, get_media_duration
from .transcriber import WhisperTranscriber
from .exporters import export_to_txt, export_to_docx, export_to_srt, format_timestamp

__all__ = [
    "extract_audio",
    "get_ffmpeg_executable",
    "get_media_duration",
    "WhisperTranscriber",
    "export_to_txt",
    "export_to_docx",
    "export_to_srt",
    "format_timestamp",
]
