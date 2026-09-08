"""
Offline AI Speech Recognition Engine using faster-whisper (CTranslate2).
Supports INT8 quantization, CPU/CUDA hardware detection, and real-time segment streaming.
"""

import os
from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Tuple, Union

try:
    import ctranslate2
    CT2_AVAILABLE = True
except ImportError:
    CT2_AVAILABLE = False

from faster_whisper import WhisperModel


@dataclass
class TranscriptionSegment:
    """Represents a single transcribed speech segment with time boundaries."""
    id: int
    start: float       # In seconds
    end: float         # In seconds
    text: str          # Transcribed text content
    confidence: float  # Segment confidence score (estimated from avg_logprob)


@dataclass
class TranscriptionResult:
    """Complete result of a transcription run."""
    language: str
    language_probability: float
    duration: float
    segments: List[TranscriptionSegment]
    full_text: str


def detect_optimal_device() -> Tuple[str, str]:
    """
    Detects whether CUDA GPU is available for inference, falling back to CPU.

    Returns:
        Tuple[str, str]: (device, recommended_compute_type)
                         e.g. ("cuda", "float16") or ("cpu", "int8")
    """
    if CT2_AVAILABLE and hasattr(ctranslate2, "get_cuda_device_count"):
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "int8_float16"
        except Exception:
            pass

    return "cpu", "int8"


class WhisperTranscriber:
    """
    Wrapper for faster-whisper WhisperModel providing offline transcription,
    quantization controls, and progress streaming.
    """

    def __init__(
        self,
        model_size_or_path: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        cpu_threads: Optional[int] = None,
        download_root: Optional[str] = None,
    ):
        """
        Initializes the faster-whisper inference engine.

        Args:
            model_size_or_path (str): Whisper model identifier ("tiny", "base", "small", "medium", "large-v3")
                                      or path to a local directory with CTranslate2 weights.
            device (str): Device to use ("auto", "cpu", "cuda").
            compute_type (str): Quantization mode ("int8", "float16", "int8_float16", "float32").
            cpu_threads (Optional[int]): Number of CPU threads for inference. Defaults to available cores.
            download_root (Optional[str]): Local directory for cached model weights.
        """
        if device == "auto":
            detected_device, default_compute = detect_optimal_device()
            self.device = detected_device
            self.compute_type = compute_type if compute_type != "auto" else default_compute
        else:
            self.device = device
            self.compute_type = compute_type

        if cpu_threads is None:
            cpu_threads = max(1, (os.cpu_count() or 4) - 1)
        self.cpu_threads = cpu_threads
        self.model_size_or_path = model_size_or_path
        self.download_root = download_root

        # Initialize the underlying CTranslate2 WhisperModel
        self.model = WhisperModel(
            model_size_or_path=self.model_size_or_path,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=self.download_root,
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        progress_callback: Optional[Callable[[float, TranscriptionSegment], None]] = None,
    ) -> TranscriptionResult:
        """
        Transcribes a 16 kHz mono WAV file into timestamped segments.

        Args:
            audio_path (str): Path to audio file.
            language (Optional[str]): 2-letter ISO language code (e.g. 'en', 'es', 'fr').
                                      If None, language is automatically detected.
            task (str): "transcribe" for transcription or "translate" to translate into English.
            beam_size (int): Beam search width (default 5).
            vad_filter (bool): Enable Silero VAD to filter out background noise/silence.
            progress_callback (Optional[Callable]): Optional function called on each segment with (progress_ratio, segment).

        Returns:
            TranscriptionResult: Complete transcription results with segments and full text.
        """
        segments_raw, info = self.model.transcribe(
            audio=audio_path,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        total_duration = info.duration if info.duration > 0 else 1.0
        segments_list: List[TranscriptionSegment] = []
        text_accumulator: List[str] = []

        for idx, seg in enumerate(segments_raw):
            # Compute confidence score approximation from average log-probability
            confidence = min(1.0, max(0.0, float(2.71828 ** seg.avg_logprob)))

            clean_text = seg.text.strip()
            segment_obj = TranscriptionSegment(
                id=idx + 1,
                start=seg.start,
                end=seg.end,
                text=clean_text,
                confidence=round(confidence, 3),
            )
            segments_list.append(segment_obj)
            text_accumulator.append(clean_text)

            if progress_callback is not None:
                progress = min(1.0, seg.end / total_duration)
                progress_callback(progress, segment_obj)

        full_transcript = " ".join(text_accumulator)

        return TranscriptionResult(
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(info.duration, 2),
            segments=segments_list,
            full_text=full_transcript,
        )

    def transcribe_stream(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Generator[Tuple[float, TranscriptionSegment, dict], None, None]:
        """
        Streaming generator yielding progress and new segments in real time.

        Yields:
            Tuple[float, TranscriptionSegment, dict]: (progress_ratio, segment, info_dict)
        """
        segments_raw, info = self.model.transcribe(
            audio=audio_path,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        info_dict = {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
        }

        total_duration = info.duration if info.duration > 0 else 1.0

        for idx, seg in enumerate(segments_raw):
            confidence = min(1.0, max(0.0, float(2.71828 ** seg.avg_logprob)))
            clean_text = seg.text.strip()
            segment_obj = TranscriptionSegment(
                id=idx + 1,
                start=seg.start,
                end=seg.end,
                text=clean_text,
                confidence=round(confidence, 3),
            )
            progress = min(1.0, seg.end / total_duration)
            yield progress, segment_obj, info_dict
