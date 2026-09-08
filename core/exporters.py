"""
Export utilities for transcription results.
Generates timestamped transcripts in .txt, .docx (Word), and .srt (SubRip) subtitle formats.
"""

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from .transcriber import TranscriptionSegment


def format_timestamp(seconds: float, format_type: str = "bracket") -> str:
    """
    Formats a time in seconds into various standard timestamp representations.

    Args:
        seconds (float): Time offset in seconds.
        format_type (str): Format style:
            - 'bracket': [HH:MM:SS]
            - 'srt': HH:MM:SS,mmm (SubRip standard)
            - 'vtt': HH:MM:SS.mmm (WebVTT standard)

    Returns:
        str: Formatted timestamp string.
    """
    if seconds < 0:
        seconds = 0.0

    total_seconds = int(seconds)
    millis = round((seconds - total_seconds) * 1000)
    if millis >= 1000:
        total_seconds += 1
        millis -= 1000

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if format_type == "bracket":
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    elif format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    elif format_type == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def export_to_txt(
    segments: List[TranscriptionSegment],
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Exports segments to a clean human-readable timestamped .txt file.

    Format:
    [HH:MM:SS - HH:MM:SS] Transcribed sentence or segment.

    Args:
        segments (List[TranscriptionSegment]): List of transcription segments.
        file_path (Optional[str]): If provided, writes content to this file path.
        metadata (Optional[Dict[str, Any]]): Optional metadata to include in the header.

    Returns:
        str: The full text content.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("VIDEO TRANSCRIPTION (OFFLINE AI ENGINE)")
    lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if metadata:
        for k, v in metadata.items():
            lines.append(f"{k.capitalize()}: {v}")

    lines.append("=" * 80)
    lines.append("")

    for seg in segments:
        start_str = format_timestamp(seg.start, "bracket")
        end_str = format_timestamp(seg.end, "bracket")
        lines.append(f"{start_str} - {end_str}  {seg.text}")

    content = "\n".join(lines)

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return content


def export_to_srt(
    segments: List[TranscriptionSegment],
    file_path: Optional[str] = None,
) -> str:
    """
    Exports segments to standard SubRip (.srt) subtitle format.

    Format:
    1
    00:00:01,234 --> 00:00:04,567
    Transcribed text line.

    Args:
        segments (List[TranscriptionSegment]): List of transcription segments.
        file_path (Optional[str]): If provided, writes content to this file path.

    Returns:
        str: The complete SRT formatted content.
    """
    srt_blocks = []

    for idx, seg in enumerate(segments, start=1):
        start_ts = format_timestamp(seg.start, "srt")
        end_ts = format_timestamp(seg.end, "srt")

        block = f"{idx}\n{start_ts} --> {end_ts}\n{seg.text}\n"
        srt_blocks.append(block)

    content = "\n".join(srt_blocks)

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return content


def export_to_docx(
    segments: List[TranscriptionSegment],
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    title: str = "Video Transcription Transcript",
) -> io.BytesIO:
    """
    Exports segments to a beautifully styled Microsoft Word (.docx) document.

    Args:
        segments (List[TranscriptionSegment]): List of transcription segments.
        file_path (Optional[str]): If provided, writes content to this file path.
        metadata (Optional[Dict[str, Any]]): Metadata (filename, language, duration, etc.).
        title (str): Document title.

    Returns:
        io.BytesIO: In-memory byte buffer containing the .docx file data.
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
    except ImportError as e:
        raise ImportError(
            "python-docx is required for Word export. Install with: pip install python-docx"
        ) from e

    doc = Document()

    # Document Title
    heading = doc.add_heading(title, level=0)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Metadata Callout Box / Table
    if metadata:
        meta_table = doc.add_table(rows=len(metadata), cols=2)
        meta_table.style = "Table Grid"
        for row_idx, (key, value) in enumerate(metadata.items()):
            row_cells = meta_table.rows[row_idx].cells
            row_cells[0].text = str(key).capitalize()
            row_cells[1].text = str(value)

            # Bold the metadata keys
            for p in row_cells[0].paragraphs:
                for run in p.runs:
                    run.font.bold = True
                    run.font.size = Pt(9.5)
            for p in row_cells[1].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9.5)

        doc.add_paragraph("")  # Spacing

    # Section Heading
    doc.add_heading("Transcript with Timestamps", level=2)

    # Segments
    for seg in segments:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15

        # Timestamp Run (Styled Bold)
        time_text = f"{format_timestamp(seg.start, 'bracket')} - {format_timestamp(seg.end, 'bracket')}  "
        time_run = p.add_run(time_text)
        time_run.font.bold = True
        time_run.font.size = Pt(10)
        time_run.font.color.rgb = RGBColor(30, 80, 160)  # Professional steel blue

        # Text Run
        text_run = p.add_run(seg.text)
        text_run.font.size = Pt(10.5)

    # Save to disk if file_path is specified
    if file_path:
        doc.save(file_path)

    # Always return in-memory BytesIO buffer for Streamlit download buttons
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
