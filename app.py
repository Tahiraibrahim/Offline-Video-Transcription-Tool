"""
Offline AI-Powered Video Transcription Tool.
Streamlit Web Application running locally without any cloud uploads.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
import streamlit as st

from core.audio import (
    SUPPORTED_VIDEO_EXTENSIONS,
    extract_audio,
    get_ffmpeg_executable,
    get_media_duration,
)
from core.exporters import (
    export_to_docx,
    export_to_srt,
    export_to_txt,
    format_timestamp,
)
from core.transcriber import WhisperTranscriber, detect_optimal_device

# Page Configuration
st.set_page_config(
    page_title="Offline AI Video Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a sleek, modern UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-offline {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        margin-bottom: 1rem;
    }
    .segment-card {
        padding: 10px 14px;
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .timestamp-badge {
        font-family: monospace;
        font-weight: 700;
        color: #2563EB;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_transcription_engine(
    model_size: str,
    device: str,
    compute_type: str,
) -> WhisperTranscriber:
    """
    Caches the Faster-Whisper model in memory across reruns.
    """
    return WhisperTranscriber(
        model_size_or_path=model_size,
        device=device,
        compute_type=compute_type,
    )


def main():
    # Sidebar - Configuration & Hardware Status
    with st.sidebar:
        st.title("⚙️ Engine Settings")

        # Hardware Detection Info
        detected_dev, rec_compute = detect_optimal_device()
        st.markdown(
            f"**Detected Hardware:** `{detected_dev.upper()}` "
            f"({'GPU Accelerated' if detected_dev == 'cuda' else 'CPU Mode'})"
        )

        try:
            ffmpeg_path = get_ffmpeg_executable()
            st.success("FFmpeg: Ready", icon="✅")
        except Exception as e:
            st.error(f"FFmpeg: Missing ({e})", icon="⚠️")

        st.markdown("---")

        # Model Configuration
        st.subheader("AI Model")
        model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large-v3"],
            index=1,  # Default to 'base'
            help="Base/Small provides the optimal balance of speed and accuracy on standard CPUs/GPUs.",
        )

        device_option = st.selectbox(
            "Inference Device",
            options=["auto", "cpu", "cuda"],
            index=0,
            help="'auto' automatically selects GPU if CUDA is available, otherwise CPU.",
        )

        compute_type = st.selectbox(
            "Quantization / Precision",
            options=["int8", "float16", "int8_float16", "float32"],
            index=0,  # Default to 'int8'
            help="INT8 quantization drastically speeds up inference and reduces memory footprint.",
        )

        task = st.radio(
            "Task Mode",
            options=["transcribe", "translate"],
            format_func=lambda x: "Transcribe Speech" if x == "transcribe" else "Translate to English",
            index=0,
        )

        st.markdown("---")
        st.subheader("Audio & Detection")

        lang_options = {
            "Auto Detect": None,
            "English (en)": "en",
            "Spanish (es)": "es",
            "French (fr)": "fr",
            "German (de)": "de",
            "Italian (it)": "it",
            "Portuguese (pt)": "pt",
            "Chinese (zh)": "zh",
            "Japanese (ja)": "ja",
            "Korean (ko)": "ko",
            "Hindi (hi)": "hi",
            "Arabic (ar)": "ar",
            "Russian (ru)": "ru",
        }
        selected_lang_label = st.selectbox("Spoken Language", options=list(lang_options.keys()), index=0)
        language = lang_options[selected_lang_label]

        vad_filter = st.checkbox(
            "Enable Voice Activity Filter (VAD)",
            value=True,
            help="Filters out silent segments and non-speech background audio for cleaner transcripts.",
        )

        beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=5, step=1)

    # Main Application Header
    st.markdown('<div class="main-header">🎙️ Offline AI Video Transcription</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Fast local video transcription powered by FFmpeg, Faster-Whisper, and CTranslate2 INT8 inference.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge-offline">🔒 100% Offline &amp; Private &bull; No Cloud Uploads</span>',
        unsafe_allow_html=True,
    )

    # File Uploader
    supported_ext_list = [ext.replace(".", "") for ext in SUPPORTED_VIDEO_EXTENSIONS]
    uploaded_file = st.file_uploader(
        "Upload a video file to transcribe",
        type=supported_ext_list,
        help=f"Supported formats: {', '.join(supported_ext_list).upper()}",
    )

    if uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        file_size_mb = uploaded_file.size / (1024 * 1024)

        col_meta1, col_meta2 = st.columns([1, 1])
        with col_meta1:
            st.info(f"**Filename:** `{uploaded_file.name}` ({file_size_mb:.2f} MB)")
        with col_meta2:
            st.write("")

        # Video Preview
        with st.expander("🎬 Video Preview", expanded=False):
            st.video(uploaded_file)

        # Transcribe Action Button
        if st.button("🚀 Extract Audio & Transcribe", type="primary", use_container_width=True):
            # Save uploaded file to temp directory
            temp_dir = tempfile.mkdtemp(prefix="transcribe_")
            video_temp_path = os.path.join(temp_dir, uploaded_file.name)
            wav_temp_path = os.path.join(temp_dir, f"{Path(uploaded_file.name).stem}_16k.wav")

            try:
                with open(video_temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                status_container = st.status("Processing Video...", expanded=True)

                # Step 1: FFmpeg Extraction
                status_container.write("🎵 Extracting audio stream using FFmpeg (16 kHz mono WAV)...")
                start_time = time.time()
                _, duration = extract_audio(video_temp_path, output_wav_path=wav_temp_path)
                extract_time = time.time() - start_time
                status_container.write(f"✅ Audio extracted in {extract_time:.2f}s (Media duration: {duration:.1f}s)")

                # Step 2: Model Loading
                status_container.write(f"🧠 Initializing Whisper model (`{model_size}`, `{compute_type}`)...")
                engine = load_transcription_engine(
                    model_size=model_size,
                    device=device_option,
                    compute_type=compute_type,
                )

                # Step 3: Transcription with Progress Tracking
                status_container.write("⚡ Transcribing audio segments with faster-whisper...")
                progress_bar = st.progress(0, text="Transcribing: 0%")
                live_preview_box = st.empty()

                segments = []
                info_detected = {}
                infer_start = time.time()

                for prog, segment, info_dict in engine.transcribe_stream(
                    audio_path=wav_temp_path,
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                ):
                    segments.append(segment)
                    info_detected = info_dict
                    progress_pct = int(prog * 100)
                    progress_bar.progress(progress_pct, text=f"Transcribing: {progress_pct}%")

                    # Live preview snippet
                    last_seg_time = format_timestamp(segment.start, "bracket")
                    live_preview_box.markdown(
                        f"**Latest Segment:** `{last_seg_time}` *\"{segment.text}\"*"
                    )

                progress_bar.progress(100, text="Transcription Complete: 100%")
                infer_time = time.time() - infer_start
                status_container.update(
                    label=f"Transcription Completed in {infer_time:.2f}s!",
                    state="complete",
                    expanded=False,
                )

                # Store results in session state for persistent export & display
                st.session_state["transcription_results"] = {
                    "filename": uploaded_file.name,
                    "duration": duration,
                    "segments": segments,
                    "language": info_detected.get("language", "Unknown"),
                    "language_prob": info_detected.get("language_probability", 1.0),
                    "infer_time": infer_time,
                    "model_used": f"{model_size} ({compute_type})",
                }

            except Exception as ex:
                st.error(f"Error during transcription: {str(ex)}")
            finally:
                # Cleanup temp directory
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    # Results & Export Section (Persisted in Session State)
    if "transcription_results" in st.session_state:
        res = st.session_state["transcription_results"]
        segments = res["segments"]
        duration = res["duration"]

        st.markdown("---")
        st.subheader("📊 Transcription Results")

        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Audio Duration", f"{int(duration // 60):02d}:{int(duration % 60):02d}")
        m2.metric("Detected Language", f"{res['language'].upper()} ({int(res['language_prob'] * 100)}%)")
        total_words = sum(len(s.text.split()) for s in segments)
        m3.metric("Total Words", f"{total_words}")
        m4.metric("Inference Time", f"{res['infer_time']:.2f}s")

        # Export Buttons Center
        st.markdown("#### 📥 Export Formats")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        metadata_dict = {
            "Filename": res["filename"],
            "Language": f"{res['language']} (confidence: {res['language_prob']:.2f})",
            "Duration": f"{duration:.2f} seconds",
            "Model": res["model_used"],
        }

        # 1. Plain Text (.txt)
        txt_content = export_to_txt(segments, metadata=metadata_dict)
        with exp_col1:
            st.download_button(
                label="📄 Download .TXT Transcript",
                data=txt_content,
                file_name=f"{Path(res['filename']).stem}_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # 2. Microsoft Word (.docx)
        docx_buffer = export_to_docx(segments, metadata=metadata_dict)
        with exp_col2:
            st.download_button(
                label="📝 Download .DOCX Document",
                data=docx_buffer.getvalue(),
                file_name=f"{Path(res['filename']).stem}_transcript.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

        # 3. Subtitles (.srt)
        srt_content = export_to_srt(segments)
        with exp_col3:
            st.download_button(
                label="🎬 Download .SRT Subtitles",
                data=srt_content,
                file_name=f"{Path(res['filename']).stem}_subtitles.srt",
                mime="application/x-subrip",
                use_container_width=True,
            )

        st.markdown("---")

        # Tabbed Viewer
        tab_segments, tab_fulltext, tab_srt = st.tabs([
            "⏱️ Timestamped Segments",
            "📜 Full Continuous Text",
            "🎬 SubRip (.SRT) View",
        ])

        with tab_segments:
            search_query = st.text_input("🔍 Filter segments by keyword", "")
            filtered_segments = [
                s for s in segments if search_query.lower() in s.text.lower()
            ] if search_query else segments

            st.caption(f"Showing {len(filtered_segments)} of {len(segments)} segments")
            for seg in filtered_segments:
                start_fmt = format_timestamp(seg.start, "bracket")
                end_fmt = format_timestamp(seg.end, "bracket")
                st.markdown(
                    f"""
                    <div class="segment-card">
                        <span class="timestamp-badge">{start_fmt} - {end_fmt}</span>
                        <span>{seg.text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_fulltext:
            full_text = " ".join([seg.text for seg in segments])
            st.text_area("Full Transcript", value=full_text, height=350)

        with tab_srt:
            st.text_area("SRT Preview", value=srt_content, height=350)


if __name__ == "__main__":
    main()
