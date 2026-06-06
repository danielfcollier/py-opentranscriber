"""Voice Activity Detection pre-processing for opentranscriber."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import whisper

# silero_vad is imported lazily inside functions so that the module can be
# loaded in CPU-only test environments without triggering torchaudio's CUDA
# extension. The actual library is required at runtime.

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_VAD_MODEL: torch.nn.Module | None = None


def get_vad_model() -> torch.nn.Module:
    """Return the cached silero-vad model, loading it on first call."""
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad

        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


@dataclass
class AudioChunk:
    """A speech-only audio slice with its offset in the original timeline."""

    audio: np.ndarray  # float32 mono at SAMPLE_RATE Hz
    start: float  # seconds offset in the original file
    end: float  # seconds offset in the original file


def detect_speech_segments(
    audio: np.ndarray,
    model: torch.nn.Module | None = None,
    *,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
    speech_pad_ms: int = 30,
    threshold: float = 0.5,
) -> list[dict]:
    """Return VAD speech timestamps (seconds) for a mono 16 kHz float32 array."""
    if model is None:
        model = get_vad_model()
    from silero_vad import get_speech_timestamps

    wav = torch.from_numpy(audio)
    return get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        threshold=threshold,
    )


def build_chunks(
    audio: np.ndarray,
    speech_segments: list[dict],
    *,
    target_duration: float = 45.0,
    max_duration: float = 60.0,
    min_duration: float = 5.0,
) -> list[AudioChunk]:
    """
    Group VAD speech segments into chunks split at natural silence boundaries.

    Targets ~45 s chunks (configurable), hard-caps at 60 s, and discards
    results shorter than 5 s to avoid sending near-empty audio to Whisper.
    """
    if not speech_segments:
        return []

    # Accumulate segments into contiguous (start, end) time ranges
    ranges: list[tuple[float, float]] = []
    chunk_start = speech_segments[0]["start"]
    chunk_end = speech_segments[0]["end"]

    for seg in speech_segments[1:]:
        projected_span = seg["end"] - chunk_start
        if projected_span > max_duration or projected_span >= target_duration:
            ranges.append((chunk_start, chunk_end))
            chunk_start = seg["start"]
        chunk_end = seg["end"]
    ranges.append((chunk_start, chunk_end))

    # Convert time ranges to AudioChunk objects, dropping chunks that are too short
    total_duration = len(audio) / SAMPLE_RATE
    chunks: list[AudioChunk] = []
    for start, end in ranges:
        if end - start < min_duration:
            logger.debug("Discarding %.1f s chunk [%.1f–%.1f s]", end - start, start, end)
            continue
        end = min(end, total_duration)
        s = int(start * SAMPLE_RATE)
        e = int(end * SAMPLE_RATE)
        chunks.append(AudioChunk(audio=audio[s:e], start=start, end=end))

    return chunks


def merge_chunk_results(chunk_results: list[tuple[dict, float]]) -> dict:
    """
    Merge per-chunk Whisper results into a single timeline-adjusted result.

    Each entry in chunk_results is (whisper_result, chunk_start_offset_seconds).
    Segment timestamps are shifted by the chunk's start offset so they align
    with the original file's timeline.
    """
    all_segments: list[dict] = []
    language_counts: dict[str, int] = {}
    seg_id = 0

    for result, offset in chunk_results:
        lang = result.get("language", "")
        language_counts[lang] = language_counts.get(lang, 0) + 1

        # Each Whisper seek value is in units of 10 ms (HOP_LENGTH=160 @ 16 kHz)
        seek_offset = int(offset * 100)
        for seg in result.get("segments", []):
            all_segments.append(
                {
                    "id": seg_id,
                    "seek": seg.get("seek", 0) + seek_offset,
                    "start": seg["start"] + offset,
                    "end": seg["end"] + offset,
                    "text": seg["text"],
                    "tokens": seg.get("tokens", []),
                    "temperature": seg.get("temperature", 0.0),
                    "avg_logprob": seg.get("avg_logprob", 0.0),
                    "compression_ratio": seg.get("compression_ratio", 0.0),
                    "no_speech_prob": seg.get("no_speech_prob", 0.0),
                }
            )
            seg_id += 1

    full_text = " ".join(s["text"].strip() for s in all_segments)
    dominant_lang = max(language_counts, key=lambda k: language_counts[k]) if language_counts else ""
    return {"text": full_text, "segments": all_segments, "language": dominant_lang}


def detect_and_chunk(
    file_path: str,
    vad_model: torch.nn.Module | None = None,
) -> tuple[np.ndarray, list[AudioChunk]]:
    """
    Load audio, run VAD, and return (full_audio, speech_chunks).

    Falls back to a single full-file chunk when no speech is detected so
    that the rest of the pipeline can proceed without special-casing.
    """
    logger.info("Loading audio for VAD analysis…")
    audio = whisper.load_audio(file_path)  # float32 mono at 16 kHz

    if vad_model is None:
        logger.info("Loading silero-vad model…")
        vad_model = get_vad_model()

    logger.info("Running Voice Activity Detection…")
    segments = detect_speech_segments(audio, vad_model)

    if not segments:
        logger.warning("VAD found no speech; falling back to full-file transcription")
        total = len(audio) / SAMPLE_RATE
        return audio, [AudioChunk(audio=audio, start=0.0, end=total)]

    total = len(audio) / SAMPLE_RATE
    speech_s = sum(s["end"] - s["start"] for s in segments)
    logger.info(
        "VAD: %d speech segments, %.0f s / %.0f s total (%.0f%% speech)",
        len(segments),
        speech_s,
        total,
        100 * speech_s / total,
    )

    chunks = build_chunks(audio, segments)
    logger.info("Created %d chunks for transcription", len(chunks))
    return audio, chunks


# ---------------------------------------------------------------------------
# Checkpoint helpers — save/resume transcription progress across interruptions
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalars and arrays that Whisper may embed in results."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def checkpoint_path(file_path: str) -> Path:
    """Return the checkpoint file path for a given input file."""
    p = Path(file_path).resolve()
    return p.with_name(p.name + ".checkpoint.json")


def save_checkpoint(
    ckpt_path: Path,
    file_path: str,
    all_chunks: list[AudioChunk],
    completed: list[tuple[dict, float]],
) -> None:
    """Atomically persist transcription progress to disk.

    Writes to a .tmp file first then renames so an interrupted save never
    leaves a half-written checkpoint.
    """
    data = {
        "input_file": file_path,
        "total_chunks": len(all_chunks),
        "vad_chunks": [{"start": c.start, "end": c.end} for c in all_chunks],
        "completed": [{"index": i, "offset": offset, "result": result} for i, (result, offset) in enumerate(completed)],
    }
    tmp = ckpt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, cls=_NumpyEncoder), encoding="utf-8")
    tmp.rename(ckpt_path)


def resume_or_detect_and_chunk(
    file_path: str,
    ckpt_path: Path,
    vad_model: torch.nn.Module | None = None,
) -> tuple[np.ndarray, list[AudioChunk], list[tuple[dict, float]]]:
    """Load audio and return (audio, all_chunks, already_completed_results).

    If a valid checkpoint exists for *file_path*, the VAD model is never loaded
    and chunk timestamps are reconstructed from the checkpoint — saving several
    minutes on long files.  Otherwise a fresh VAD analysis is run and an empty
    completed list is returned.
    """
    logger.info("Loading audio…")
    audio = whisper.load_audio(file_path)
    total_duration = len(audio) / SAMPLE_RATE

    # --- try to resume from checkpoint ---
    if ckpt_path.exists():
        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            all_chunks = [
                AudioChunk(
                    audio=audio[int(c["start"] * SAMPLE_RATE) : int(min(c["end"], total_duration) * SAMPLE_RATE)],
                    start=c["start"],
                    end=min(c["end"], total_duration),
                )
                for c in data["vad_chunks"]
            ]
            completed: list[tuple[dict, float]] = [(c["result"], c["offset"]) for c in data["completed"]]
            logger.info(
                "Checkpoint found: resuming from chunk %d/%d",
                len(completed) + 1,
                len(all_chunks),
            )
            return audio, all_chunks, completed
        except Exception as exc:
            logger.warning("Ignoring unreadable checkpoint (%s); running fresh VAD", exc)

    # --- fresh VAD analysis ---
    if vad_model is None:
        logger.info("Loading silero-vad model…")
        vad_model = get_vad_model()

    logger.info("Running Voice Activity Detection…")
    segments = detect_speech_segments(audio, vad_model)

    if not segments:
        logger.warning("VAD found no speech; falling back to full-file transcription")
        return audio, [AudioChunk(audio=audio, start=0.0, end=total_duration)], []

    speech_s = sum(s["end"] - s["start"] for s in segments)
    logger.info(
        "VAD: %d speech segments, %.0f s / %.0f s total (%.0f%% speech)",
        len(segments),
        speech_s,
        total_duration,
        100 * speech_s / total_duration,
    )

    all_chunks = build_chunks(audio, segments)
    logger.info("Created %d chunks for transcription", len(all_chunks))
    return audio, all_chunks, []
