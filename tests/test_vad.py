"""Unit tests for the VAD pre-processing module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from opentranscriber.vad import SAMPLE_RATE, AudioChunk, build_chunks, detect_and_chunk, detect_speech_segments, merge_chunk_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


# ---------------------------------------------------------------------------
# AudioChunk dataclass
# ---------------------------------------------------------------------------


class TestAudioChunk:
    def test_fields_stored(self):
        audio = np.zeros(160, dtype=np.float32)
        chunk = AudioChunk(audio=audio, start=1.0, end=1.01)
        assert chunk.start == 1.0
        assert chunk.end == 1.01
        assert len(chunk.audio) == 160


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------


class TestBuildChunks:
    def test_empty_segments_returns_empty_list(self):
        assert build_chunks(_silence(60.0), []) == []

    def test_single_segment_below_min_duration_discarded(self):
        audio = _silence(60.0)
        # 3 s < default min_duration (5 s)
        chunks = build_chunks(audio, [{"start": 0.0, "end": 3.0}])
        assert chunks == []

    def test_single_valid_segment_becomes_one_chunk(self):
        audio = _silence(60.0)
        chunks = build_chunks(audio, [{"start": 0.0, "end": 30.0}])
        assert len(chunks) == 1
        assert chunks[0].start == pytest.approx(0.0)
        assert chunks[0].end == pytest.approx(30.0)

    def test_audio_slice_length_matches_time_range(self):
        audio = _silence(60.0)
        chunks = build_chunks(audio, [{"start": 5.0, "end": 40.0}])
        assert len(chunks) == 1
        expected = int((40.0 - 5.0) * SAMPLE_RATE)
        assert len(chunks[0].audio) == expected

    def test_close_segments_merged_into_one_chunk(self):
        audio = _silence(60.0)
        segs = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 10.0}, {"start": 11.0, "end": 15.0}]
        chunks = build_chunks(audio, segs)
        assert len(chunks) == 1
        assert chunks[0].start == pytest.approx(0.0)
        assert chunks[0].end == pytest.approx(15.0)

    def test_chunk_start_offset_preserved(self):
        audio = _silence(200.0)
        chunks = build_chunks(audio, [{"start": 100.0, "end": 140.0}])
        assert len(chunks) == 1
        assert chunks[0].start == pytest.approx(100.0)

    def test_splits_when_projected_span_exceeds_target(self):
        audio = _silence(200.0)
        # First segment alone is already 47 s → adding second pushes span past target(45 s)
        segs = [{"start": 0.0, "end": 47.0}, {"start": 48.0, "end": 90.0}]
        chunks = build_chunks(audio, segs, target_duration=45.0, max_duration=60.0)
        assert len(chunks) == 2
        assert chunks[0].start == pytest.approx(0.0)
        assert chunks[0].end == pytest.approx(47.0)
        assert chunks[1].start == pytest.approx(48.0)
        assert chunks[1].end == pytest.approx(90.0)

    def test_splits_when_projected_span_exceeds_max(self):
        audio = _silence(200.0)
        # span from start to second segment end = 80 s > max (60 s)
        segs = [{"start": 0.0, "end": 50.0}, {"start": 52.0, "end": 80.0}]
        chunks = build_chunks(audio, segs, target_duration=45.0, max_duration=60.0)
        assert len(chunks) == 2

    def test_all_chunk_starts_are_non_negative(self):
        audio = _silence(300.0)
        segs = [{"start": i * 10.0, "end": i * 10.0 + 8.0} for i in range(20)]
        chunks = build_chunks(audio, segs)
        assert all(c.start >= 0.0 for c in chunks)

    def test_chunk_end_does_not_exceed_audio_duration(self):
        duration = 30.0
        audio = _silence(duration)
        segs = [{"start": 25.0, "end": 35.0}]  # end beyond audio length
        chunks = build_chunks(audio, segs)
        if chunks:
            assert chunks[0].end <= duration + 1e-6

    def test_custom_min_duration(self):
        audio = _silence(60.0)
        # 3 s normally discarded; allow it with min_duration=2.0
        chunks = build_chunks(audio, [{"start": 0.0, "end": 3.0}], min_duration=2.0)
        assert len(chunks) == 1

    def test_many_short_segments_create_expected_chunk_count(self):
        # 10 segments × 4 s each with 1 s gaps; total span = 49 s → one chunk
        audio = _silence(200.0)
        segs = [{"start": i * 5.0, "end": i * 5.0 + 4.0} for i in range(10)]
        chunks = build_chunks(audio, segs, target_duration=50.0)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# detect_speech_segments (mocked VAD model)
# ---------------------------------------------------------------------------


class TestDetectSpeechSegments:
    # silero_vad is a MagicMock stub (injected by conftest.py).  We control the
    # return value of get_speech_timestamps via the stub's attribute.

    def test_returns_list(self):
        import silero_vad as _svad

        audio = _silence(5.0)
        _svad.get_speech_timestamps.return_value = []
        result = detect_speech_segments(audio, MagicMock())
        assert isinstance(result, list)

    def test_passes_tensor_and_params_to_vad(self):
        import torch
        import silero_vad as _svad

        audio = _silence(5.0)
        expected = [{"start": 1.0, "end": 4.0}]
        _svad.get_speech_timestamps.return_value = expected

        result = detect_speech_segments(audio, MagicMock())

        args, kwargs = _svad.get_speech_timestamps.call_args
        assert isinstance(args[0], torch.Tensor)
        assert kwargs.get("sampling_rate") == SAMPLE_RATE
        assert kwargs.get("return_seconds") is True
        assert result == expected

    def test_loads_cached_model_when_none_passed(self):
        import silero_vad as _svad

        audio = _silence(2.0)
        _svad.get_speech_timestamps.return_value = []
        with patch("opentranscriber.vad.get_vad_model", return_value=MagicMock()) as mock_get:
            detect_speech_segments(audio, model=None)
            mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# merge_chunk_results
# ---------------------------------------------------------------------------


def _seg(start, end, text, *, seg_id=0, seek=0):
    return {
        "id": seg_id,
        "seek": seek,
        "start": start,
        "end": end,
        "text": text,
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": -0.3,
        "compression_ratio": 1.2,
        "no_speech_prob": 0.05,
    }


class TestMergeChunkResults:
    def test_empty_input_returns_empty_result(self):
        result = merge_chunk_results([])
        assert result["segments"] == []
        assert result["text"] == ""
        assert result["language"] == ""

    def test_single_chunk_no_offset(self):
        chunk_result = {"segments": [_seg(1.0, 3.0, "Hello")], "language": "en"}
        result = merge_chunk_results([(chunk_result, 0.0)])
        assert len(result["segments"]) == 1
        assert result["segments"][0]["start"] == pytest.approx(1.0)
        assert result["segments"][0]["end"] == pytest.approx(3.0)
        assert result["language"] == "en"
        assert result["text"] == "Hello"

    def test_offset_applied_to_timestamps(self):
        chunk_result = {"segments": [_seg(0.5, 2.5, "World")], "language": "en"}
        result = merge_chunk_results([(chunk_result, 100.0)])
        assert result["segments"][0]["start"] == pytest.approx(100.5)
        assert result["segments"][0]["end"] == pytest.approx(102.5)

    def test_two_chunks_merged_in_order(self):
        r1 = {"segments": [_seg(1.0, 3.0, "Hello")], "language": "en"}
        r2 = {"segments": [_seg(0.5, 2.5, "World")], "language": "en"}
        result = merge_chunk_results([(r1, 0.0), (r2, 100.0)])
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Hello"
        assert result["segments"][1]["text"] == "World"
        assert result["segments"][1]["start"] == pytest.approx(100.5)

    def test_segment_ids_are_sequential(self):
        r1 = {"segments": [_seg(0.0, 1.0, "A"), _seg(1.0, 2.0, "B")], "language": "en"}
        r2 = {"segments": [_seg(0.0, 1.0, "C")], "language": "en"}
        result = merge_chunk_results([(r1, 0.0), (r2, 50.0)])
        ids = [s["id"] for s in result["segments"]]
        assert ids == [0, 1, 2]

    def test_full_text_joined_correctly(self):
        r1 = {"segments": [_seg(0.0, 1.0, " Hello ")], "language": "en"}
        r2 = {"segments": [_seg(0.0, 1.0, " World ")], "language": "en"}
        result = merge_chunk_results([(r1, 0.0), (r2, 10.0)])
        assert result["text"] == "Hello World"

    def test_dominant_language_selected(self):
        en1 = {"segments": [_seg(0.0, 1.0, "A")], "language": "en"}
        en2 = {"segments": [_seg(0.0, 1.0, "B")], "language": "en"}
        pt = {"segments": [_seg(0.0, 1.0, "C")], "language": "pt"}
        result = merge_chunk_results([(en1, 0.0), (en2, 5.0), (pt, 10.0)])
        assert result["language"] == "en"

    def test_seek_adjusted_by_offset(self):
        chunk_result = {"segments": [_seg(0.0, 1.0, "A", seek=50)], "language": "en"}
        result = merge_chunk_results([(chunk_result, 10.0)])
        # 10 s × 100 frames/s = 1000 seek units added
        assert result["segments"][0]["seek"] == 50 + 1000


# ---------------------------------------------------------------------------
# detect_and_chunk (main pipeline entry point)
# ---------------------------------------------------------------------------


class TestDetectAndChunk:
    def _make_audio(self, duration_s: float) -> np.ndarray:
        return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)

    def test_fallback_to_single_chunk_when_no_speech(self):
        import silero_vad as _svad

        audio = self._make_audio(30.0)
        _svad.get_speech_timestamps.return_value = []

        with patch("opentranscriber.vad.get_vad_model", return_value=MagicMock()):
            with patch("whisper.load_audio", return_value=audio):
                _, chunks = detect_and_chunk("fake.mp4")

        assert len(chunks) == 1
        assert chunks[0].start == pytest.approx(0.0)
        assert chunks[0].end == pytest.approx(30.0)
        assert len(chunks[0].audio) == len(audio)

    def test_speech_segments_become_chunks(self):
        import silero_vad as _svad

        audio = self._make_audio(120.0)
        _svad.get_speech_timestamps.return_value = [
            {"start": 0.0, "end": 50.0},
            {"start": 70.0, "end": 110.0},
        ]

        with patch("opentranscriber.vad.get_vad_model", return_value=MagicMock()):
            with patch("whisper.load_audio", return_value=audio):
                _, chunks = detect_and_chunk("fake.mp4")

        assert len(chunks) >= 1
        assert all(isinstance(c, AudioChunk) for c in chunks)

    def test_returned_audio_is_full_array(self):
        import silero_vad as _svad

        audio = self._make_audio(60.0)
        _svad.get_speech_timestamps.return_value = [{"start": 5.0, "end": 55.0}]

        with patch("opentranscriber.vad.get_vad_model", return_value=MagicMock()):
            with patch("whisper.load_audio", return_value=audio) as mock_load:
                full_audio, _ = detect_and_chunk("fake.mp4")

        mock_load.assert_called_once_with("fake.mp4")
        assert full_audio is audio

    def test_chunk_slices_match_speech_timestamps(self):
        import silero_vad as _svad

        audio = self._make_audio(60.0)
        _svad.get_speech_timestamps.return_value = [{"start": 10.0, "end": 50.0}]

        with patch("opentranscriber.vad.get_vad_model", return_value=MagicMock()):
            with patch("whisper.load_audio", return_value=audio):
                _, chunks = detect_and_chunk("fake.mp4")

        assert len(chunks) == 1
        assert chunks[0].start == pytest.approx(10.0)
        assert chunks[0].end == pytest.approx(50.0)
        expected_samples = int((50.0 - 10.0) * SAMPLE_RATE)
        assert len(chunks[0].audio) == expected_samples
