"""CLI tests using typer's CliRunner."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestHelp:
    def test_help_long_flag(self, runner, cli_app):
        result = runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_help_short_flag(self, runner, cli_app):
        result = runner.invoke(cli_app, ["-h"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_no_args_shows_help(self, runner, cli_app):
        # no_args_is_help=True exits 2 (missing required arg), not 0
        result = runner.invoke(cli_app, [])
        assert result.exit_code == 2
        assert "Usage:" in result.output


class TestVersion:
    def test_version_long_flag(self, runner, cli_app):
        result = runner.invoke(cli_app, ["--version"])
        assert result.exit_code == 0
        assert "opentranscriber" in result.output

    def test_version_short_flag(self, runner, cli_app):
        result = runner.invoke(cli_app, ["-v"])
        assert result.exit_code == 0
        assert "opentranscriber" in result.output


class TestValidation:
    def test_invalid_model_rejected(self, runner, cli_app, tmp_path):
        f = tmp_path / "audio.mp4"
        f.touch()
        result = runner.invoke(cli_app, [str(f), "--model", "xlarge"])
        assert result.exit_code == 2

    def test_invalid_format_rejected(self, runner, cli_app, tmp_path):
        f = tmp_path / "audio.mp4"
        f.touch()
        result = runner.invoke(cli_app, [str(f), "--format", "docx"])
        assert result.exit_code == 2

    def test_missing_file_exits_1(self, runner, cli_app):
        result = runner.invoke(cli_app, ["nonexistent_file.mp4"])
        assert result.exit_code == 1
        assert "nonexistent_file.mp4" in result.output


class TestTranscription:
    def test_vad_enabled_by_default(self, runner, cli_app, tmp_path):
        f = tmp_path / "audio.mp4"
        f.touch()
        with patch("opentranscriber.cli.transcribe_media") as mock_t:
            result = runner.invoke(cli_app, [str(f)])
        assert result.exit_code == 0
        mock_t.assert_called_once_with(str(f), "base", "srt", use_vad=True)

    def test_no_vad_flag_disables_vad(self, runner, cli_app, tmp_path):
        f = tmp_path / "audio.mp4"
        f.touch()
        with patch("opentranscriber.cli.transcribe_media") as mock_t:
            result = runner.invoke(cli_app, [str(f), "--no-vad"])
        assert result.exit_code == 0
        mock_t.assert_called_once_with(str(f), "base", "srt", use_vad=False)

    def test_model_and_format_options_forwarded(self, runner, cli_app, tmp_path):
        f = tmp_path / "audio.mp4"
        f.touch()
        with patch("opentranscriber.cli.transcribe_media") as mock_t:
            result = runner.invoke(cli_app, [str(f), "--model", "large", "--format", "vtt"])
        assert result.exit_code == 0
        mock_t.assert_called_once_with(str(f), "large", "vtt", use_vad=True)

    def test_short_option_aliases(self, runner, cli_app, tmp_path):
        f = tmp_path / "audio.mp4"
        f.touch()
        with patch("opentranscriber.cli.transcribe_media") as mock_t:
            result = runner.invoke(cli_app, [str(f), "-m", "small", "-f", "txt"])
        assert result.exit_code == 0
        mock_t.assert_called_once_with(str(f), "small", "txt", use_vad=True)

    def test_runtime_error_exits_1(self, runner, cli_app, tmp_path):
        f = tmp_path / "audio.mp4"
        f.touch()
        with patch("opentranscriber.cli.transcribe_media", side_effect=RuntimeError("model load failed")):
            result = runner.invoke(cli_app, [str(f)])
        assert result.exit_code == 1
