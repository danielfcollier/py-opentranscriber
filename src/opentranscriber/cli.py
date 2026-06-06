import logging
import os
from enum import Enum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated

import typer
import whisper
from whisper.utils import get_writer

from opentranscriber import setup_logging

logger = logging.getLogger(__name__)

try:
    __version__ = pkg_version("opentranscriber")
except PackageNotFoundError:
    __version__ = "unknown"

_CONTEXT_SETTINGS = {"help_option_names": ["--help", "-h"]}


class ModelSize(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


class OutputFormat(str, Enum):
    txt = "txt"
    srt = "srt"
    vtt = "vtt"
    tsv = "tsv"
    json = "json"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"opentranscriber {__version__}")
        raise typer.Exit()


# Annotated type aliases (gear pattern)
InputFileArg = Annotated[Path, typer.Argument(help="Path to the audio/video file")]
ModelOption = Annotated[ModelSize, typer.Option("--model", "-m", help="Whisper model size (default: base)")]
FormatOption = Annotated[OutputFormat, typer.Option("--format", "-f", help="Output format (default: srt)")]
VadOption = Annotated[bool, typer.Option("--vad/--no-vad", help="Voice Activity Detection (filters silence before transcription)")]
VersionOption = Annotated[bool | None, typer.Option("--version", "-v", callback=_version_callback, is_eager=True, help="Show version and exit.")]

main = typer.Typer(
    help="Local, privacy-focused audio/video transcription powered by Whisper.",
    context_settings=_CONTEXT_SETTINGS,
    no_args_is_help=True,
)


@main.command()
def transcribe(
    input_file: InputFileArg,
    model: ModelOption = ModelSize.base,
    format: FormatOption = OutputFormat.srt,
    vad: VadOption = True,
    version: VersionOption = None,
) -> None:
    """Transcribe an audio or video file."""
    setup_logging()
    try:
        transcribe_media(str(input_file), model.value, format.value, use_vad=vad)
    except KeyboardInterrupt:
        typer.echo("")  # newline after the ^C on the terminal
        if vad:
            typer.secho(
                "Interrupted. Run the same command again to resume from where you left off.",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho("Interrupted.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)
    except (FileNotFoundError, RuntimeError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def transcribe_media(file_path: str, model_type: str, output_format: str, *, use_vad: bool = True) -> None:
    """
    Core transcription logic with optional VAD pre-processing.
    Raises exceptions on failure instead of exiting directly.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    chunks = None
    chunk_results: list = []
    if use_vad:
        from opentranscriber.vad import checkpoint_path, resume_or_detect_and_chunk

        ckpt = checkpoint_path(file_path)
        _, chunks, chunk_results = resume_or_detect_and_chunk(file_path, ckpt)

    logger.info(f"Loading Whisper model: {model_type}...")
    try:
        model = whisper.load_model(model_type)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    logger.info(f"Transcribing '{file_path}'...")
    try:
        if use_vad and chunks:
            from opentranscriber.vad import merge_chunk_results, save_checkpoint

            start_idx = len(chunk_results)
            if start_idx > 0:
                logger.info(f"Resuming from chunk {start_idx + 1}/{len(chunks)}")

            for i, chunk in enumerate(chunks[start_idx:], start_idx + 1):
                logger.info(f"Transcribing chunk {i}/{len(chunks)} [{chunk.start:.1f}s–{chunk.end:.1f}s]...")
                r = model.transcribe(chunk.audio, fp16=False)
                chunk_results.append((r, chunk.start))
                save_checkpoint(ckpt, file_path, chunks, chunk_results)

            result = merge_chunk_results(chunk_results)
            ckpt.unlink(missing_ok=True)
        else:
            # fp16=False is crucial for CPU execution
            result = model.transcribe(file_path, fp16=False)
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

    output_directory = os.path.dirname(file_path) or "."
    logger.info(f"Saving output as {output_format.upper()}...")
    try:
        writer = get_writer(output_format, output_directory)
        writer(result, file_path)
        logger.info(f"Success! Output saved to: {os.path.abspath(output_directory)}")
    except Exception as e:
        raise RuntimeError(f"Failed to save file: {e}") from e
