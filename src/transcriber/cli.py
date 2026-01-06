import argparse
import logging
import os
import sys

import whisper
from whisper.utils import get_writer

from transcriber import setup_logging

logger = logging.getLogger(__name__)


def transcribe_media(file_path, model_type, output_format):
    """
    Core transcription logic.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    logger.info(f"Loading Whisper model: {model_type}...")
    try:
        # fp16=False is safer for CPU usage to avoid warnings
        model = whisper.load_model(model_type)
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        sys.exit(1)

    logger.info(f"Transcribing '{file_path}'...")
    try:
        result = model.transcriber(file_path, fp16=False)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)

    output_directory = os.path.dirname(file_path) or "."
    logger.info(f"Saving output as {output_format.upper()}...")

    try:
        writer = get_writer(output_format, output_directory)
        writer(result, file_path)
        logger.info(f"Success! Output saved to: {os.path.abspath(output_directory)}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")


def main():
    """
    Entry point for the CLI.
    """
    setup_logging()

    parser = argparse.ArgumentParser(description="Professional Video Transcription CLI")
    parser.add_argument("input_file", help="Path to the audio/video file")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size (default: base)",
    )
    parser.add_argument(
        "--format", default="srt", choices=["txt", "srt", "vtt", "tsv", "json"], help="Output format (default: srt)"
    )

    args = parser.parse_args()
    transcribe_media(args.input_file, args.model, args.format)


if __name__ == "__main__":
    main()
