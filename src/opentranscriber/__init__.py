import logging
import os
import sys


def setup_logging(level=logging.INFO):
    """
    Configures the root logger with a professional format.
    """
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        root_logger.addHandler(handler)


def setup_ffmpeg_path():
    """
    Crucial for the .exe: Adds the bundled FFmpeg to the system PATH
    so Whisper (subprocess) can find it.
    """
    if getattr(sys, "frozen", False):
        # PyInstaller unpacks everything to sys._MEIPASS
        base_path = sys._MEIPASS

        # Add this temporary folder to the PATH
        os.environ["PATH"] += os.pathsep + base_path

        # Verify it works (Optional logging)
        logging.info(f"Bundled environment detected. Added to PATH: {base_path}")
