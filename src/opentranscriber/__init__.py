import logging
import os
import sys


class _NullWriter:
    """A no-op stream, used to stand in for a missing console stream."""

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass


def setup_streams():
    """
    In a --windowed build, sys.stdout/sys.stderr are None (no attached
    console). Anything that writes to them directly — print(), the
    `warnings` module, tqdm — would crash with AttributeError. Standing in
    a no-op stream fixes all of those at once, not just our own logging.
    """
    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()


def setup_logging(level=logging.INFO):
    """
    Configures the root logger with a professional format.
    """
    setup_streams()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
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


def suppress_ffmpeg_console():
    """
    On Windows, subprocess calls made from a --windowed (console-less) app
    can briefly flash a console window. Patches whisper's internal ffmpeg
    call to launch without one.
    """
    if sys.platform != "win32":
        return

    import subprocess

    import whisper.audio as whisper_audio

    original_run = subprocess.run

    def run_without_console(*args, **kwargs):
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        return original_run(*args, **kwargs)

    whisper_audio.run = run_without_console
