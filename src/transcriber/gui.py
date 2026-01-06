import logging
import os
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox

import whisper
from whisper.utils import get_writer

from transcriber import setup_ffmpeg_path, setup_logging

logger = logging.getLogger(__name__)


class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Open Transcriber")
        self.root.geometry("450x400")

        self.output_dir = None

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        # Header
        self.label_title = tk.Label(
            self.root, text="Audio and Video Transcription through AI", font=("Helvetica", 16, "bold")
        )
        self.label_title.pack(pady=10)

        # File Selection
        self.label_file = tk.Label(self.root, text="No file selected", fg="gray", wraplength=400)
        self.label_file.pack(pady=5)

        self.btn_select = tk.Button(self.root, text="Select Media File", command=self.select_file, height=2, width=20)
        self.btn_select.pack(pady=5)

        self.btn_open = tk.Button(
            self.root, text="📂 Open Output Folder", command=self.open_folder, state=tk.DISABLED, height=2, width=20
        )
        self.btn_open.pack(pady=5)

        # Status
        self.status_label = tk.Label(self.root, text="Ready", fg="blue")
        self.status_label.pack(pady=10)

        self._setup_credits()

    def _setup_credits(self):
        credits_frame = tk.Frame(self.root)
        credits_frame.pack(side=tk.BOTTOM, pady=15)

        # Name
        tk.Label(credits_frame, text="Created by Daniel Collier", font=("Arial", 9, "bold")).pack()

        # Clickable LinkedIn Link
        link_label = tk.Label(
            credits_frame,
            text="linkedin.com/in/danielfcollier",
            font=("Arial", 9, "underline"),
            fg="blue",
            cursor="hand2",
        )
        link_label.pack()

        # Bind the click event to open the browser
        link_label.bind("<Button-1>", lambda e: webbrowser.open_new("https://www.linkedin.com/in/danielfcollier/"))

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=[("Media Files", "*.mp4 *.mp3 *.wav *.mkv *.m4a *.flac *.ogg"), ("All Files", "*.*")],
        )
        if file_path:
            self.label_file.config(text=f"Selected: {os.path.basename(file_path)}")
            # Reset button state on new selection
            self.btn_open.config(state=tk.DISABLED)
            self.start_transcription(file_path)

    def open_folder(self):
        """Opens the output directory in the system file explorer."""
        if self.output_dir and os.path.exists(self.output_dir):
            webbrowser.open(self.output_dir)

    def start_transcription(self, file_path):
        """
        Starts the worker thread to prevent GUI freezing.
        """
        self.btn_select.config(state=tk.DISABLED)
        self.update_status("Starting worker thread...", "orange")
        logger.info(f"Starting transcription thread for {file_path}")

        # Threading: The GUI runs on MainThread, Whisper runs on this new thread.
        thread = threading.Thread(target=self.run_worker, args=(file_path,))
        thread.daemon = True  # Ensure thread dies if app closes
        thread.start()

    def run_worker(self, file_path):
        """
        The heavy lifting (Model loading + Transcription).
        """
        try:
            # 1. Load Model
            self.update_status("Loading Model (Small)...", "blue")
            logger.info("Worker: Loading model...")
            model = whisper.load_model("small")

            # 2. Transcribe
            self.update_status("Transcribing... (This takes time)", "purple")
            logger.info("Worker: Transcribing...")
            result = model.transcribe(file_path, fp16=False)

            # 3. Save
            self.output_dir = os.path.dirname(file_path)
            writer = get_writer("srt", self.output_dir)
            writer(result, file_path)

            logger.info("Worker: Success.")
            self.update_status("Done!", "green")

            # Success actions on Main Thread
            def on_success():
                self.btn_open.config(state=tk.NORMAL)  # Enable the button
                messagebox.showinfo("Success", f"Saved to:\n{self.output_dir}")

            self.root.after(0, on_success)

        except Exception as e:
            logger.error(f"Worker Error: {e}")
            self.update_status("Error", "red")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))  # noqa

        finally:
            # Re-enable select button
            self.root.after(0, lambda: self.btn_select.config(state=tk.NORMAL))

    def update_status(self, text, color):
        """Safely updates the GUI from the worker thread."""
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))


def main():
    setup_logging()
    setup_ffmpeg_path()
    root = tk.Tk()
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
