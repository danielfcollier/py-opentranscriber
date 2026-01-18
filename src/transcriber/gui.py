import logging
import os
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

import whisper
from whisper.utils import get_writer

from transcriber import setup_ffmpeg_path, setup_logging

logger = logging.getLogger(__name__)


class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Open Transcriber")
        self.root.geometry("500x450")  # Slightly wider for new controls

        self.output_dir = None
        self.cancel_event = threading.Event()
        self.worker_thread = None

        # Model Selection Variable
        self.model_var = tk.StringVar(value="base")

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        # Header
        self.label_title = tk.Label(
            self.root, text="Audio and Video Transcription through AI", font=("Helvetica", 16, "bold")
        )
        self.label_title.pack(pady=10)

        # --- Options Frame ---
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=5)

        tk.Label(options_frame, text="Model Size:").pack(side=tk.LEFT, padx=5)

        # Dropdown for model selection
        models = ["tiny", "base", "small", "medium", "large"]
        self.model_menu = ttk.Combobox(
            options_frame, textvariable=self.model_var, values=models, state="readonly", width=10
        )
        self.model_menu.pack(side=tk.LEFT, padx=5)

        # --- File Selection ---
        self.label_file = tk.Label(self.root, text="No file selected", fg="gray", wraplength=400)
        self.label_file.pack(pady=5)

        # Buttons Frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        self.btn_select = tk.Button(btn_frame, text="Select Media File", command=self.select_file, height=2, width=20)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.btn_cancel = tk.Button(
            btn_frame, text="Cancel", command=self.cancel_transcription, state=tk.DISABLED, height=2, width=10, fg="red"
        )
        self.btn_cancel.pack(side=tk.LEFT, padx=5)

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
            # Reset buttons on new selection
            self.btn_open.config(state=tk.DISABLED)
            self.start_transcription(file_path)

    def open_folder(self):
        """Opens the output directory in the system file explorer."""
        if self.output_dir and os.path.exists(self.output_dir):
            webbrowser.open(self.output_dir)

    def cancel_transcription(self):
        """Signals the worker thread to stop."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_event.set()
            self.update_status("Cancelling...", "red")
            self.btn_cancel.config(state=tk.DISABLED)

    def start_transcription(self, file_path):
        """
        Starts the worker thread to prevent GUI freezing.
        """
        self.btn_select.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.model_menu.config(state="disabled")

        self.cancel_event.clear()
        self.update_status("Starting worker thread...", "orange")

        model_size = self.model_var.get()
        logger.info(f"Starting transcription thread for {file_path} with model {model_size}")

        # Threading: The GUI runs on MainThread, Whisper runs on this new thread.
        self.worker_thread = threading.Thread(target=self.run_worker, args=(file_path, model_size))
        self.worker_thread.daemon = True  # Ensure thread dies if app closes
        self.worker_thread.start()

    def run_worker(self, file_path, model_size):
        """
        The heavy lifting (Model loading + Transcription).
        Includes checks for cancellation flag.
        """
        try:
            if self.cancel_event.is_set():
                raise InterruptedError()

            # 1. Load Model
            self.update_status(f"Loading Model ({model_size})...", "blue")
            logger.info("Worker: Loading model...")
            model = whisper.load_model(model_size)

            if self.cancel_event.is_set():
                raise InterruptedError()

            # 2. Transcribe
            self.update_status("Transcribing... (This takes time)", "purple")
            logger.info("Worker: Transcribing...")

            # Note: This specific call blocks the thread. We check cancel after it finishes.
            result = model.transcribe(file_path, fp16=False)

            if self.cancel_event.is_set():
                raise InterruptedError()

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

            if not self.cancel_event.is_set():
                self.root.after(0, on_success)

        except InterruptedError:
            logger.warning("Worker: Transcription cancelled by user.")
            self.update_status("Cancelled", "red")

        except Exception as e:
            logger.error(f"Worker Error: {e}")
            self.update_status("Error", "red")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))  # noqa

        finally:
            # Re-enable controls
            def reset_ui():
                self.btn_select.config(state=tk.NORMAL)
                self.btn_cancel.config(state=tk.DISABLED)
                self.model_menu.config(state="readonly")

            self.root.after(0, reset_ui)

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
