import logging
import os
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk

import pygame
import whisper

# We import this just to ensure the module is loaded in sys.modules
import whisper.transcribe
from whisper.utils import get_writer

from opentranscriber import setup_ffmpeg_path, setup_logging

logger = logging.getLogger(__name__)


class TkinterTqdm:
    """
    A specific tqdm-like class that redirects progress to a Tkinter progress bar.
    """

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total", 100)
        self.current = 0
        self.unit = kwargs.get("unit", "it")
        self.on_progress = getattr(TkinterTqdm, "on_progress_callback", None)

    def update(self, n=1):
        self.current += n
        if self.on_progress:
            self.on_progress(self.current, self.total)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TqdmShim:
    """Shim for whisper versions that import tqdm as a module."""

    tqdm = TkinterTqdm


class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Open Transcriber")
        self.root.geometry("800x800")  # Wider for the editor

        # State Variables
        self.output_dir = None
        self.audio_path = None
        self.transcription_result = None
        self.segment_widgets = []

        # Pagination & Audio State
        self.current_page = 0
        self.PAGE_SIZE = 50
        self.audio_total_duration = 0
        self.audio_start_offset = 0  # To track seek position
        self.is_user_seeking = False

        # Threading
        self.cancel_event = threading.Event()
        self.worker_thread = None

        # Config
        self.model_var = tk.StringVar(value="base")
        self.format_var = tk.StringVar(value="srt")

        # Start with the Main Menu
        self.setup_main_menu()

    # =========================================================================
    # VIEW 1: Main Menu (Selection & Config)
    # =========================================================================
    def setup_main_menu(self):
        self._clear_window()

        # Header
        tk.Label(
            self.root, text="Audio and Video Transcription through AI", font=("Helvetica", 16, "bold")
        ).pack(pady=10)

        # Options
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=5)

        tk.Label(options_frame, text="Model Size:").pack(side=tk.LEFT, padx=5)
        self.model_menu = ttk.Combobox(
            options_frame,
            textvariable=self.model_var,
            values=["tiny", "base", "small"],
            state="readonly",
            width=10,
        )
        self.model_menu.pack(side=tk.LEFT, padx=5)

        tk.Label(options_frame, text="Format:").pack(side=tk.LEFT, padx=5)
        self.format_menu = ttk.Combobox(
            options_frame, textvariable=self.format_var, values=["srt", "txt"], state="readonly", width=6
        )
        self.format_menu.pack(side=tk.LEFT, padx=5)

        # File Selection
        self.label_file = tk.Label(self.root, text="No file selected", fg="gray", wraplength=400)
        self.label_file.pack(pady=5)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        self.btn_select = tk.Button(btn_frame, text="Select Media File", command=self.select_file, height=2, width=20)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.btn_cancel = tk.Button(
            btn_frame, text="Cancel", command=self.cancel_transcription, state=tk.DISABLED, height=2, width=10, fg="red"
        )
        self.btn_cancel.pack(side=tk.LEFT, padx=5)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        # Hidden by default

        # Status
        self.status_label = tk.Label(self.root, text="Ready", fg="blue")
        self.status_label.pack(pady=10)

        self._setup_credits()

    def _setup_credits(self):
        credits_frame = tk.Frame(self.root)
        credits_frame.pack(side=tk.BOTTOM, pady=20)

        tk.Label(credits_frame, text="Created by Daniel Collier", font=("Arial", 9, "bold")).pack()

        link_linkedin = tk.Label(
            credits_frame,
            text="linkedin.com/in/danielfcollier",
            font=("Arial", 9, "underline"),
            fg="blue",
            cursor="hand2",
        )
        link_linkedin.pack(pady=(0, 15))
        link_linkedin.bind("<Button-1>", lambda e: webbrowser.open("https://www.linkedin.com/in/danielfcollier/"))

        # Support Section
        tk.Label(credits_frame, text="Para Brasileiros (Pix)", font=("Arial", 8, "bold")).pack()
        link_pix = tk.Label(
            credits_frame, text="☕ Me dê um café", font=("Arial", 9, "underline"), fg="blue", cursor="hand2"
        )
        link_pix.pack(pady=(0, 10))
        link_pix.bind("<Button-1>", lambda e: webbrowser.open("https://livepix.gg/danielcollier"))

        tk.Label(credits_frame, text="International Support", font=("Arial", 8, "bold")).pack()
        link_coffee = tk.Label(
            credits_frame, text="☕ Buy me a coffee", font=("Arial", 9, "underline"), fg="blue", cursor="hand2"
        )
        link_coffee.pack()
        link_coffee.bind("<Button-1>", lambda e: webbrowser.open("https://www.buymeacoffee.com/danielcollier"))

    # =========================================================================
    # VIEW 2: Editor (Paginated & Slider)
    # =========================================================================
    def setup_editor_ui(self):
        self._clear_window()
        self.root.title(f"Editor - {os.path.basename(self.audio_path)}")

        # Calculate Duration
        segments = self.transcription_result.get("segments", [])
        if segments:
            self.audio_total_duration = segments[-1]["end"]
        else:
            self.audio_total_duration = 100  # Fallback

        # 1. Header (Controls + Slider)
        header_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        header_frame.pack(fill="x")

        # Top Row: Buttons
        btn_row = tk.Frame(header_frame, bg="#f0f0f0")
        btn_row.pack(fill="x", padx=10, pady=5)

        tk.Button(btn_row, text="⏸ Pause/Play", command=self.pause_audio).pack(side=tk.LEFT)

        # Pagination Controls
        tk.Button(btn_row, text="< Prev Page", command=self.prev_page).pack(side=tk.LEFT, padx=20)
        self.lbl_page = tk.Label(btn_row, text="Page 1", bg="#f0f0f0")
        self.lbl_page.pack(side=tk.LEFT)
        tk.Button(btn_row, text="Next Page >", command=self.next_page).pack(side=tk.LEFT, padx=20)

        tk.Button(btn_row, text="💾 Save & Finish", command=self.save_edits, bg="#ddffdd").pack(side=tk.RIGHT)

        # Bottom Row: Slider
        slider_row = tk.Frame(header_frame, bg="#f0f0f0")
        slider_row.pack(fill="x", padx=10)

        tk.Label(slider_row, text="Timeline:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.seek_var = tk.DoubleVar()
        self.slider = ttk.Scale(
            slider_row,
            from_=0,
            to=self.audio_total_duration,
            orient="horizontal",
            variable=self.seek_var,
            command=self.on_slider_drag,
        )
        self.slider.pack(side=tk.LEFT, fill="x", expand=True, padx=10)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        self.lbl_time = tk.Label(slider_row, text="00:00", bg="#f0f0f0", width=6)
        self.lbl_time.pack(side=tk.LEFT)

        # 2. Editor Area (Scrollable)
        canvas = tk.Canvas(self.root)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 3. Initialize Audio
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(self.audio_path)
            self.update_slider_loop()  # Start the UI updater
        except Exception as e:
            messagebox.showwarning("Audio Error", f"Could not load audio: {e}")

        # 4. Render First Page
        self.current_page = 0
        self.render_page()

    def render_page(self):
        """Renders only a slice of segments to prevent UI freezing."""
        # 1. Clear existing widgets in the scroll frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.segment_widgets = []  # Reset widget list for THIS page (We need a strategy for full save)

        # Note: We need to store edits globally, not just in widgets.
        # But for simplicity in this pagination fix, we will commit edits to memory
        # before changing pages. *Implemented in change_page logic*

        segments = self.transcription_result.get("segments", [])
        start_idx = self.current_page * self.PAGE_SIZE
        end_idx = start_idx + self.PAGE_SIZE
        page_segments = segments[start_idx:end_idx]

        self.lbl_page.config(text=f"Page {self.current_page + 1} / {len(segments) // self.PAGE_SIZE + 1}")

        for i, seg in enumerate(page_segments):
            # global_index is needed to map back to main list
            self._create_segment_row(start_idx + i, seg)

    def _create_segment_row(self, global_index, segment):
        row = tk.Frame(self.scrollable_frame, pady=5)
        row.pack(fill="x", padx=5)

        start_str = self._format_time(segment["start"])
        lbl_time = tk.Label(row, text=f"{start_str}", font=("Consolas", 8), width=6)
        lbl_time.pack(side=tk.LEFT)

        btn_play = tk.Button(row, text="▶", command=lambda s=segment["start"]: self.play_segment(s))
        btn_play.pack(side=tk.LEFT, padx=5)

        txt = tk.Text(row, height=3, width=50, wrap="word", font=("Arial", 10))
        txt.insert("1.0", segment["text"].strip())
        txt.pack(side=tk.LEFT, fill="x", expand=True, padx=5)

        # We assume segment_widgets corresponds to the CURRENT PAGE indices
        self.segment_widgets.append({"widget": txt, "global_index": global_index})

    def commit_page_edits(self):
        """Saves text from current page widgets back to the main memory structure."""
        segments = self.transcription_result["segments"]
        for item in self.segment_widgets:
            idx = item["global_index"]
            widget = item["widget"]
            # Save back to memory
            segments[idx]["text"] = widget.get("1.0", "end-1c")

    def next_page(self):
        self.commit_page_edits()
        total_pages = len(self.transcription_result["segments"]) // self.PAGE_SIZE
        if self.current_page < total_pages:
            self.current_page += 1
            self.render_page()

    def prev_page(self):
        self.commit_page_edits()
        if self.current_page > 0:
            self.current_page -= 1
            self.render_page()

    # =========================================================================
    # Audio Logic: Seeking & Updates
    # =========================================================================
    def update_slider_loop(self):
        """Updates the slider position based on audio playback."""
        if pygame.mixer.music.get_busy() and not self.is_user_seeking:
            # Pygame get_pos returns milliseconds played SINCE LAST PLAY command
            current_play_time = pygame.mixer.music.get_pos() / 1000.0
            total_time = self.audio_start_offset + current_play_time

            self.seek_var.set(total_time)
            self.lbl_time.config(text=self._format_time(total_time))

        # Schedule next check
        if hasattr(self, "slider"):  # Check if UI still exists
            self.root.after(500, self.update_slider_loop)

    def on_slider_drag(self, value):
        self.is_user_seeking = True

    def on_slider_release(self, event):
        """User finished dragging: Seek audio."""
        seek_time = self.seek_var.get()
        self.play_segment(seek_time)
        self.is_user_seeking = False

    def play_segment(self, start_time):
        try:
            self.audio_start_offset = float(start_time)
            pygame.mixer.music.play(start=self.audio_start_offset)
            self.seek_var.set(self.audio_start_offset)
        except Exception as e:
            logger.error(f"Audio error: {e}")

    def pause_audio(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()

    # =========================================================================
    # Logic: Transcription
    # =========================================================================
    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Media", "*.mp4 *.mp3 *.wav *.mkv *.m4a *.ogg")])
        if path:
            self.label_file.config(text=f"Selected: {os.path.basename(path)}")
            self.start_transcription(path)

    def start_transcription(self, file_path):
        self.btn_select.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.model_menu.config(state="disabled")
        self.format_menu.config(state="disabled")

        # Setup Progress Bar
        self.progress.pack(pady=5, before=self.status_label)
        self.progress["value"] = 0
        self.progress["maximum"] = 100

        self.cancel_event.clear()
        self.update_status("Starting worker...", "orange")

        model_size = self.model_var.get()
        self.worker_thread = threading.Thread(target=self.run_worker, args=(file_path, model_size))
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _update_progress_bar(self, current, total):
        def _update():
            if total:
                self.progress["maximum"] = total
                self.progress["value"] = current
                percent = int((current / total) * 100)
                self.status_label.config(text=f"Transcribing... {percent}%")

        self.root.after(0, _update)

    def run_worker(self, file_path, model_size):
        try:
            if self.cancel_event.is_set():
                raise InterruptedError()

            self.update_status(f"Loading Model ({model_size})...", "blue")
            logger.info("Worker: Loading model...")
            model = whisper.load_model(model_size)

            if self.cancel_event.is_set():
                raise InterruptedError()

            self.update_status("Transcribing...", "purple")
            logger.info("Worker: Transcribing...")

            # --- MONKEY PATCH START ---
            TkinterTqdm.on_progress_callback = self._update_progress_bar
            transcribe_module = sys.modules["whisper.transcribe"]
            original_tqdm = transcribe_module.tqdm

            if hasattr(original_tqdm, "tqdm"):
                transcribe_module.tqdm = TqdmShim
            else:
                transcribe_module.tqdm = TkinterTqdm

            try:
                result = model.transcribe(file_path, fp16=False)
            finally:
                transcribe_module.tqdm = original_tqdm
            # --- MONKEY PATCH END ---

            if self.cancel_event.is_set():
                raise InterruptedError()

            self.transcription_result = result
            self.audio_path = file_path

            logger.info("Worker: Transcription complete. Opening editor.")
            self.root.after(0, self.setup_editor_ui)

        except InterruptedError:
            self.update_status("Cancelled", "red")
            self.root.after(0, self._reset_main_ui)
        except Exception as e:
            logger.error(f"Worker Error: {e}")
            self.update_status("Error", "red")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))  # noqa
            self.root.after(0, self._reset_main_ui)

    # =========================================================================
    # Helpers & Saving
    # =========================================================================
    def save_edits(self):
        try:
            # Important: Commit the edits from the CURRENT page before saving
            self.commit_page_edits()

            output_format = self.format_var.get()
            output_dir = os.path.dirname(self.audio_path)

            writer = get_writer(output_format, output_dir)
            writer(self.transcription_result, self.audio_path)

            messagebox.showinfo("Success", f"Saved to:\n{output_dir}")

            pygame.mixer.music.stop()
            pygame.mixer.quit()
            self.setup_main_menu()

        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def _reset_main_ui(self):
        try:
            self.progress.pack_forget()
            self.btn_select.config(state=tk.NORMAL)
            self.btn_cancel.config(state=tk.DISABLED)
            self.model_menu.config(state="readonly")
            self.format_menu.config(state="readonly")
        except Exception:
            pass

    def update_status(self, text, color):
        if hasattr(self, "status_label"):
            self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    def _format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def cancel_transcription(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_event.set()
            self.update_status("Cancelling...", "red")
            self.btn_cancel.config(state=tk.DISABLED)


def main():
    setup_logging()
    setup_ffmpeg_path()
    root = tk.Tk()
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
