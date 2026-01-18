# 🎙️ OpenTranscriber

> **A private, offline, and open-source audio/video transcription tool.**

OpenTranscriber is a powerful desktop application built with Python and OpenAI's **Whisper** technology. It allows you to transcribe video and audio files locally on your machine—ensuring your data never leaves your computer.

![License](https://img.shields.io/github/license/danielfcollier/py-transcriber)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## ✨ Features

* **🔒 100% Privacy:** Runs entirely offline. No data is sent to the cloud.
* **🧠 AI-Powered:** Uses OpenAI's state-of-the-art **Whisper** models (Tiny, Base, Small, Medium, Large).
* **🎞️ Media Support:** Works with MP4, MP3, WAV, MKV, FLAC, OGG, and M4A.
* **📝 Built-in Editor:**
    * **Live Preview:** Listen to the audio while editing segments.
    * **Pagination:** Handle large files smoothly without freezing.
    * **Timeline Seeking:** Click and drag to jump to specific parts of the audio.
* **⚙️ Formats:** Export to **SRT** (Subtitles) or **TXT** (Plain Text).
* **💻 Cross-Platform:** Designed for Windows, Linux, and macOS.

## 🚀 Installation

### Prerequisites
* **Python 3.12+**
* **FFmpeg** (Required for audio processing)
    * *Linux:* `sudo apt install ffmpeg`
    * *Mac:* `brew install ffmpeg`
    * *Windows:* Download from [ffmpeg.org](https://ffmpeg.org/download.html) (or use `choco install ffmpeg`).

### 1. Clone the Repository
```bash
git clone https://github.com/danielfcollier/py-opentranscriber.git
cd py-opentranscriber
```

### 2. Install Dependencies (using `uv`)

We use [uv](https://github.com/astral-sh/uv) for fast package management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

## 🖥️ Usage

You can run the application in two modes: **GUI (Graphic Interface)** or **CLI (Command Line)**.

### 🎨 Graphical Interface (Recommended)

Launch the visual tool to select files, choose models, and edit transcripts.

```bash
make run-gui
# OR
uv run opentranscriber-gui
```

**How to use:**

1. Click **Select Media File**.
2. Choose the **Model Size** (Base is fast, Large is accurate).
3. Click **Start Transcription**.
4. Once finished, the **Editor** will open.
5. Play the audio, correct the text, and click **Save & Finish**.

### ⌨️ Command Line Interface

For automation or headless servers.

```bash
make run-cli input_file="video.mp4"
# OR
uv run opentranscriber-cli "video.mp4" --model small --format srt
```

**Arguments:**

* `input_file`: Path to the file.
* `--model`: `tiny`, `base`, `small`, `medium`, `large` (default: base).
* `--format`: `srt` or `txt` (default: srt).

## 🛠️ Development

### Project Structure

```text
src/transcriber/
├── gui.py       # Main GUI application (Tkinter)
├── cli.py       # Command Line Interface logic
├── __main__.py  # Entry point
└── ...
```

### Running Tests and Linters

```bash
# Run Linting (Ruff)
make lint

# Run Security Scan (Trivy)
trivy fs .
```

## ☕ Support

If this tool saved you time, consider buying me a coffee!

* **🇧🇷 Brazilians (Pix):** [LivePix](https://livepix.gg/danielcollier)
* **🌍 International:** [BuyMeACoffee](https://www.buymeacoffee.com/danielcollier)


## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
