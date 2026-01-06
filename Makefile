################################################################################
# Transcriber Project Makefile (Cross-Platform)
################################################################################

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON  := .venv/bin/python3
UV      := uv

# Directories
SRC_DIR := src/transcriber
DIST_DIR := dist
BUILD_DIR := build

# --- OS DETECTION ---
# We detect the OS to choose the correct FFmpeg URL and PyInstaller separators.
OS_NAME := $(shell uname -s 2>/dev/null || echo Windows)

ifeq ($(OS_NAME), Linux)
    # Linux Settings
    FFMPEG_URL := https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    FFMPEG_BIN := ffmpeg
    FFMPEG_EXT := 
    PYINSTALLER_SEP := :
    EXE_EXT := 
    # Command to extract tar.xz
    EXTRACT_CMD := tar -xf ffmpeg-archive -C . --strip-components=1 --wildcards "*/ffmpeg"
else ifeq ($(OS_NAME), Darwin)
    # MacOS Settings
    FFMPEG_URL := https://evermeet.cx/ffmpeg/getrelease/zip
    FFMPEG_BIN := ffmpeg
    FFMPEG_EXT := 
    PYINSTALLER_SEP := :
    EXE_EXT := 
    EXTRACT_CMD := unzip -o ffmpeg-archive -d . && chmod +x ffmpeg
else
    # Windows Settings (Default fallback)
    FFMPEG_URL := https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
    FFMPEG_BIN := ffmpeg.exe
    FFMPEG_EXT := .exe
    PYINSTALLER_SEP := ;
    EXE_EXT := .exe
    EXTRACT_CMD := unzip -o -j ffmpeg-archive "*/bin/ffmpeg.exe" -d .
endif

# Styling
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m # No Color

.PHONY: all default help clean install lint format run-gui build

default: help

help: ## Show this help message.
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# Setup
# ==============================================================================

install: ## Install dependencies using uv sync.
	@echo -e "$(GREEN)>>> Installing dependencies...$(NC)"
	@$(UV) sync --extra dev

clean: ## Remove build artifacts.
	@echo -e "$(GREEN)>>> Cleaning up...$(NC)"
	@rm -rf .venv $(DIST_DIR) $(BUILD_DIR) *.spec $(FFMPEG_BIN) ffmpeg-archive

# ==============================================================================
# Execution & Quality
# ==============================================================================

lint: ## Run Ruff linter.
	@$(UV) run ruff check src

format: ## Format code.
	@$(UV) run ruff format src
	@$(UV) run ruff check src --fix

run-gui: ## Launch the GUI.
	@$(UV) run transcriber-gui

# ==============================================================================
# Cross-Platform Building
# ==============================================================================

download-ffmpeg: ## Download the correct FFmpeg for this OS.
	@echo -e "$(YELLOW)>>> Detected OS: $(OS_NAME)$(NC)"
	@echo -e "$(YELLOW)>>> Downloading FFmpeg from: $(FFMPEG_URL)$(NC)"
	@curl -L -o ffmpeg-archive $(FFMPEG_URL)
	@echo -e "$(YELLOW)>>> Extracting...$(NC)"
	@$(EXTRACT_CMD)
	@rm ffmpeg-archive
	@echo -e "$(GREEN)>>> $(FFMPEG_BIN) is ready for bundling.$(NC)"

build: ## Build the App (Auto-detects OS).
	@if [ ! -f "$(FFMPEG_BIN)" ]; then \
		echo -e "$(RED)Error: $(FFMPEG_BIN) not found! Run 'make download-ffmpeg' first.$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(YELLOW)>>> Building $(OS_NAME) App...$(NC)"
	@$(UV) run pyinstaller --noconfirm --onefile --windowed \
		--name "AutoTranscriber" \
		--collect-all openai_whisper \
		--add-binary "$(FFMPEG_BIN)$(PYINSTALLER_SEP)." \
		src/transcriber/gui.py
	@echo -e "$(GREEN)>>> Build Complete: $(DIST_DIR)/AutoTranscriber$(EXE_EXT)$(NC)"