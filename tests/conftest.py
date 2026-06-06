"""
Test configuration: mock unresolvable CUDA-only wheels and provide shared fixtures.

silero-vad depends on torchaudio which may ship a GPU-built extension that
cannot load on CPU-only machines. Replacing both modules with MagicMocks
lets the VAD unit tests run purely in Python with no native extensions.
"""
import sys
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

for _mod in ("torchaudio", "silero_vad", "silero_vad.model", "silero_vad.utils_vad"):
    sys.modules.setdefault(_mod, MagicMock())


@pytest.fixture(scope="session")
def runner():
    return CliRunner()


@pytest.fixture(scope="session")
def cli_app():
    from opentranscriber.cli import main

    return main
