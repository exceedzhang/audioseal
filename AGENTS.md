# AGENTS.md - Guidelines for Agentic Coding in AudioSeal

This file provides guidelines for AI coding agents working on the AudioSeal codebase.

## Project Overview

AudioSeal is a PyTorch-based audio watermarking library for efficient localized audio watermarking. It includes:
- Generator models that embed imperceptible watermarks into audio
- Detector models that identify watermark fragments in audio
- Support for streaming audio processing

## Development Environment Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/facebookresearch/audioseal.git
cd audioseal
pip install -e ".[dev]"

# Install pre-commit hooks (run after cloning)
pre-commit install .
```

## Build/Lint/Test Commands

### Running Tests

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_models.py

# Run a single test function
pytest tests/test_models.py::test_detector

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_detector"
```

### Code Quality Tools

```bash
# Run all linters via pre-commit
pre-commit run --all-files

# Run black (code formatting)
black src/ tests/

# Run isort (import sorting)
isort src/ tests/

# Run flake8 (linting)
flake8 src/ tests/

# Run mypy (type checking)
mypy src/

# Skip specific checks in mypy (common flags)
mypy --ignore-missing-imports src/
```

### Installation/Building

```bash
# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install from PyPI
pip install audioseal
```

## Code Style Guidelines

### General Principles

- **Python version**: Support Python 3.8+ (3.10+ for streaming support)
- **Dependencies**: Keep minimal; only add when necessary. PyTorch >= 1.13.0 required.
- **License**: Include Meta copyright header (see existing files)

### Formatting

- **Line length**: Follow black's default (88 characters)
- **Indentation**: 4 spaces
- **Trailing commas**: Use where appropriate for cleaner diffs
- **Line endings**: Unix-style (LF)

### Import Organization (isort profile: black)

Order imports as:
1. Standard library
2. Third-party packages
3. Local/relative imports

```python
# Correct
import functools
import logging
import sys
from contextlib import contextmanager
from typing import Optional, Tuple

import torch

from audioseal import builder
from audioseal.loader import AudioSeal
from audioseal.models import AudioSealDetector, AudioSealWM
```

### Type Hints

- Use type hints for function arguments and return values
- Use `Optional[X]` instead of `X | None` for Python 3.8 compatibility
- Use `Tuple[X, Y]` instead of `tuple[X, Y]` for Python 3.8 compatibility
- Mypy is configured with `strict = false`; don't add strict typing

```python
# Good
def process_audio(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    ...

def detect_watermark(
    audio: torch.Tensor,
    sample_rate: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `AudioSealDetector`, `MsgProcessor`)
- **Functions/methods**: `snake_case` (e.g., `get_watermark`, `detect_watermark`)
- **Variables**: `snake_case` (e.g., `watermarked_audio`, `secret_message`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SAMPLE_RATE_WARN`)
- **Private methods**: Prefix with underscore (e.g., `_internal_method`)

### Docstrings

Use Google-style docstrings for public APIs:

```python
class AudioSealDetector(torch.nn.Module):
    """Detector for AudioSeal watermarks.

    Args:
        nbits: Number of bits in the watermark message.
        hidden_size: Hidden dimension of the model.
    """

    def __init__(self, nbits: int, hidden_size: int):
        ...
```

### Error Handling

- Use assertions for internal invariants
- Raise descriptive exceptions with clear messages
- Handle expected errors gracefully with informative logging

```python
# Good
assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"

logger = logging.getLogger("Audioseal")
logger.warning("Sample rate will be ignored in future versions")
```

### PyTorch Best Practices

- Use `torch.nn.Module` for all neural network components
- Call `super().__init__()` in subclass `__init__`
- Set `model.eval()` before inference
- Use `torch.no_grad()` for inference-only code
- Prefer in-place operations where memory is constrained

```python
model = AudioSeal.load_generator("audioseal_wm_16bits")
model.eval()

with torch.no_grad():
    watermark = model.get_watermark(audio, sample_rate=sr)
```

### Testing Guidelines

- Use `pytest` as the test framework
- Use `@pytest.fixture` for test fixtures
- Include fixtures in `conftest.py` if shared across tests
- Test both positive and negative cases
- Test edge cases (empty tensors, wrong shapes, etc.)

```python
@pytest.fixture
def example_audio(tmp_path):
    url = "https://example.com/audio.wav"
    with open(tmp_path / "test.wav", "wb") as f:
        resp = urllib.request.urlopen(url)
        f.write(resp.read())
    wav, sr = torchaudio.load(tmp_path / "test.wav")
    yield wav.expand(2, 1, -1), sr


def test_detector(example_audio):
    audio, sr = example_audio
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    ...
```

### Git/Version Control

- Create feature branches from `main`
- Use meaningful commit messages
- Run `pre-commit run --all-files` before committing
- Ensure tests pass before submitting PRs

### File Structure

```
audioseal/
├── src/audioseal/          # Main source code
│   ├── __init__.py
│   ├── builder.py          # Model building utilities
│   ├── loader.py           # Model loading from HuggingFace
│   ├── models.py           # Core model implementations
│   ├── cards/              # Model configuration YAML files
│   └── libs/               # Bundled dependencies (audiocraft, moshi)
├── tests/                  # Test suite
│   └── test_models.py
├── examples/               # Example notebooks
├── docs/                   # Documentation
└── pyproject.toml          # Project configuration
```

### Configuration Files

- `pyproject.toml`: Project metadata, dependencies, tool configs
- `.pre-commit-config.yaml`: Pre-commit hook configurations
- `pyproject.toml [tool.mypy]`: Type checking settings (Python 3.8, non-strict)

### Common Issues/Notes

1. **Audio shape**: Expects `(batch, channels, samples)` - add batch dimension if needed
2. **Sample rate**: AudioSeal does not resample internally; user must provide correct sample rate
3. **JIT scripting**: Models support `torch.jit.script` for deployment
4. **Streaming**: Use `model.streaming(batch_size=N)` context manager for streaming audio

### Key Dependencies

- `torch >= 1.13.0`
- `torchaudio == 2.1.0`
- `numpy < 2.0`
- `soundfile`
- `omegaconf`
- `einops` (Python 3.10+)

### Dev Dependencies

- `pytest`
- `black`
- `isort`
- `flake8`
- `mypy`
- `pre-commit`

## Example Usage

### Watermarking Example

The project includes an example script for adding and detecting watermarks:

**File**: `watermark_example.py`

**Usage**:
```bash
# Basic usage (detect + add watermark)
python watermark_example.py input.wav

# Specify output file and watermark message
python watermark_example.py input.wav -o output.wav -m "my message"

# Detect watermark only (no addition)
python watermark_example.py input.wav --detect-only

# View help
python watermark_example.py --help
```

**Features**:
- Auto-resample audio to 16kHz
- Auto-convert stereo to mono
- Detect if audio already has watermark (skip if already watermarked)
- Embed custom 16-bit message
- Save as WAV format
- Detect-only mode (--detect-only)

**Key Functions**:
- `load_and_resample_audio()` - Load audio and resample to 16kHz
- `detect_watermark()` - Detect watermark in audio
- `add_watermark_and_save()` - Add watermark and save to file
- `string_to_message()` - Convert string to 16-bit binary message
