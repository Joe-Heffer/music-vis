# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an audio visualization CLI tool that generates 1080p video visualizations from audio files. It creates circular frequency spectrum visualizations with bass-reactive center pulses using:
- **Librosa** for audio analysis (Mel-scaled spectrograms, RMS energy)
- **MoviePy** for video composition and audio handling
- **OpenCV** for frame rendering

The tool is designed for production use with features like temporal smoothing to prevent flickering and high-definition output.

## Development Commands

### Installation
```bash
pip install -e .
```

This installs the package in editable mode with the `audio-viz` command available globally.

### Running the Tool
```bash
# Using the installed command
audio-viz input.wav --output result.mp4

# Or directly via module
python -m musicvis input.wav --output result.mp4

# With custom settings
python -m musicvis input.wav -o output.mp4 --width 1920 --height 1080 --fps 30 --duration 60
```

### Dependencies
All dependencies are defined in `pyproject.toml`. Install manually with:
```bash
pip install numpy librosa moviepy opencv-python soundfile
```

## Architecture

### Single-Module Structure
The entire application is contained in `src/musicvis/__main__.py` as a single module. This is intentional for simplicity.

### Core Components

**AudioAnalyzer** (`src/musicvis/__main__.py:43-107`)
- Loads audio files and extracts visualization features
- Pre-calculates Mel-scaled spectrograms (better matches human hearing than linear FFT)
- Computes RMS energy for bass/volume reactive effects
- Provides time-indexed data access via `get_data_at_time(t)`
- Uses librosa's Mel spectrogram with frequency range 20-8000 Hz (most music energy)

**VisualizerRenderer** (`src/musicvis/__main__.py:109-206`)
- Stateful renderer maintaining smoothing buffers
- Implements exponential moving average for temporal smoothing (prevents flickering)
- Draws radial frequency bars in OpenCV using polar coordinates
- Creates mirrored circular spectrum (duplicates bars for full 360° display)
- Renders bass-reactive center pulse with glow effects
- Uses BGR color space (OpenCV native), converted to RGB for MoviePy

**main()** (`src/musicvis/__main__.py:208-272`)
- CLI argument parsing and validation
- Orchestrates analyzer → renderer → video export pipeline
- Handles audio syncing and duration truncation
- Uses libx264/aac codecs with medium preset for balanced speed/quality

### Key Configuration Constants
- `N_MELS = 128`: Number of frequency bars in visualization
- `SMOOTHING_FACTOR = 0.6`: Controls temporal smoothing (0.0=none, 0.9=heavy trails)
- `MIN_FREQ = 20`, `MAX_FREQ = 8000`: Frequency range for visualization
- `N_FFT = 2048`, `HOP_LENGTH = 512`: Audio analysis window settings

### Data Flow
1. Audio file → AudioAnalyzer pre-calculates entire spectrogram and RMS
2. For each video frame at time `t`:
   - `get_data_at_time(t)` retrieves spectral slice and energy
   - Renderer applies exponential smoothing to prevent jitter
   - Radial bars drawn outward from center pulse (mirrored for symmetry)
   - Frame converted BGR→RGB for MoviePy
3. MoviePy composites frames with original audio → MP4

### Visual Design
- Circular/radial spectrum bars radiating from center
- Center pulse scales with RMS energy (bass reactivity)
- Bars use non-linear height boost (`val ** 1.5`) for aesthetic "punchiness"
- Color gradient from blue (low) to orange (high) intensity
- Dark blue-grey background
