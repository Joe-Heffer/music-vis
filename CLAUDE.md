# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an audio visualisation CLI tool that generates 1080p video visualisations from audio files. It creates circular frequency spectrum visualisations with bass-reactive center pulses using:

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

### Module Structure

The application is organized into separate modules for clarity and maintainability:

- `src/musicvis/__main__.py` - CLI entry point and orchestration
- `src/musicvis/audio_analyser.py` - Audio processing and feature extraction
- `src/musicvis/visualiser_renderer.py` - Frame rendering logic
- `src/musicvis/particle.py` - Particle system for visual effects
- `src/musicvis/constants.py` - Configuration constants

### Core Components

**AudioAnalyser** (`src/musicvis/audio_analyser.py:19-106`)

- Loads audio files and extracts visualisation features
- Pre-calculates Mel-scaled spectrograms (better matches human hearing than linear FFT)
- Computes RMS energy for bass/volume reactive effects
- Provides time-indexed data access via `get_data_at_time(t)` and `get_waveform_at_time(t)`
- Uses librosa's Mel spectrogram with frequency range 20-8000 Hz (most music energy)
- Extracts waveform slices for oscilloscope visualization

**VisualiserRenderer** (`src/musicvis/visualiser_renderer.py:16-219`)

- Stateful renderer maintaining smoothing buffers and particle system
- Implements exponential moving average for temporal smoothing (prevents flickering)
- Draws radial frequency bars in OpenCV using polar coordinates
- Creates mirrored circular spectrum (duplicates bars for full 360° display)
- Renders bass-reactive center pulse with multi-layered glow effects
- Draws oscilloscope waveform ring inside center circle
- Emits particles from high-energy spectrum bars
- Implements trail/ghost effect by blending previous frames
- Uses BGR color space (OpenCV native), converted to RGB for MoviePy

**Particle** (`src/musicvis/particle.py:4-28`)

- Represents individual particles emitted from spectrum bars
- Physics-based movement with velocity and lifetime
- Fades out over configurable lifetime (1.5 seconds default)
- Used to create dynamic visual effects reacting to high-energy frequencies

**main()** (`src/musicvis/__main__.py:32-101`)

- CLI argument parsing and validation
- Orchestrates analyzer → renderer → video export pipeline
- Handles audio syncing and duration truncation
- Uses libx264/aac codecs with medium preset for balanced speed/quality

### Key Configuration Constants

All constants are centralized in `src/musicvis/constants.py`:

**Audio Analysis:**
- `N_MELS = 128`: Number of frequency bars in visualisation
- `N_FFT = 2048`, `HOP_LENGTH = 512`: Audio analysis window settings
- `MIN_FREQ = 20`, `MAX_FREQ = 8000`: Frequency range for visualisation
- `WAVEFORM_SAMPLES = 200`: Number of samples in oscilloscope ring

**Visual Effects:**
- `SMOOTHING_FACTOR = 0.6`: Controls temporal smoothing (0.0=none, 0.9=heavy trails)
- `MAX_PARTICLES = 500`: Maximum concurrent particles
- `PARTICLE_LIFETIME = 1.5`: Particle duration in seconds
- `PARTICLE_SPEED_BASE = 50`: Base particle velocity in pixels/second
- `TRAIL_LENGTH = 5`: Number of previous frames for ghost effect
- `TRAIL_ALPHA_DECAY = 0.7`: Trail fade rate

### Data Flow

1. Audio file → AudioAnalyser pre-calculates entire spectrogram, RMS, and waveform data
2. For each video frame at time `t`:
   - `get_data_at_time(t)` retrieves spectral slice and energy
   - `get_waveform_at_time(t)` retrieves waveform samples for oscilloscope
   - Renderer applies exponential smoothing to prevent jitter
   - Trail frames blended with fade-out effect
   - Radial bars drawn outward from center pulse (mirrored for symmetry)
   - High-energy bars emit particles with physics simulation
   - Center orb with multi-layer glow and reactive rim
   - Oscilloscope waveform ring drawn inside center circle
   - Frame converted BGR→RGB for MoviePy
3. MoviePy composites frames with original audio → MP4

### Visual Design

- **Circular spectrum**: Radial frequency bars radiating from center with mirrored symmetry
- **Center orb**: Pulsing circle that scales with RMS energy (bass reactivity)
  - Multi-layered glow effect for depth
  - Color-reactive rim that brightens with music intensity
- **Oscilloscope ring**: Live waveform visualization inside the center orb
- **Particle system**: Particles emitted from high-energy bars (>0.6 amplitude)
  - Physics-based movement radiating outward
  - Fade out over 1.5 second lifetime
  - Maximum 500 concurrent particles
- **Trail effect**: Ghost images of previous spectrum frames with exponential fade
- **Color scheme**:
  - Multi-stop gradient (deep blue → cyan/pink → orange → yellow)
  - Colors vary by both amplitude and angular position
  - Bars use non-linear height boost (`val ** 1.5`) for aesthetic "punchiness"
- **Background**: Very dark (near-black) for maximum contrast
