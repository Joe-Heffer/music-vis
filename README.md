# music-vis

A production-ready CLI tool for generating stunning audio visualizations from your music files.

## Features

- **Circular Frequency Spectrum**: Beautiful radial visualization that displays audio frequencies in a 360° pattern
- **Bass-Reactive Center Pulse**: Dynamic center orb that pulses with the music's energy
- **Temporal Smoothing**: Professional, flicker-free output with configurable smoothing
- **High-Definition Output**: Renders in 1080p by default (customizable resolution)
- **Perceptually Accurate**: Uses Mel-scaled spectrograms that match human hearing
- **Production Quality**: Optimized rendering with H.264 encoding

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/music-vis.git
cd music-vis

# Install in editable mode
pip install -e .
```

### Dependencies

- Python >= 3.8
- numpy
- librosa
- moviepy
- opencv-python
- soundfile

All dependencies are automatically installed when you run `pip install -e .`

## Usage

### Basic Usage

```bash
# Generate visualization from audio file
audio-viz input.wav --output result.mp4
```

### Advanced Options

```bash
# Custom resolution and framerate
audio-viz song.wav -o video.mp4 --width 3840 --height 2160 --fps 60

# Limit duration (useful for testing)
audio-viz song.wav -o preview.mp4 --duration 30

# Run via module
python -m musicvis input.wav --output result.mp4
```

### Command Line Options

```
positional arguments:
  input                 Path to input audio file (WAV/MP3)

optional arguments:
  -h, --help           Show help message
  --output, -o         Path to output video file (default: output.mp4)
  --width              Video width in pixels (default: 1920)
  --height             Video height in pixels (default: 1080)
  --fps                Frames per second (default: 30)
  --duration           Limit duration in seconds (optional)
```

## How It Works

1. **Audio Analysis**: Librosa extracts Mel-scaled frequency spectrums and RMS energy from your audio
2. **Frame Generation**: OpenCV renders each frame with radial frequency bars and reactive center pulse
3. **Temporal Smoothing**: Exponential moving average prevents flickering for smooth animations
4. **Video Export**: MoviePy composites frames with original audio and exports as MP4

### Visualization Details

- **Frequency Range**: 20 Hz - 8 kHz (captures most music energy)
- **Frequency Bars**: 128 Mel-scaled bands (mirrored for circular display)
- **Smoothing**: Configurable factor (0.6 default) balances responsiveness and stability
- **Color Gradient**: Blue (low intensity) to orange (high intensity)
- **Center Pulse**: Scales with RMS energy for bass reactivity

## Configuration

Advanced users can modify visualization parameters in `src/musicvis/__main__.py`:

```python
DEFAULT_FPS = 30
N_MELS = 128              # Number of frequency bars
SMOOTHING_FACTOR = 0.6    # 0.0 = no smoothing, 0.9 = heavy trails
MIN_FREQ = 20
MAX_FREQ = 8000
```

## Examples

```bash
# Create a 4K visualization
audio-viz track.wav -o 4k_output.mp4 --width 3840 --height 2160

# Quick preview (first 30 seconds)
audio-viz track.wav -o preview.mp4 --duration 30

# High framerate for smooth motion
audio-viz track.wav -o smooth.mp4 --fps 60
```

## Performance Notes

- Rendering time depends on audio duration and output resolution
- 1080p @ 30fps typically processes at 1-5x real-time (on modern hardware)
- Higher resolutions and framerates increase render time proportionally
- First run may be slower due to audio analysis pre-computation
