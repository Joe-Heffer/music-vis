#!/usr/bin/env python3
"""
Audio Visualizer CLI Tool
=========================

A production-ready command-line tool to generate 1080p video visualizations
from WAV audio files. It utilizes Librosa for audio analysis and MoviePy/OpenCV
for rendering.

Features:
- Circular frequency spectrum visualization.
- Bass-reactive center pulse.
- Temporal smoothing for non-flickering, professional output.
- High-definition output (default 1920x1080).

Usage:
    python audio_visualizer.py input.wav --output result.mp4
    python audio_visualizer.py -h (for help)

Dependencies:
    pip install numpy librosa moviepy opencv-python soundfile
"""

import argparse
import sys
import os
import numpy as np
import librosa
import cv2
from moviepy import AudioFileClip, VideoClip
from moviepy.audio.AudioClip import CompositeAudioClip

# --- Configuration Constants ---
DEFAULT_FPS = 30
DEFAULT_Res = (1920, 1080)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128  # Number of frequency bars
SMOOTHING_FACTOR = 0.6  # 0.0 = no smoothing, 0.9 = heavy trails
MIN_FREQ = 20
MAX_FREQ = 8000  # Cap at 8kHz for visual relevance (most music energy is here)


class AudioAnalyzer:
    """
    Handles loading audio and extracting features suitable for visualization.
    """

    def __init__(self, filepath):
        print(f"[+] Loading audio: {filepath}...")
        try:
            # Load audio with original sampling rate
            self.y, self.sr = librosa.load(filepath, sr=None)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        except Exception as e:
            sys.exit(f"[!] Error loading audio file: {e}")

        # Pre-calculate features
        print("[+] Analyzing audio frequencies and dynamics...")
        self._calculate_spectrogram()
        self._calculate_rms()

    def _calculate_spectrogram(self):
        """
        Compute a Mel-scaled spectrogram.
        Mel scale matches human hearing better than linear FFT.
        """
        # Compute Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=self.y,
            sr=self.sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=MIN_FREQ,
            fmax=MAX_FREQ,
        )

        # Convert to decibels (Log scale) for better visual dynamic range
        self.S_dB = librosa.power_to_db(S, ref=np.max)

        # Normalize to 0-1 range for easy plotting
        # Typically dB ranges from -80 to 0. We clip the noise floor.
        self.S_norm = (self.S_dB + 80) / 80
        self.S_norm = np.clip(self.S_norm, 0, 1)

    def _calculate_rms(self):
        """
        Compute Root Mean Square (Energy/Volume) for the pulsing effect.
        """
        rms = librosa.feature.rms(y=self.y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        # Normalize RMS reasonably
        self.rms = librosa.util.normalize(rms[0], axis=0)

    def get_data_at_time(self, t):
        """
        Returns the spectral bars and rms energy for a specific timestamp `t`.
        """
        # Convert time to frame index
        frame_index = librosa.time_to_frames(
            t, sr=self.sr, hop_length=HOP_LENGTH, n_fft=N_FFT
        )

        # Boundary checks
        if frame_index >= self.S_norm.shape[1]:
            frame_index = self.S_norm.shape[1] - 1

        spectrogram_slice = self.S_norm[:, frame_index]
        energy = self.rms[frame_index] if frame_index < len(self.rms) else 0

        return spectrogram_slice, energy


class VisualizerRenderer:
    """
    Handles the drawing logic using OpenCV.
    Maintains state for smoothing (previous frames).
    """

    def __init__(self, analyzer, width, height):
        self.analyzer = analyzer
        self.w = width
        self.h = height
        self.center = (width // 2, height // 2)
        self.prev_spec = np.zeros(N_MELS)  # For smoothing

        # Colors (BGR format for OpenCV)
        self.bg_color = (10, 10, 15)  # Very dark blue-grey
        self.bar_color_low = np.array([200, 100, 50])  # Blueish
        self.bar_color_high = np.array([50, 100, 255])  # Orangeish

    def _interpolate_color(self, val):
        """Generate a gradient color based on bar height (0.0 to 1.0)."""
        # Simple linear interpolation between two colors
        color = self.bar_color_low * (1 - val) + self.bar_color_high * val
        return tuple(map(int, color))

    def make_frame(self, t):
        """
        The callback function for MoviePy.
        Generates a single video frame at time t.
        """
        # 1. Get Data
        raw_spec, energy = self.analyzer.get_data_at_time(t)

        # 2. Apply Smoothing (Exponential Moving Average)
        # current = alpha * new + (1 - alpha) * old
        smooth_spec = (
            1 - SMOOTHING_FACTOR
        ) * raw_spec + SMOOTHING_FACTOR * self.prev_spec
        self.prev_spec = smooth_spec

        # 3. Setup Canvas
        frame = np.full((self.h, self.w, 3), self.bg_color, dtype=np.uint8)

        # 4. Calculate Geometry
        # Base radius of the center circle + pulse based on volume energy
        base_radius = int(min(self.w, self.h) * 0.15)
        pulse_radius = int(base_radius + (energy * 40))

        max_bar_height = int(min(self.w, self.h) * 0.30)

        # 5. Draw Spectrum (Radial)
        # We duplicate the spectrum to make it a full circle (mirror left/right)
        num_bars = len(smooth_spec)
        angle_step = 360 / (num_bars * 2)  # *2 because we mirror

        # Draw two halves: Right side (0 to 180) and Left side (360 down to 180)
        for i in range(num_bars):
            val = smooth_spec[i]

            # Non-linear height boost for aesthetic "punchiness"
            display_val = val**1.5
            bar_len = int(display_val * max_bar_height)

            if bar_len < 2:
                continue  # Skip tiny bars

            color = self._interpolate_color(display_val)

            # Calculate start and end points for radial lines
            # We draw lines radiating OUT from the pulse circle

            # Angle for right side
            angle_r = np.deg2rad(i * angle_step - 90)  # Start from top (-90)
            # Angle for left side (mirrored)
            angle_l = np.deg2rad(-i * angle_step - 90)

            # Define simple helper for polar to cartesian
            def pol2cart(rho, phi):
                x = int(rho * np.cos(phi) + self.center[0])
                y = int(rho * np.sin(phi) + self.center[1])
                return (x, y)

            # Right bar
            start_r = pol2cart(pulse_radius + 5, angle_r)
            end_r = pol2cart(pulse_radius + 5 + bar_len, angle_r)
            cv2.line(frame, start_r, end_r, color, 4, cv2.LINE_AA)

            # Left bar
            start_l = pol2cart(pulse_radius + 5, angle_l)
            end_l = pol2cart(pulse_radius + 5 + bar_len, angle_l)
            cv2.line(frame, start_l, end_l, color, 4, cv2.LINE_AA)

        # 6. Draw Center Pulse Orb
        # Glow effect (layered circles)
        cv2.circle(
            frame, self.center, pulse_radius + 10, (40, 40, 50), -1, cv2.LINE_AA
        )  # Outer glow
        cv2.circle(
            frame, self.center, pulse_radius, (200, 200, 200), 2, cv2.LINE_AA
        )  # Rim

        # Inner text/logo area (optional, just a solid color for now)
        # Make color react to bass (white to red)
        center_color = (int(50 + energy * 100), 20, 20)
        cv2.circle(frame, self.center, pulse_radius - 2, center_color, -1, cv2.LINE_AA)

        return frame


def main():
    parser = argparse.ArgumentParser(
        description="Generate a music visualization video from an audio file."
    )
    parser.add_argument("input", help="Path to input audio file (WAV/MP3)")
    parser.add_argument(
        "--output", "-o", default="output.mp4", help="Path to output video file"
    )
    parser.add_argument("--width", type=int, default=1920, help="Video width")
    parser.add_argument("--height", type=int, default=1080, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--duration", type=int, help="Limit duration in seconds (optional)"
    )

    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.input):
        sys.exit(f"[!] Input file not found: {args.input}")

    # 2. Analyze Audio
    analyzer = AudioAnalyzer(args.input)

    # 3. Setup Video Generation
    duration = analyzer.duration
    if args.duration and args.duration < duration:
        duration = args.duration
        print(f"[i] Truncating duration to {duration} seconds.")

    print(f"[+] Preparing render: {args.width}x{args.height} @ {args.fps}fps")
    print(f"[+] Duration: {duration:.2f} seconds")

    renderer = VisualizerRenderer(analyzer, args.width, args.height)

    # 4. Create MoviePy Clip
    # make_frame(t) requires RGB, OpenCV uses BGR usually, but MoviePy expects RGB.
    # We generated BGR in renderer for standard CV2 habits, let's flip it in the lambda or render RGB.
    # Actually, let's just render RGB in the renderer to save a flip operation.
    # (Modified renderer colors to be RGB-ish or just flip here).
    # We will flip here for safety.

    def make_frame_wrapper(t):
        frame = renderer.make_frame(t)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    video_clip = VideoClip(make_frame_wrapper, duration=duration)

    # Attach original audio
    audio_clip = AudioFileClip(args.input)
    # Ensure audio is cut if we truncated duration
    audio_clip = audio_clip.subclipped(0, duration)
    video_clip = video_clip.with_audio(audio_clip)

    # 5. Export
    print("[+] Rendering video... (This may take a while)")
    video_clip.write_videofile(
        args.output,
        fps=args.fps,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",  # Balance between speed and compression
        logger="bar",
    )

    print(f"\n[+] Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
