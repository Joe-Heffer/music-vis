#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import cv2
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import VideoClip

from musicvis.audio_analyser import AudioAnalyser
from musicvis.visualiser_renderer import VisualiserRenderer

DESCRIPTION = """
Audio Visualiser CLI Tool
=========================

A production-ready command-line tool to generate 1080p video visualisations
from WAV audio files. It utilizes Librosa for audio analysis and MoviePy/OpenCV
for rendering.

Features:
- Circular frequency spectrum visualisation.
- Bass-reactive center pulse.
- Temporal smoothing for non-flickering, professional output.
- High-definition output (default 1920x1080).
"""

logger = logging.getLogger(__name__)


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Command line arguments
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input", help="Path to input audio file (WAV/MP3)")
    parser.add_argument("--output", "-o", default="output.mp4", help="Path to output video file")
    parser.add_argument("--width", type=int, default=1920, help="Video width")
    parser.add_argument("--height", type=int, default=1080, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=int, help="Limit duration in seconds (optional)")
    args = parser.parse_args()

    # 1. Validation
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[!] Input file not found: {args.input}")

    # 2. Analyze Audio
    analyzer = AudioAnalyser(args.input)

    # 3. Setup Video Generation
    duration = analyzer.duration
    if args.duration and args.duration < duration:
        duration = args.duration
        logger.info(f"[i] Truncating duration to {duration} seconds.")

    logger.info(f"[+] Preparing render: {args.width}x{args.height} @ {args.fps}fps")
    logger.info(f"[+] Duration: {duration:.2f} seconds")

    renderer = VisualiserRenderer(analyzer, args.width, args.height, args.fps)

    # 4. Create MoviePy Clip
    # make_frame(t) requires RGB, OpenCV uses BGR usually, but MoviePy expects RGB.
    # We generated BGR in renderer for standard CV2 habits, let's flip it in the lambda
    # or render RGB.
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
    audio_clip = audio_clip.subclip(0, duration)
    video_clip = video_clip.set_audio(audio_clip)

    # 5. Export
    logger.info("[+] Rendering video... (This may take a while)")
    video_clip.write_videofile(
        args.output,
        fps=args.fps,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",  # Balance between speed and compression
        logger="bar",
    )

    logger.info(f"\n[+] Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
