import cv2
import numpy as np

from musicvis.constants import (
    MAX_PARTICLES,
    N_MELS,
    PARTICLE_SIZE,
    PARTICLE_SPEED_BASE,
    SMOOTHING_FACTOR,
    TRAIL_ALPHA_DECAY,
    TRAIL_LENGTH,
)
from musicvis.particle import Particle


class VisualiserRenderer:
    """
    Handles the drawing logic using OpenCV.
    Maintains state for smoothing (previous frames).
    """

    def __init__(self, analyzer, width, height, fps):
        self.analyzer = analyzer
        self.w = width
        self.h = height
        self.fps = fps
        self.center = (width // 2, height // 2)
        self.prev_spec = np.zeros(N_MELS)  # For smoothing

        # Particle system
        self.particles = []
        self.last_time = 0

        # Trail effect - store previous frames
        self.trail_frames = []

        # Colors (BGR format for OpenCV)
        self.bg_color = (5, 5, 10)  # Darker background for more contrast

        # Multi-color gradient for bars (more vibrant)
        self.color_stops = [
            np.array([180, 60, 20]),  # Deep blue
            np.array([255, 100, 180]),  # Cyan/pink
            np.array([100, 200, 255]),  # Orange
            np.array([50, 255, 255]),  # Yellow
        ]

    def _interpolate_color(self, val, angle=0):
        """Generate a gradient color based on bar height and position."""
        # Multi-stop gradient
        num_stops = len(self.color_stops)

        # Mix value with angle for color variation around the circle
        color_position = (val * 0.7 + (angle / 360) * 0.3) * (num_stops - 1)

        # Find which two color stops to interpolate between
        idx = int(color_position)
        if idx >= num_stops - 1:
            return tuple(map(int, self.color_stops[-1]))

        # Interpolate between adjacent stops
        t = color_position - idx
        color = self.color_stops[idx] * (1 - t) + self.color_stops[idx + 1] * t
        return tuple(map(int, color))

    def make_frame(self, t):
        """
        The callback function for MoviePy.
        Generates a single video frame at time t.
        """
        dt = t - self.last_time if self.last_time > 0 else 1 / self.fps
        self.last_time = t

        # 1. Get Data
        raw_spec, energy = self.analyzer.get_data_at_time(t)
        waveform = self.analyzer.get_waveform_at_time(t)

        # 2. Apply Smoothing (Exponential Moving Average)
        smooth_spec = (1 - SMOOTHING_FACTOR) * raw_spec + SMOOTHING_FACTOR * self.prev_spec
        self.prev_spec = smooth_spec

        # 3. Setup Canvas with trail effect
        frame = np.full((self.h, self.w, 3), self.bg_color, dtype=np.uint8)

        # Draw trail frames with fading
        if len(self.trail_frames) > 0:
            for i, trail_frame in enumerate(self.trail_frames):
                alpha = TRAIL_ALPHA_DECAY ** (len(self.trail_frames) - i)
                frame = cv2.addWeighted(frame, 1.0, trail_frame, alpha * 0.3, 0)

        # 4. Calculate Geometry
        base_radius = int(min(self.w, self.h) * 0.15)
        pulse_radius = int(base_radius + (energy * 40))
        max_bar_height = int(min(self.w, self.h) * 0.30)

        # Define polar to cartesian helper
        def pol2cart(rho, phi):
            x = int(rho * np.cos(phi) + self.center[0])
            y = int(rho * np.sin(phi) + self.center[1])
            return (x, y)

        # 5. Draw Spectrum (Radial)
        num_bars = len(smooth_spec)
        angle_step = 360 / (num_bars * 2)

        spectrum_layer = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        for i in range(num_bars):
            val = smooth_spec[i]
            display_val = val**1.5
            bar_len = int(display_val * max_bar_height)

            if bar_len < 2:
                continue

            # Right side
            angle_r_deg = i * angle_step - 90
            angle_r = np.deg2rad(angle_r_deg)
            color_r = self._interpolate_color(display_val, angle_r_deg % 360)

            start_r = pol2cart(pulse_radius + 5, angle_r)
            end_r = pol2cart(pulse_radius + 5 + bar_len, angle_r)
            cv2.line(spectrum_layer, start_r, end_r, color_r, 4, cv2.LINE_AA)

            # Emit particles from high-energy bars
            if (
                display_val > 0.6
                and len(self.particles) < MAX_PARTICLES
                and np.random.random() > 0.7
            ):
                # Emit from tip of bar
                vx = np.cos(angle_r) * PARTICLE_SPEED_BASE * (display_val + 0.5)
                vy = np.sin(angle_r) * PARTICLE_SPEED_BASE * (display_val + 0.5)
                self.particles.append(Particle(end_r[0], end_r[1], vx, vy, color_r, t))

            # Left side (mirrored)
            angle_l_deg = -i * angle_step - 90
            angle_l = np.deg2rad(angle_l_deg)
            color_l = self._interpolate_color(display_val, angle_l_deg % 360)

            start_l = pol2cart(pulse_radius + 5, angle_l)
            end_l = pol2cart(pulse_radius + 5 + bar_len, angle_l)
            cv2.line(spectrum_layer, start_l, end_l, color_l, 4, cv2.LINE_AA)

            # Emit particles from left side too
            if (
                display_val > 0.6
                and len(self.particles) < MAX_PARTICLES
                and np.random.random() > 0.7
            ):
                vx = np.cos(angle_l) * PARTICLE_SPEED_BASE * (display_val + 0.5)
                vy = np.sin(angle_l) * PARTICLE_SPEED_BASE * (display_val + 0.5)
                self.particles.append(Particle(end_l[0], end_l[1], vx, vy, color_l, t))

        frame = cv2.add(frame, spectrum_layer)

        # 6. Update and draw particles
        self.particles = [p for p in self.particles if p.is_alive(t)]
        for particle in self.particles:
            particle.update(dt)
            alpha = particle.get_alpha(t)
            if 0 <= particle.x < self.w and 0 <= particle.y < self.h:
                color = tuple(int(c * alpha) for c in particle.color)
                cv2.circle(
                    frame,
                    (int(particle.x), int(particle.y)),
                    PARTICLE_SIZE,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

        # 7. Draw Center Pulse Orb
        # Multi-layered glow
        glow_colors = [(30, 20, 40), (50, 30, 60), (70, 40, 80)]
        for i, glow_color in enumerate(glow_colors):
            glow_radius = pulse_radius + 15 - i * 5
            cv2.circle(frame, self.center, glow_radius, glow_color, -1, cv2.LINE_AA)

        # Rim with color reactivity
        rim_color = (
            int(150 + energy * 100),
            int(150 + energy * 80),
            int(200 + energy * 55),
        )
        cv2.circle(frame, self.center, pulse_radius, rim_color, 3, cv2.LINE_AA)

        # Inner fill - colorful and reactive
        center_r = int(100 + energy * 150)
        center_g = int(50 + energy * 100)
        center_b = int(150 + energy * 100)
        center_color = (center_b, center_g, center_r)
        cv2.circle(frame, self.center, pulse_radius - 3, center_color, -1, cv2.LINE_AA)

        # 8. Draw inner waveform ring (on top of center circle)
        waveform_radius = int(pulse_radius * 0.65)
        waveform_amplitude = int(pulse_radius * 0.45)

        points = []
        for i, sample in enumerate(waveform):
            angle = (i / len(waveform)) * 2 * np.pi
            # Add waveform amplitude to radius
            radius = waveform_radius + int(sample * waveform_amplitude)
            x = int(radius * np.cos(angle - np.pi / 2) + self.center[0])
            y = int(radius * np.sin(angle - np.pi / 2) + self.center[1])
            points.append([x, y])

        if len(points) > 2:
            points = np.array(points, dtype=np.int32)
            # Draw waveform with bright, visible colors
            cv2.polylines(frame, [points], True, (255, 180, 100), 3, cv2.LINE_AA)

        # Store current frame for trail effect
        self.trail_frames.append(spectrum_layer.copy())
        if len(self.trail_frames) > TRAIL_LENGTH:
            self.trail_frames.pop(0)

        return frame
