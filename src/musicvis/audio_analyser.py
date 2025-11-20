import logging
import sys

import librosa
import numpy as np

from musicvis.constants import (
    HOP_LENGTH,
    MAX_FREQ,
    MIN_FREQ,
    N_FFT,
    N_MELS,
    WAVEFORM_SAMPLES,
)

logger = logging.getLogger(__name__)


class AudioAnalyser:
    """
    Handles loading audio and extracting features suitable for visualisation.
    """

    def __init__(self, filepath):
        logger.info(f"[+] Loading audio: {filepath}...")
        try:
            # Load audio with original sampling rate
            self.y, self.sr = librosa.load(filepath, sr=None)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        except Exception as e:
            sys.exit(f"[!] Error loading audio file: {e}")

        # Pre-calculate features
        logger.info("[+] Analyzing audio frequencies and dynamics...")
        self._calculate_spectrogram()
        self._calculate_rms()

    def _calculate_spectrogram(self):
        """
        Compute a Mel-scaled spectrogram.
        Mel scale matches human hearing better than linear FFT.
        """
        # Compute Mel spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=self.y,
            sr=self.sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=MIN_FREQ,
            fmax=MAX_FREQ,
        )

        # Convert to decibels (Log scale) for better visual dynamic range
        self.S_dB = librosa.power_to_db(spectrogram, ref=np.max)

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
        frame_index = librosa.time_to_frames(t, sr=self.sr, hop_length=HOP_LENGTH, n_fft=N_FFT)

        # Boundary checks
        if frame_index >= self.S_norm.shape[1]:
            frame_index = self.S_norm.shape[1] - 1

        spectrogram_slice = self.S_norm[:, frame_index]
        energy = self.rms[frame_index] if frame_index < len(self.rms) else 0

        return spectrogram_slice, energy

    def get_waveform_at_time(self, t):
        """
        Returns a slice of the waveform around time `t` for visualisation.
        """
        # Calculate sample index
        sample_index = int(t * self.sr)

        # Get window of samples around this point
        half_window = WAVEFORM_SAMPLES // 2
        start = max(0, sample_index - half_window)
        end = min(len(self.y), sample_index + half_window)

        # Extract and normalize waveform slice
        waveform_slice = self.y[start:end]

        # Pad if needed
        if len(waveform_slice) < WAVEFORM_SAMPLES:
            waveform_slice = np.pad(waveform_slice, (0, WAVEFORM_SAMPLES - len(waveform_slice)))

        return waveform_slice[:WAVEFORM_SAMPLES]
