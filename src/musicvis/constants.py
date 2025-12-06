# --- Configuration Constants ---
DEFAULT_FPS = 30
DEFAULT_RESOLUTION = (1920, 1080)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128  # Number of frequency bars
SMOOTHING_FACTOR = 0.6  # 0.0 = no smoothing, 0.9 = heavy trails
MIN_FREQ = 20
MAX_FREQ = 8000  # Cap at 8kHz for visual relevance (most music energy is here)

# Particle system settings
MAX_PARTICLES = 500
PARTICLE_LIFETIME = 1.5  # seconds
PARTICLE_SPEED_BASE = 50  # pixels per second
PARTICLE_SIZE = 3

# Trail/ghost effect settings
TRAIL_LENGTH = 5  # Number of previous frames to keep
TRAIL_ALPHA_DECAY = 0.7  # How quickly trails fade

# Waveform settings
WAVEFORM_SAMPLES = 200  # Number of samples to display in waveform ring
