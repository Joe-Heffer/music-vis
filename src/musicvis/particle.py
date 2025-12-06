from musicvis.constants import PARTICLE_LIFETIME


class Particle:
    """Represents a single particle emitted from spectrum bars."""

    def __init__(self, x, y, vx, vy, color, birth_time):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.birth_time = birth_time

    def update(self, dt):
        """Update particle position."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def is_alive(self, current_time):
        """Check if particle should still be rendered."""
        return (current_time - self.birth_time) < PARTICLE_LIFETIME

    def get_alpha(self, current_time):
        """Get particle opacity based on lifetime."""
        age = current_time - self.birth_time
        return max(0, 1 - (age / PARTICLE_LIFETIME))
