import cv2
import torch
import numpy as np
from torch import Tensor
from utils import display, get_frames
from typing import Tuple, Union, List
from IPython.display import clear_output


def get_width_and_height(video: str) -> Tuple[int, int]:
    """Utility for frame size.

    Args:
        video (str): path to video.

    Returns:
        Tuple[int, int]: video width and height.
    """
    video = cv2.VideoCapture(video)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            return frame.shape[1], frame.shape[0]


def initialize_paticles(
    n_paticles: int, width: int, heigh: int, vel_range: float
) -> Tensor:
    """Initializes random particles for filter.

    Args:
        n_paticles (int): number of particles.
        width (int): frame width.
        heigh (int): frame hight.
        vel_range (float): max particles velocity.

    Returns:
        Tensor: random particles with [x, y, vx, vy].
    """
    particles = torch.rand(n_paticles, 4)
    particles *= torch.tensor([width, heigh, vel_range, vel_range])
    particles[:, 2:] -= vel_range / 2
    return particles


def apply_velocity(particles: Tensor) -> Tensor:
    """Update particles positions w.r.t. particles velocitys.

    Args:
        particles (Tensor): filter particles.

    Returns:
        Tensor: updated filter particles.
    """
    particles[:, :2] += particles[:, 2:]
    return particles


def restrict_edges(particles: Tensor, width: int, height: int) -> Tensor:
    """Force particles to stay within image boundaries.

    Args:
        particles (Tensor): filter particles.
        width (int): frame width.
        height (int): frame hight.

    Returns:
        Tensor: filter particles.
    """

    particles[:, 0] = torch.max(
        torch.tensor(0), torch.min(torch.tensor(width - 1), particles[:, 0])
    )
    particles[:, 1] = torch.max(
        torch.tensor(0), torch.min(torch.tensor(height - 1), particles[:, 1])
    )
    return particles


def particles_to_coordinates(particles: Tensor) -> Tensor:
    """Get image pixel coordinates from paticles.

    Args:
        particles (Tensor): filter particles.

    Returns:
        Tensor: pixel coordinates.
    """
    return particles[:, :2].type(torch.long)


def get_errors(
    particles_coordinates: Tensor, frame: np.ndarray, target_color_bgr: Tensor
) -> Tensor:
    """Calculate errors for particles.

    Args:
        particles_coordinates (Tensor): image pixels coordinates.
        frame (np.ndarray): video frame.
        target_color_bgr (Tensor): tracked object color.

    Returns:
        Tensor: errors for particles.
    """
    particles_values = frame[
        particles_coordinates[:, 1], particles_coordinates[:, 0], :
    ]
    errors = (target_color_bgr - particles_values) ** 2
    errors = errors.squeeze(0).sum(dim=1)
    return errors


def get_weights(
    errors: Tensor, particles_coordinates: Tensor, width: int, height: int, pow: int = 4
) -> Tensor:
    """Calculate particles weights w.r.t. their errors.

    Args:
        errors (Tensor): particles errors.
        particles_coordinates (Tensor): pixel coordinates of particles.
        width (int): frame width.
        height (int): frame height.
        pow (int, optional): power that weights are rised.

    Returns:
        Tensor: particles weights.
    """
    weights = (errors.max() - errors).type(torch.float64)
    boundary_mask = (
        (particles_coordinates[:, 0] == 0)
        | (particles_coordinates[:, 0] == width - 1)
        | (particles_coordinates[:, 1] == 0)
        | (particles_coordinates[:, 1] == height - 1)
    )
    weights[boundary_mask] = 0
    weights = weights ** pow
    return weights


def resample(particles: Tensor, weights: Tensor, n_particles: int) -> Tensor:
    """Resample particles w.r.t. their weights.

    Args:
        particles (Tensor): filter particles.
        weights (Tensor): paticles weights.
        n_particles (int): number of particles.

    Returns:
        Tensor: resampled particles.
    """
    probs = weights / weights.sum()
    particles_idx = probs.multinomial(n_particles, True)
    particles = particles[particles_idx]
    return particles


def get_object_location(particles: Tensor) -> Tuple[int, int]:
    """Tracked object location from particles.

    Args:
        particles (Tensor): filter particles.

    Returns:
        Tuple[int, int]: [x, y] object pixel coordinates.
    """
    loc = particles[:, :2].mean(0)
    return int(loc[0]), int(loc[1])


def apply_noise(
    particles: Tensor, pos_sigm: float, vel_sigm: float, num_particles: int
) -> Tensor:
    """Add noise to filter particles.

    Args:
        particles (Tensor): filter particles.
        pos_sigm (float): sigma for position normal distribution.
        vel_sigm (float): sigma for velocity normal distribution.
        num_particles (int): number of particles.

    Returns:
        Tensor: filter particles with noise.
    """
    noise = torch.cat(
        [
            torch.normal(0, pos_sigm, (num_particles, 2)),
            torch.normal(0, vel_sigm, (num_particles, 2)),
        ],
        axis=1,
    )
    particles += noise
    return particles


class ParticleFilterTracker:
    def __init__(
        self,
        video_path: str,
        n_particles: int,
        tgt_color_rgb: List,
        vel_range: int = 10,
        pos_sigm: int = 10,
        vel_sigm: int = 5,
        display: bool = False,
        save_frames: bool = False,
    ) -> None:
        self.n_particles = n_particles
        self.vel_range = vel_range
        self.pos_sigm = pos_sigm
        self.vel_sigm = vel_sigm
        self.display = display
        self.save_frames = save_frames
        self.width, self.height = get_width_and_height(video_path)
        self.tgt_color = self.rgb_to_bgr(tgt_color_rgb)
        self.particles = initialize_paticles(
            n_particles, self.width, self.height, vel_range
        )

    def rgb_to_bgr(self, color: List) -> Tensor:
        color[0], color[2] = color[2], color[0]
        return torch.tensor(color).view(1, 1, -1)

    def handle_frame(self, frame: np.ndarray) -> Tensor:
        self.particles = apply_velocity(self.particles)
        self.particles = restrict_edges(self.particles, self.width, self.height)
        particles_coord = particles_to_coordinates(self.particles)
        errors = get_errors(particles_coord, frame, self.tgt_color)
        weights = get_weights(errors, particles_coord, self.width, self.height)
        self.particles = resample(self.particles, weights, self.n_particles)
        location = get_object_location(self.particles)
        self.particles = apply_noise(
            self.particles, self.pos_sigm, self.vel_sigm, self.n_particles
        )
        return location

    def handle_video(self, video_path: str, first_frame: int = 0, n_frames: int = 5):
        n = 0
        for frame in get_frames(video_path):
            if frame is not None and n < first_frame + n_frames:
                if n >= first_frame:
                    location = self.handle_frame(frame)
                    display(
                        frame,
                        self.particles,
                        location,
                        self.save_frames,
                        n - first_frame,
                    )
                    clear_output()
            else:
                break
            n += 1
