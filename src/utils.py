import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple, Union, List


def get_frames(video: str) -> Union[np.ndarray, None]:
    """Video frames iterator.

    Args:
        video (str): path to video.

    Yields:
        Iterator[Union[np.ndarray, None]]: video frame or none at the end of the video.
    """
    video = cv2.VideoCapture(video)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None


def display(
    frame: np.ndarray,
    particles: Tensor = None,
    location: Tensor = None,
    save: bool = False,
    n: int = 0,
) -> None:
    """Utility for frame representation.

    Args:
        frame (np.ndarray): video frame.
        particles (Tensor, optional): particles of filter.
        location (Tensor, optional): location of tracked object.
        save (bool, optional): save image.
        n (int, optional): frame number for saved image.
    """
    if particles is not None:
        for i in range(len(particles)):
            x = int(particles[i, 0])
            y = int(particles[i, 1])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 10)
    if location is not None:
        cv2.circle(frame, location, 10, (0, 0, 255), 35)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    if save:
        plt.savefig(f"./pics/{n}.png", bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()
