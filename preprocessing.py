import cv2
import numpy as np

from typing import Union
from pathlib import Path


def write_image(image_path: Union[str, Path], image: np.ndarray):
    """Writes an RGB image using OpenCV."""
    if isinstance(image_path, Path):
        image_path = str(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)


def read_image(image_path: Union[str, Path]) -> np.ndarray:
    """Reads an image from a path and converts it to RGB format."""
    if isinstance(image_path, Path):
        image_path = str(image_path)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image.astype(np.float32)


def read_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """Reads a mask from a path and transform it to binary."""
    if isinstance(mask_path, Path):
        mask_path = str(mask_path)
    mask = cv2.imread(mask_path, 0)
    if mask.max() == 255:
        mask = mask / 255.0
    assert_mask_is_binary(mask)
    return mask.astype(np.float32)


def assert_mask_is_binary(mask: np.ndarray):
    """Counts all the pixels different to zero and one to check if binary."""
    assert (
        np.count_nonzero((mask != 0) & (mask != 1)) == 0
    ), f"Mask is not binary. Unique values: {np.unique(mask)}"
