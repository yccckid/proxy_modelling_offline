from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import Segmentation
from .qwen import Target


class SamSegmenter:
    def __init__(self, config: Segmentation):
        root = str(config.sam_agent_root)
        sam_source = str(config.sam_agent_root / "sam3_src")
        for path in (sam_source, root):
            if path not in sys.path:
                sys.path.insert(0, path)
        if not config.checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {config.checkpoint}")

        from sam3_segmenter import SAM3Segmenter

        self._impl = SAM3Segmenter(
            checkpoint=str(config.checkpoint),
            device=config.device,
            confidence_threshold=config.confidence_threshold,
        )
        self._minimum_area = config.min_mask_area_px

    def predict(
        self, image_bgr: np.ndarray, target: Target, point: tuple[int, int] | None
    ) -> np.ndarray:
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        instances = self._impl.predict(
            image=image,
            text=target.label,
            mode=target.mode,
            center_point=point,
        )
        mask = np.zeros(image_bgr.shape[:2], dtype=bool)
        for instance_mask, _, _ in instances:
            mask |= instance_mask
        if int(mask.sum()) < self._minimum_area:
            raise RuntimeError(
                f"SAM mask is too small ({int(mask.sum())} < {self._minimum_area})"
            )
        return mask


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] == 0:
        return None
    return round(moments["m10"] / moments["m00"]), round(
        moments["m01"] / moments["m00"]
    )


def warp_mask(
    previous_image: np.ndarray, current_image: np.ndarray, previous_mask: np.ndarray
) -> np.ndarray:
    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    h, w = previous_mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(
        previous_mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped.astype(bool)


def postprocess_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    if dilate_px > 0:
        size = 2 * dilate_px + 1
        mask_u8 = cv2.dilate(mask_u8, np.ones((size, size), np.uint8))
    return mask_u8.astype(bool)
