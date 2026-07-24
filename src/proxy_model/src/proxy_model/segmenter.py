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
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_mask: np.ndarray,
    smoothing_sigma_px: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a mask with the dense flow from the current frame to the previous frame."""
    flow = dense_optical_flow(previous_image, current_image, smoothing_sigma_px)
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
    return warped.astype(bool), flow


def dense_optical_flow(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    smoothing_sigma_px: float = 4.0,
) -> np.ndarray:
    """Return a spatially coherent Farneback flow from current to previous frame."""
    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=41,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    if smoothing_sigma_px > 0.0:
        flow = cv2.GaussianBlur(
            flow,
            (0, 0),
            sigmaX=smoothing_sigma_px,
            sigmaY=smoothing_sigma_px,
            borderType=cv2.BORDER_REPLICATE,
        )
    return flow


def flow_to_color(flow: np.ndarray, min_magnitude_px: float = 0.5) -> np.ndarray:
    """Encode dense flow with a soft, light-background color-wheel visualization.

    Still pixels use a light lavender background.  As motion increases, pixels
    blend into the direction color, preserving the pastel presentation used by
    modern optical-flow visualizations rather than an HSV image on a dark base.
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected flow with shape (H, W, 2), got {flow.shape}")
    magnitude = np.linalg.norm(flow, axis=2)
    angle = np.arctan2(-flow[..., 1], -flow[..., 0]) / np.pi
    wheel_position = (angle + 1.0) * 0.5 * (len(_FLOW_COLOR_WHEEL) - 1)
    lower = np.floor(wheel_position).astype(np.int32)
    upper = (lower + 1) % len(_FLOW_COLOR_WHEEL)
    fraction = (wheel_position - lower)[..., None]
    direction_color = (
        _FLOW_COLOR_WHEEL[lower] * (1.0 - fraction)
        + _FLOW_COLOR_WHEEL[upper] * fraction
    )

    visible_magnitude = np.maximum(magnitude - min_magnitude_px, 0.0)
    nonzero_magnitude = visible_magnitude[visible_magnitude > 0]
    if nonzero_magnitude.size == 0:
        motion_strength = np.zeros_like(magnitude)
    else:
        scale = max(float(np.percentile(nonzero_magnitude, 99)), 1e-6)
        motion_strength = np.clip(visible_magnitude / scale, 0.0, 1.0)
    background_rgb = np.array((250.0, 244.0, 255.0), dtype=np.float32)
    rgb = background_rgb * (1.0 - motion_strength[..., None]) + direction_color * motion_strength[..., None]
    return np.clip(rgb[..., ::-1], 0, 255).astype(np.uint8)


def _make_flow_color_wheel() -> np.ndarray:
    """Create the standard optical-flow direction wheel in RGB order."""
    segments = (
        ((255, 0, 0), (255, 255, 0), 15),
        ((255, 255, 0), (0, 255, 0), 6),
        ((0, 255, 0), (0, 255, 255), 4),
        ((0, 255, 255), (0, 0, 255), 11),
        ((0, 0, 255), (255, 0, 255), 13),
        ((255, 0, 255), (255, 0, 0), 6),
    )
    return np.concatenate(
        [
            np.linspace(start, end, count, endpoint=False, dtype=np.float32)
            for start, end, count in segments
        ]
    )


_FLOW_COLOR_WHEEL = _make_flow_color_wheel()


def postprocess_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    if dilate_px > 0:
        size = 2 * dilate_px + 1
        mask_u8 = cv2.dilate(mask_u8, np.ones((size, size), np.uint8))
    return mask_u8.astype(bool)
