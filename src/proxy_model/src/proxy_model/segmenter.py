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
        return self.predict_with_confidence(image_bgr, target, point)[0]

    def predict_with_confidence(
        self, image_bgr: np.ndarray, target: Target, point: tuple[int, int] | None
    ) -> tuple[np.ndarray, float]:
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
        # The image predictor provides instance-level grounding confidences.  A
        # union is used for mode=all, while its mean score remains a conservative
        # scalar confidence for the quality log.
        confidence = float(np.mean([score for _, score, _ in instances]))
        return mask, confidence


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] == 0:
        return None
    return round(moments["m10"] / moments["m00"]), round(
        moments["m01"] / moments["m00"]
    )
