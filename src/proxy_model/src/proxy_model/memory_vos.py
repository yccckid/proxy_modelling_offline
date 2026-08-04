"""SAM3 native-memory video segmentation and image-only quality utilities.

The module deliberately does not use lidar, poses or optical flow.  SAM3's
video predictor owns the working memory; this adapter keeps only verified
permanent anchors and exposes stable numpy probability maps to the pipeline.
"""
from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Segmentation
from .qwen import Target


@dataclass(frozen=True)
class Anchor:
    frame_id: int
    mask: np.ndarray
    quality: float
    appearance: np.ndarray
    aspect_ratio: float
    source: str


@dataclass(frozen=True)
class Prediction:
    probability: np.ndarray
    confidence: float
    elapsed_s: float


@dataclass(frozen=True)
class MaskQuality:
    score: float
    mean_probability: float
    component_ratio: float
    hole_ratio: float
    edge_ratio: float
    image_quality: float
    sharpness: float
    exposure: float


class Sam3MemoryVOS:
    """Thin adapter around SAM3/SAM3.1's native video memory implementation."""

    def __init__(self, options: Segmentation, frames_dir: Path, target: Target):
        root = str(options.sam_agent_root)
        source = str(options.sam_agent_root / "sam3_src")
        for path in (source, root):
            if path not in sys.path:
                sys.path.insert(0, path)
        if not options.video_checkpoint and not options.checkpoint.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {options.checkpoint}")
        if options.device != "cuda":
            raise ValueError("SAM3 video memory inference currently requires segmentation.device: cuda")

        from sam3.model_builder import build_sam3_predictor

        checkpoint = options.video_checkpoint or options.checkpoint
        self._predictor = build_sam3_predictor(
            checkpoint_path=str(checkpoint),
            version=options.video_model_version,
            compile=options.video_compile,
            async_loading_frames=True,
        )
        self._frames_dir = frames_dir
        self._target = target
        self._mask_threshold = options.mask_threshold
        self.frame_times_s: list[float] = []
        tracker = getattr(self._predictor.model, "tracker", None)
        native_capacity = int(getattr(tracker, "num_maskmem", options.max_working_memory))
        # SAM3 allocates positional embeddings at construction, so this adapter
        # may safely reduce (but never inflate) the native memory-bank capacity.
        self.working_memory_capacity = min(options.max_working_memory, native_capacity)
        if tracker is not None:
            tracker.num_maskmem = self.working_memory_capacity
        self.peak_working_memory = self.working_memory_capacity

    def propagate(
        self,
        anchor: Anchor,
        direction: str,
        frame_count: int,
    ) -> dict[int, Prediction]:
        """Run one independent directional pass from a verified anchor."""
        if direction not in {"forward", "backward"}:
            raise ValueError(f"Unknown direction: {direction}")
        h, w = anchor.mask.shape
        x, y, bw, bh = _box_xywh(anchor.mask)
        box = [[x / w, y / h, bw / w, bh / h]]
        session = self._predictor.handle_request(
            {"type": "start_session", "resource_path": str(self._frames_dir), "offload_video_to_cpu": True}
        )
        session_id = session["session_id"]
        predictions: dict[int, Prediction] = {}
        try:
            # The box is the verified SAM3 anchor geometry. Text preserves semantic
            # grounding while SAM3's native memory tracks it through the interval.
            self._predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": anchor.frame_id,
                    "text": self._target.label,
                    "bounding_boxes": box,
                    "bounding_box_labels": [1],
                    "output_prob_thresh": self._mask_threshold,
                }
            )
            begin = time.perf_counter()
            stream = self._predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": direction,
                    "start_frame_index": anchor.frame_id,
                    "max_frame_num_to_track": frame_count,
                    "output_prob_thresh": self._mask_threshold,
                }
            )
            last = begin
            for response in stream:
                now = time.perf_counter()
                frame_id = int(response["frame_index"])
                mask, confidence = self._select_output(response["outputs"], (h, w))
                elapsed = now - last
                last = now
                predictions[frame_id] = Prediction(
                    probability=mask.astype(np.float32) * confidence,
                    confidence=confidence,
                    elapsed_s=elapsed,
                )
                self.frame_times_s.append(elapsed)
            # The prompted frame is a trusted image-SAM anchor, not a VOS estimate.
            predictions[anchor.frame_id] = Prediction(
                probability=anchor.mask.astype(np.float32), confidence=1.0, elapsed_s=0.0
            )
            return predictions
        finally:
            self._predictor.handle_request(
                {"type": "close_session", "session_id": session_id, "run_gc_collect": False}
            )
            gc.collect()

    @staticmethod
    def _select_output(outputs: dict, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
        masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
        probabilities = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
        if masks.ndim != 3 or masks.shape[0] == 0:
            return np.zeros(shape, dtype=bool), 0.0
        if probabilities.size != masks.shape[0]:
            probabilities = np.ones(masks.shape[0], dtype=np.float32)
        # A single visual prompt should yield one object. In the rare case that
        # SAM3 returns several tracks, retain the most confident one, rather than
        # merging semantically different instances.
        index = int(np.argmax(probabilities))
        mask = masks[index]
        if mask.shape != shape:
            mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask, float(np.clip(probabilities[index], 0.0, 1.0))

    def close(self) -> None:
        self._predictor.shutdown()


def postprocess_mask(mask: np.ndarray, options: Segmentation) -> np.ndarray:
    """Conservative post-processing: no dilation and no large-hole filling."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= options.min_component_area_px:
            cleaned[labels == label] = True
    if not options.allow_hole_filling or options.max_hole_area_px <= 0 or not cleaned.any():
        return cleaned
    inverse = (~cleaned).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    h, w = mask.shape
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and int(area) <= options.max_hole_area_px:
            cleaned[labels == label] = True
    return cleaned


def evaluate_mask(
    image: np.ndarray,
    probability: np.ndarray,
    mask_threshold: float,
    fb_iou: float | None,
) -> MaskQuality:
    mask = probability >= mask_threshold
    area = int(mask.sum())
    mean_probability = float(probability[mask].mean()) if area else 0.0
    component_ratio = _largest_component_ratio(mask)
    hole_ratio = _hole_ratio(mask)
    edge_ratio = _edge_ratio(mask)
    sharpness, exposure, image_quality = image_quality_score(image)
    fb = 1.0 if fb_iou is None else float(np.clip(fb_iou, 0.0, 1.0))
    # Equal, explicit weights make individual failure modes auditable in CSV.
    score = float(np.mean((mean_probability, component_ratio, 1.0 - hole_ratio, 1.0 - edge_ratio, image_quality, fb)))
    return MaskQuality(score, mean_probability, component_ratio, hole_ratio, edge_ratio, image_quality, sharpness, exposure)


def image_quality_score(image: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 300.0, 0.0, 1.0))
    valid_exposure = (gray >= 8) & (gray <= 247)
    exposure = float(valid_exposure.mean())
    return sharpness, exposure, 0.5 * sharpness + 0.5 * exposure


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def anchor_descriptor(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    x, y, w, h = _box_xywh(mask)
    roi = image[y : y + h, x : x + w]
    roi_mask = mask[y : y + h, x : x + w]
    if roi.size == 0 or not roi_mask.any():
        return np.zeros(512, dtype=np.float32), 1.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], roi_mask.astype(np.uint8), [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist.astype(np.float32)
    hist /= max(float(np.linalg.norm(hist)), 1e-8)
    return hist, max(w / max(h, 1), h / max(w, 1))


def view_novelty(image: np.ndarray, mask: np.ndarray, anchors: list[Anchor]) -> float:
    if not anchors or not mask.any():
        return 1.0
    appearance, ratio = anchor_descriptor(image, mask)
    normalized = _normalized_mask(mask)
    distances: list[float] = []
    for anchor in anchors:
        silhouette = 1.0 - mask_iou(normalized, _normalized_mask(anchor.mask))
        ratio_distance = abs(float(np.log(max(ratio, 1e-6) / max(anchor.aspect_ratio, 1e-6))))
        cosine = float(np.dot(appearance, anchor.appearance))
        distances.append(0.45 * silhouette + 0.15 * ratio_distance + 0.40 * (1.0 - cosine))
    return float(min(distances))


def _normalized_mask(mask: np.ndarray) -> np.ndarray:
    x, y, w, h = _box_xywh(mask)
    crop = mask[y : y + h, x : x + w].astype(np.uint8)
    return cv2.resize(crop, (128, 128), interpolation=cv2.INTER_NEAREST).astype(bool)


def _box_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _largest_component_ratio(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area == 0:
        return 0.0
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return float(stats[1:, cv2.CC_STAT_AREA].max() / area)


def _hole_ratio(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area == 0:
        return 1.0
    inverse = (~mask).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    h, w = mask.shape
    holes = 0
    for label in range(1, count):
        x, y, bw, bh, component_area = stats[label]
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            holes += int(component_area)
    return float(holes / area)


def _edge_ratio(mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    border = np.zeros_like(mask, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    edge_pixels = int((mask & border).sum())
    perimeter = max(int(cv2.Canny(mask.astype(np.uint8) * 255, 50, 150).astype(bool).sum()), 1)
    return float(min(1.0, edge_pixels / perimeter))
