from __future__ import annotations

import csv
import json
import resource
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import AppConfig
from .geometry import (
    largest_cluster_keep,
    lidar_to_camera,
    points_in_mask,
    points_mask_consistency,
    points_projected_in_mask,
    pose_matrix,
    transform_points,
)
from .qwen import Target, resolve_target
from .ros_utils import (
    TimedMessage,
    apply_mask,
    decode_image,
    encode_mask_image,
    encode_masked_image,
    filter_pointcloud,
    mask_overlay,
    message_stamp_ns,
    nearest,
    pointcloud_mask_overlay,
    write_frame,
    write_labeled_pcd,
    xyz_view,
)
from .segmenter import (
    SamSegmenter,
    mask_centroid,
)
from .memory_vos import (
    Anchor,
    Prediction,
    Sam3MemoryVOS,
    anchor_descriptor,
    evaluate_mask,
    mask_iou,
    postprocess_mask,
    view_novelty,
)


class ObjectBagPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.frames_dir = config.output.cache_dir / "frames"
        self.masks_dir = config.output.cache_dir / "masks"
        self.overlays_dir = config.output.cache_dir / "overlays"
        self.probabilities_dir = config.output.cache_dir / "probabilities"
        self.disagreements_dir = config.output.cache_dir / "disagreements"
        self.metrics_path = config.output.cache_dir / "segmentation_metrics.csv"
        self.labeled_pcd_dir = config.output.cache_dir / "labeled_pcd"
        self.point_images_dir = config.output.cache_dir / "point_image"
        self.image_stamps: list[int] = []
        self.image_paths: list[Path] = []
        self.image_source_indices: list[int] = []
        self.camera_poses: list[TimedMessage] = []
        self.lidar_poses: list[TimedMessage] = []
        self.target: Target | None = None
        self.segmentation_rows: list[dict[str, object]] = []
        self.sam3_calls = 0
        self.sam3_recoveries = 0
        self.identity_rejections = 0
        self.permanent_anchors: list[Anchor] = []
        self.anchor_novelty_by_id: dict[int, float] = {}
        self.vos: Sam3MemoryVOS | None = None
        self.vos_frame_times_s: list[float] = []
        self.sam3_frame_times_s: dict[int, float] = {}
        self.working_memory_peak_frames = 0
        self.segmentation_elapsed_s = 0.0

    def run(self) -> None:
        self._validate()
        self._prepare_cache()
        started = time.perf_counter()
        self._extract_frames_and_poses()
        self._segment_frames()
        stats = self._rewrite_bag()
        stats["segmentation"] = self._segmentation_statistics()
        stats["elapsed_s"] = round(time.perf_counter() - started, 3)
        stats["target"] = asdict(self.target) if self.target else None
        stats["input_bag"] = str(self.config.input_bag)
        stats["output_bag"] = str(self.config.output_bag)
        (self.config.output.cache_dir / "summary.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    def _validate(self) -> None:
        if not self.config.input_bag.is_file():
            raise FileNotFoundError(f"Input bag not found: {self.config.input_bag}")
        if self.config.output_bag.exists() and not self.config.output.overwrite:
            raise FileExistsError(
                f"Output bag exists: {self.config.output_bag}. "
                "Set output.overwrite: true to replace it."
            )
        if self.config.input_bag == self.config.output_bag:
            raise ValueError("input_bag and output_bag must be different")

    def _prepare_cache(self) -> None:
        if self.config.output.cache_dir.exists() and self.config.output.overwrite:
            shutil.rmtree(self.config.output.cache_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output.save_overlays:
            self.overlays_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output.save_probabilities:
            self.probabilities_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output.save_disagreements:
            self.disagreements_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output.save_labeled_pcd:
            self.labeled_pcd_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output.save_point_images:
            self.point_images_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_bag.parent.mkdir(parents=True, exist_ok=True)
        if self.config.output_bag.exists():
            self.config.output_bag.unlink()

    def _extract_frames_and_poses(self) -> None:
        import rosbag

        topics = self.config.topics
        interested = [
            topics.image,
            topics.image_pose,
            topics.pointcloud_pose,
        ]
        with rosbag.Bag(str(self.config.input_bag), "r") as bag:
            count = 0
            for topic, message, bag_time in tqdm(
                bag.read_messages(topics=interested),
                total=sum(
                    bag.get_type_and_topic_info().topics[t].message_count
                    for t in set(interested)
                ),
                desc="Extracting",
            ):
                stamp_ns = message_stamp_ns(message, bag_time)
                timed = TimedMessage(stamp_ns, int(bag_time.to_nsec()), message)
                if topic == topics.image:
                    image = decode_image(message)
                    if image.shape[1] != self.config.camera.width or image.shape[0] != self.config.camera.height:
                        raise ValueError(
                            f"Image is {image.shape[1]}x{image.shape[0]}, configured camera is "
                            f"{self.config.camera.width}x{self.config.camera.height}"
                        )
                    path = self.frames_dir / f"{count:06d}.jpg"
                    write_frame(path, image)
                    self.image_stamps.append(stamp_ns)
                    self.image_paths.append(path)
                    self.image_source_indices.append(count)
                    count += 1
                if topic == topics.image_pose:
                    self.camera_poses.append(timed)
                if topic == topics.pointcloud_pose:
                    self.lidar_poses.append(timed)
        self.camera_poses.sort(key=lambda item: item.stamp_ns)
        self.lidar_poses.sort(key=lambda item: item.stamp_ns)
        self._sort_frames_by_timestamp()
        if not self.image_paths:
            raise RuntimeError(f"No images found on topic {topics.image}")
        if not self.camera_poses or not self.lidar_poses:
            raise RuntimeError("Camera or lidar odometry topic is empty")

    def _sort_frames_by_timestamp(self) -> None:
        """Apply the document's timestamp-stable ordering after bag decoding."""
        ordered = sorted(
            zip(self.image_stamps, self.image_source_indices, self.image_paths),
            key=lambda item: (item[0], item[1]),
        )
        if [path for _, _, path in ordered] == self.image_paths:
            return
        temporary: list[tuple[int, int, Path]] = []
        for output_index, (stamp, source_index, path) in enumerate(ordered):
            temporary_path = self.frames_dir / f".sort_{output_index:06d}.jpg"
            path.replace(temporary_path)
            temporary.append((stamp, source_index, temporary_path))
        self.image_stamps = []
        self.image_source_indices = []
        self.image_paths = []
        for output_index, (stamp, source_index, temporary_path) in enumerate(temporary):
            path = self.frames_dir / f"{output_index:06d}.jpg"
            temporary_path.replace(path)
            self.image_stamps.append(stamp)
            self.image_source_indices.append(source_index)
            self.image_paths.append(path)

    def _segment_frames(self) -> None:
        segmentation_started = time.perf_counter()
        seed = self.config.segmentation.seed_frame
        if not 0 <= seed < len(self.image_paths):
            raise ValueError(f"seed_frame {seed} is outside [0, {len(self.image_paths) - 1}]")
        seed_image = self._read_frame(seed)
        pil_seed = Image.fromarray(cv2.cvtColor(seed_image, cv2.COLOR_BGR2RGB))
        self.target = resolve_target(
            pil_seed, self.config.segmentation.prompt, self.config.segmentation.qwen
        )
        print(
            f"Target: label={self.target.label!r}, mode={self.target.mode}, "
            f"center={self.target.center_point}"
        )
        sam = SamSegmenter(self.config.segmentation)
        sam3_started = time.perf_counter()
        seed_mask, seed_confidence = sam.predict_with_confidence(
            seed_image, self.target, self.target.center_point
        )
        self.sam3_frame_times_s[seed] = time.perf_counter() - sam3_started
        self.sam3_calls += 1
        seed_anchor = self._make_anchor(seed, seed_image, seed_mask, seed_confidence, "sam3_seed")
        if seed_anchor is None:
            raise RuntimeError("The initial SAM3 mask did not pass anchor-quality validation")
        self.permanent_anchors = [seed_anchor]
        self.anchor_novelty_by_id[seed] = 0.0
        self.vos = Sam3MemoryVOS(self.config.segmentation, self.frames_dir, self.target)
        try:
            provisional = self._initial_memory_pass(seed_anchor)
            self._add_new_view_anchors(sam, provisional)
            self._ensure_terminal_anchors(sam, provisional)
            final_predictions = self._bidirectional_memory_pass()
            self._recover_failed_frames(sam, final_predictions)
            self._write_segmentation_outputs(final_predictions)
            self.segmentation_elapsed_s = time.perf_counter() - segmentation_started
            self._write_segmentation_metrics()
        finally:
            self.segmentation_elapsed_s = time.perf_counter() - segmentation_started
            if self.vos is not None:
                self.vos_frame_times_s = list(self.vos.frame_times_s)
                self.working_memory_peak_frames = self.vos.peak_working_memory
            self.vos.close()
            self.vos = None

    def _initial_memory_pass(self, seed: Anchor) -> dict[int, Prediction]:
        assert self.vos is not None
        count = len(self.image_paths)
        outputs: dict[int, Prediction] = {}
        outputs.update(self.vos.propagate(seed, "forward", count - 1 - seed.frame_id))
        if seed.frame_id > 0:
            outputs.update(self.vos.propagate(seed, "backward", seed.frame_id))
        return outputs

    def _add_new_view_anchors(self, sam: SamSegmenter, provisional: dict[int, Prediction]) -> None:
        options = self.config.segmentation
        candidates: list[tuple[float, int]] = []
        for index, prediction in provisional.items():
            if index == self.permanent_anchors[0].frame_id:
                continue
            image = self._read_frame(index)
            quality = evaluate_mask(image, prediction.probability, options.mask_threshold, None)
            mask = prediction.probability >= options.mask_threshold
            novelty = view_novelty(image, mask, self.permanent_anchors)
            if quality.score >= options.quality_threshold and novelty >= options.view_novelty_threshold:
                candidates.append((novelty, index))
        # Novel views first; the gap test below is temporal NMS.
        for _, index in sorted(candidates, reverse=True):
            if len(self.permanent_anchors) >= options.max_anchor_memory:
                break
            if not self._far_from_anchors(index):
                continue
            anchor = self._sam_anchor(sam, index, provisional[index], "sam3_view_anchor")
            if anchor is not None and self._identity_similarity(anchor) >= 0.10:
                self.permanent_anchors.append(anchor)
                self.anchor_novelty_by_id[index] = novelty
            elif anchor is not None:
                self.identity_rejections += 1

    def _ensure_terminal_anchors(self, sam: SamSegmenter, provisional: dict[int, Prediction]) -> None:
        """Try to establish end anchors without a fixed SAM3 interval."""
        if len(self.permanent_anchors) >= self.config.segmentation.max_anchor_memory:
            return
        for index in (0, len(self.image_paths) - 1):
            if any(anchor.frame_id == index for anchor in self.permanent_anchors):
                continue
            if not self._far_from_anchors(index):
                continue
            prediction = provisional.get(index)
            if prediction is None:
                continue
            anchor = self._sam_anchor(sam, index, prediction, "sam3_terminal_anchor")
            if anchor is not None and self._identity_similarity(anchor) >= 0.10:
                self.permanent_anchors.append(anchor)
                self.anchor_novelty_by_id[index] = view_novelty(
                    self._read_frame(index), anchor.mask,
                    [item for item in self.permanent_anchors if item.frame_id != index],
                )
            elif anchor is not None:
                self.identity_rejections += 1

    def _bidirectional_memory_pass(self) -> dict[int, dict[str, object]]:
        assert self.vos is not None
        anchors = sorted(self.permanent_anchors, key=lambda item: item.frame_id)
        count = len(self.image_paths)
        result: dict[int, dict[str, object]] = {}
        for anchor in anchors:
            result[anchor.frame_id] = {
                "probability": anchor.mask.astype(np.float32),
                "forward": None,
                "backward": None,
                "source": "sam3_anchor",
                "anchor_ids": str(anchor.frame_id),
                "vos_elapsed_s": 0.0,
                "fb_iou": 1.0,
                "disagreement": None,
                "failure_flags": "",
            }
        if anchors[0].frame_id > 0:
            backward = self.vos.propagate(anchors[0], "backward", anchors[0].frame_id)
            self._merge_one_direction(result, backward, "backward", anchors[0].frame_id, -1, anchors[0].frame_id)
        for left, right in zip(anchors, anchors[1:]):
            forward = self.vos.propagate(left, "forward", right.frame_id - left.frame_id)
            backward = self.vos.propagate(right, "backward", right.frame_id - left.frame_id)
            self._merge_interval(result, forward, backward, left.frame_id, right.frame_id)
        if anchors[-1].frame_id < count - 1:
            forward = self.vos.propagate(anchors[-1], "forward", count - 1 - anchors[-1].frame_id)
            self._merge_one_direction(result, forward, "forward", anchors[-1].frame_id, anchors[-1].frame_id, count)
        return result

    def _merge_interval(
        self,
        result: dict[int, dict[str, object]],
        forward: dict[int, Prediction],
        backward: dict[int, Prediction],
        left: int,
        right: int,
    ) -> None:
        for index in range(left + 1, right):
            fw, bw = forward.get(index), backward.get(index)
            if fw is None or bw is None:
                prediction = fw or bw
                if prediction is not None:
                    self._merge_one_direction(result, {index: prediction}, "forward" if fw else "backward", left, left, right)
                continue
            fw_mask = fw.probability >= self.config.segmentation.mask_threshold
            bw_mask = bw.probability >= self.config.segmentation.mask_threshold
            fb_iou = mask_iou(fw_mask, bw_mask)
            fw_weight = max(fw.confidence, 1e-4) / max(index - left, 1)
            bw_weight = max(bw.confidence, 1e-4) / max(right - index, 1)
            probability = (fw_weight * fw.probability + bw_weight * bw.probability) / (fw_weight + bw_weight)
            result[index] = {
                "probability": probability,
                "forward": fw,
                "backward": bw,
                "source": "fused",
                "anchor_ids": f"{left}|{right}",
                "vos_elapsed_s": fw.elapsed_s + bw.elapsed_s,
                "fb_iou": fb_iou,
                "disagreement": np.abs(fw.probability - bw.probability),
                "failure_flags": "" if fb_iou >= self.config.segmentation.fb_iou_threshold else "low_fb_iou",
            }

    def _merge_one_direction(
        self,
        result: dict[int, dict[str, object]],
        predictions: dict[int, Prediction],
        direction: str,
        anchor_id: int,
        start: int,
        stop: int,
    ) -> None:
        for index, prediction in predictions.items():
            if index < start or index >= stop or index == anchor_id:
                continue
            result[index] = {
                "probability": prediction.probability,
                "forward": prediction if direction == "forward" else None,
                "backward": prediction if direction == "backward" else None,
                "source": "one_direction",
                "anchor_ids": str(anchor_id),
                "vos_elapsed_s": prediction.elapsed_s,
                "fb_iou": None,
                "disagreement": None,
                "failure_flags": "one_direction_only",
            }

    def _recover_failed_frames(self, sam: SamSegmenter, predictions: dict[int, dict[str, object]]) -> None:
        options = self.config.segmentation
        recoveries = 0
        reliable_areas: list[int] = []
        for index in sorted(predictions):
            entry = predictions[index]
            probability = np.asarray(entry["probability"], dtype=np.float32)
            image = self._read_frame(index)
            quality = evaluate_mask(image, probability, options.mask_threshold, entry["fb_iou"])
            mask = probability >= options.mask_threshold
            area_bad = self._area_is_outlier(int(mask.sum()), reliable_areas)
            failed = quality.score < options.quality_threshold or area_bad
            if entry["fb_iou"] is not None and float(entry["fb_iou"]) < options.fb_iou_threshold:
                failed = True
            if failed and entry["source"] != "sam3_anchor" and recoveries < options.max_recovery_frames:
                anchor = self._sam_anchor(sam, index, Prediction(probability, quality.mean_probability, 0.0), "sam3_recovery")
                recoveries += 1
                self.sam3_recoveries += 1
                if anchor is not None:
                    entry["probability"] = anchor.mask.astype(np.float32)
                    entry["source"] = "sam3_recovery"
                    entry["failure_flags"] = str(entry["failure_flags"]) + "|recovered"
                    if self._identity_similarity(anchor) >= 0.10 and len(self.permanent_anchors) < options.max_anchor_memory:
                        self.permanent_anchors.append(anchor)
                        self.anchor_novelty_by_id[index] = view_novelty(
                            image, anchor.mask,
                            [item for item in self.permanent_anchors if item.frame_id != index],
                        )
                    quality = evaluate_mask(image, np.asarray(entry["probability"]), options.mask_threshold, None)
                    mask = np.asarray(entry["probability"]) >= options.mask_threshold
                else:
                    entry["failure_flags"] = str(entry["failure_flags"]) + "|recovery_failed"
            if quality.score >= options.quality_threshold and mask.any():
                reliable_areas.append(int(mask.sum()))

    def _sam_anchor(
        self,
        sam: SamSegmenter,
        index: int,
        prediction: Prediction,
        source: str,
    ) -> Anchor | None:
        probability = prediction.probability
        confidence = float(probability.max())
        if confidence < self.config.segmentation.recovery_prompt_min_confidence:
            nearest = min(self.permanent_anchors, key=lambda item: abs(item.frame_id - index))
            point = mask_centroid(nearest.mask)
        else:
            y, x = np.unravel_index(int(np.argmax(probability)), probability.shape)
            point = (int(x), int(y))
        sam3_started = time.perf_counter()
        try:
            mask, sam_confidence = sam.predict_with_confidence(self._read_frame(index), self.target, point)
            self.sam3_calls += 1
        except Exception as error:
            self.sam3_frame_times_s[index] = self.sam3_frame_times_s.get(index, 0.0) + (
                time.perf_counter() - sam3_started
            )
            print(f"\nWarning: SAM3 recovery failed at frame {index}: {error}")
            return None
        self.sam3_frame_times_s[index] = self.sam3_frame_times_s.get(index, 0.0) + (
            time.perf_counter() - sam3_started
        )
        return self._make_anchor(index, self._read_frame(index), mask, sam_confidence, source)

    def _make_anchor(
        self,
        index: int,
        image: np.ndarray,
        mask: np.ndarray,
        confidence: float,
        source: str,
    ) -> Anchor | None:
        probability = mask.astype(np.float32) * max(float(confidence), self.config.segmentation.mask_threshold)
        quality = evaluate_mask(image, probability, self.config.segmentation.mask_threshold, None)
        if quality.score < self.config.segmentation.quality_threshold:
            return None
        appearance, ratio = anchor_descriptor(image, mask)
        return Anchor(index, mask, quality.score, appearance, ratio, source)

    def _far_from_anchors(self, index: int) -> bool:
        minimum_ns = self.config.segmentation.min_anchor_time_gap_s * 1e9
        return all(abs(self.image_stamps[index] - self.image_stamps[a.frame_id]) >= minimum_ns for a in self.permanent_anchors)

    def _identity_similarity(self, candidate: Anchor) -> float:
        others = [anchor for anchor in self.permanent_anchors if anchor.frame_id != candidate.frame_id]
        if not others:
            return 1.0
        return max(float(np.dot(candidate.appearance, anchor.appearance)) for anchor in others)

    @staticmethod
    def _area_is_outlier(area: int, reliable_areas: list[int]) -> bool:
        if len(reliable_areas) < 3 or area == 0:
            return area == 0
        reference = float(np.median(reliable_areas[-5:]))
        return area < 0.30 * reference or area > 3.0 * reference

    def _write_segmentation_outputs(self, predictions: dict[int, dict[str, object]]) -> None:
        options = self.config.segmentation
        for index in range(len(self.image_paths)):
            entry = predictions.get(index)
            if entry is None:
                raise RuntimeError(f"SAM3 memory predictor produced no result for frame {index}")
            image = self._read_frame(index)
            probability = np.asarray(entry["probability"], dtype=np.float32)
            raw_mask = probability >= options.mask_threshold
            mask = postprocess_mask(raw_mask, options)
            probability = probability.copy()
            probability[~mask] = 0.0
            quality = evaluate_mask(image, probability, options.mask_threshold, entry["fb_iou"])
            self._save_mask(index, image, mask)
            probability_path = ""
            if self.config.output.save_probabilities:
                path = self.probabilities_dir / f"{index:06d}.png"
                if not cv2.imwrite(str(path), np.rint(np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8)):
                    raise RuntimeError(f"Failed to save probability map: {path}")
                probability_path = str(path)
            disagreement_path = ""
            disagreement = entry.get("disagreement")
            if self.config.output.save_disagreements and disagreement is not None:
                path = self.disagreements_dir / f"{index:06d}.png"
                if not cv2.imwrite(str(path), np.rint(np.clip(disagreement, 0.0, 1.0) * 255).astype(np.uint8)):
                    raise RuntimeError(f"Failed to save forward/backward disagreement map: {path}")
                disagreement_path = str(path)
            vos_elapsed_s = float(entry["vos_elapsed_s"])
            vos_total_elapsed_s = self.vos.frame_elapsed_s.get(index, vos_elapsed_s) if self.vos else vos_elapsed_s
            sam3_elapsed_s = self.sam3_frame_times_s.get(index, 0.0)
            self.segmentation_rows.append(
                {
                    "frame_id": index,
                    "timestamp_ns": self.image_stamps[index],
                    "source_index": self.image_source_indices[index],
                    "mask_path": str(self.masks_dir / f"{index:06d}.png"),
                    "probability_path": probability_path,
                    "disagreement_path": disagreement_path,
                    "source": entry["source"],
                    "quality_score": quality.score,
                    "mean_foreground_probability": quality.mean_probability,
                    "largest_component_ratio": quality.component_ratio,
                    "hole_area_ratio": quality.hole_ratio,
                    "edge_touch_ratio": quality.edge_ratio,
                    "image_sharpness": quality.sharpness,
                    "image_exposure": quality.exposure,
                    "image_quality": quality.image_quality,
                    "forward_backward_iou": entry["fb_iou"],
                    "view_novelty": self.anchor_novelty_by_id.get(
                        index,
                        view_novelty(image, mask, self.permanent_anchors),
                    ),
                    "nearest_anchor_ids": entry["anchor_ids"],
                    "is_permanent_anchor": any(a.frame_id == index for a in self.permanent_anchors),
                    "failure_flags": str(entry["failure_flags"]).strip("|"),
                    "vos_elapsed_s": vos_elapsed_s,
                    "vos_total_elapsed_s": vos_total_elapsed_s,
                    "sam3_elapsed_s": sam3_elapsed_s,
                    "segmentation_elapsed_s": vos_total_elapsed_s + sam3_elapsed_s,
                    "sam3_called": str(entry["source"]).startswith("sam3"),
                    "sam3_recovery": entry["source"] == "sam3_recovery",
                    "ground_truth_mask_path": "",
                    "mask_iou_j": "",
                    "boundary_f": "",
                }
            )

    def _save_mask(self, index: int, image: np.ndarray, mask: np.ndarray) -> None:
        path = self.masks_dir / f"{index:06d}.png"
        if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Failed to save mask: {path}")
        if self.config.output.save_overlays:
            write_frame(self.overlays_dir / f"{index:06d}.jpg", mask_overlay(image, mask))

    def _write_segmentation_metrics(self) -> None:
        summary = self._segmentation_statistics()
        fields = list(self.segmentation_rows[0]) + list(summary)
        with self.metrics_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for row in self.segmentation_rows:
                writer.writerow({**row, **summary})

    def _segmentation_statistics(self) -> dict[str, object]:
        vos_times = self.vos.frame_times_s if self.vos else self.vos_frame_times_s
        gpu_peak_mib = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_peak_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
        except ImportError:
            pass
        duration_minutes = (self.image_stamps[-1] - self.image_stamps[0]) / 60e9 if len(self.image_stamps) > 1 else 0.0
        return {
            "sam3_calls_total": self.sam3_calls,
            "vos_mean_frame_s": round(float(np.mean(vos_times)), 6) if vos_times else 0.0,
            "vos_peak_frame_s": round(float(max(vos_times)), 6) if vos_times else 0.0,
            "gpu_peak_mib": gpu_peak_mib,
            "host_peak_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3),
            "permanent_anchor_count": len({anchor.frame_id for anchor in self.permanent_anchors}),
            "working_memory_peak_frames": self.vos.peak_working_memory if self.vos else self.working_memory_peak_frames,
            "processing_s_per_video_min": round(self.segmentation_elapsed_s / duration_minutes, 6) if duration_minutes > 0 else 0.0,
            "sam3_recovery_frame_ratio": round(self.sam3_recoveries / max(len(self.image_paths), 1), 6),
            "sam3_recovery_success_rate": round(
                sum(row["sam3_recovery"] for row in self.segmentation_rows) / self.sam3_recoveries,
                6,
            ) if self.sam3_recoveries else 0.0,
            "tracking_failure_frame_ratio": round(
                sum(bool(row["failure_flags"]) for row in self.segmentation_rows) / max(len(self.segmentation_rows), 1),
                6,
            ),
            "low_quality_frame_ratio": round(
                sum(float(row["quality_score"]) < self.config.segmentation.quality_threshold for row in self.segmentation_rows) / max(len(self.segmentation_rows), 1),
                6,
            ),
            "identity_rejection_count": self.identity_rejections,
        }

    def _read_frame(self, index: int) -> np.ndarray:
        image = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read cached frame: {self.image_paths[index]}")
        return image

    def _load_mask(self, index: int) -> np.ndarray:
        mask = cv2.imread(str(self.masks_dir / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Missing mask for frame {index}")
        return mask > 0

    def _image_index(self, stamp_ns: int) -> int | None:
        import bisect

        position = bisect.bisect_left(self.image_stamps, stamp_ns)
        candidates = range(max(0, position - 1), min(len(self.image_stamps), position + 1))
        if not candidates:
            return None
        index = min(candidates, key=lambda i: abs(self.image_stamps[i] - stamp_ns))
        tolerance_ns = self.config.point_filter.time_tolerance_s * 1e9
        return index if abs(self.image_stamps[index] - stamp_ns) <= tolerance_ns else None

    def _rewrite_bag(self) -> dict[str, int]:
        import rosbag

        topics = self.config.topics
        image_count = point_count = input_points = output_points = unmatched_points = 0
        labeled_pcd_count = point_image_count = object_labeled_points = 0
        label_export_unmatched = 0
        with rosbag.Bag(str(self.config.input_bag), "r") as source, rosbag.Bag(
            str(self.config.output_bag), "w"
        ) as destination:
            total = source.get_message_count()
            for topic, message, bag_time in tqdm(
                source.read_messages(), total=total, desc="Writing bag"
            ):
                if topic == topics.image:
                    index = self._image_index(message_stamp_ns(message, bag_time))
                    if index is None:
                        raise RuntimeError("Could not match image message to cached frame")
                    mask = self._load_mask(index)
                    image = decode_image(message)
                    output_message = encode_masked_image(
                        message,
                        apply_mask(image, mask),
                        self.config.output.jpeg_quality,
                    )
                    destination.write(topic, output_message, bag_time)
                    destination.write(topics.mask, encode_mask_image(message, mask), bag_time)
                    image_count += 1
                    continue

                if topic == topics.pointcloud:
                    stamp_ns = message_stamp_ns(message, bag_time)
                    index = self._image_index(stamp_ns)
                    camera_pose = nearest(
                        self.camera_poses,
                        stamp_ns,
                        self.config.point_filter.time_tolerance_s,
                    )
                    lidar_pose = nearest(
                        self.lidar_poses,
                        stamp_ns,
                        self.config.point_filter.time_tolerance_s,
                    )
                    points = xyz_view(message)
                    input_points += len(points)
                    if index is None or camera_pose is None or lidar_pose is None:
                        keep = np.zeros(len(points), dtype=bool)
                        unmatched_points += len(points)
                        label_export_unmatched += 1
                    else:
                        mask = self._load_mask(index)
                        transform_camera_lidar = lidar_to_camera(
                            camera_pose.message, lidar_pose.message
                        )
                        image_camera_pose = nearest(
                            self.camera_poses,
                            self.image_stamps[index],
                            self.config.point_filter.time_tolerance_s,
                        )
                        projection_camera_pose = image_camera_pose or camera_pose
                        transform_image_camera_lidar = lidar_to_camera(
                            projection_camera_pose.message, lidar_pose.message
                        )
                        object_labels, projected_uv = points_projected_in_mask(
                            points,
                            transform_image_camera_lidar,
                            mask,
                            self.config.camera,
                            self.config.point_filter.min_depth_m,
                            self.config.point_filter.max_depth_m,
                        )
                        export_name = f"{point_count:06d}"
                        if self.config.output.save_point_images:
                            overlay_path = self.overlays_dir / f"{index:06d}.jpg"
                            overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
                            if overlay is None:
                                overlay = mask_overlay(self._read_frame(index), mask)
                            write_frame(
                                self.point_images_dir / f"{export_name}.jpg",
                                pointcloud_mask_overlay(
                                    overlay, projected_uv, object_labels
                                ),
                            )
                            point_image_count += 1
                        keep = points_in_mask(
                            points,
                            transform_camera_lidar,
                            mask,
                            self.config.camera,
                            self.config.point_filter,
                        )
                        keep &= self._multiview_keep(points, index, lidar_pose.message)
                        keep = largest_cluster_keep(points, keep, self.config.point_filter)
                        object_labeled_points += int(keep.sum())
                        if self.config.output.save_labeled_pcd:
                            transform_world_lidar = pose_matrix(lidar_pose.message)
                            points_world = transform_points(points, transform_world_lidar)
                            write_labeled_pcd(
                                self.labeled_pcd_dir / f"{export_name}.pcd",
                                points_world,
                                keep,
                            )
                            labeled_pcd_count += 1
                    output_points += int(keep.sum())
                    destination.write(topic, filter_pointcloud(message, keep), bag_time)
                    point_count += 1
                    continue

                destination.write(topic, message, bag_time)

        return {
            "images": image_count,
            "masks": image_count,
            "probability_maps": len(self.image_paths)
            if self.config.output.save_probabilities
            else 0,
            "forward_backward_disagreement_maps": sum(
                bool(row["disagreement_path"]) for row in self.segmentation_rows
            ) if self.config.output.save_disagreements else 0,
            "pointclouds": point_count,
            "input_points": input_points,
            "output_points": output_points,
            "unmatched_points_removed": unmatched_points,
            "labeled_pcds": labeled_pcd_count,
            "point_images": point_image_count,
            "object_labeled_points": object_labeled_points,
            "label_export_unmatched_frames": label_export_unmatched,
        }

    def _multiview_keep(self, points: np.ndarray, image_index: int, lidar_pose: object) -> np.ndarray:
        options = self.config.point_filter
        if options.multiview_window <= 0:
            return np.ones(len(points), dtype=bool)

        begin = max(0, image_index - options.multiview_window)
        end = min(len(self.image_paths), image_index + options.multiview_window + 1)
        transform_world_lidar = pose_matrix(lidar_pose)
        transforms: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for view_index in range(begin, end):
            camera_pose = nearest(
                self.camera_poses,
                self.image_stamps[view_index],
                self.config.point_filter.time_tolerance_s,
            )
            if camera_pose is None:
                continue
            transforms.append(np.linalg.inv(pose_matrix(camera_pose.message)) @ transform_world_lidar)
            masks.append(self._load_mask(view_index))
        return points_mask_consistency(points, transforms, masks, self.config.camera, options)
