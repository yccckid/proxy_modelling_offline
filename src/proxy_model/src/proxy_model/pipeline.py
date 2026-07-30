from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import AppConfig
from .geometry import (
    forward_warp_mask_with_flow,
    geometric_forward_flow,
    largest_cluster_keep,
    lidar_to_camera,
    points_in_mask,
    points_mask_consistency,
    points_projected_in_mask,
    pose_matrix,
    transform_points,
    voxel_downsample,
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
    dense_optical_flow,
    flow_to_color,
    mask_centroid,
    postprocess_mask,
    warp_mask,
)


@dataclass(frozen=True)
class TimedPoints:
    stamp_ns: int
    points: np.ndarray


class ObjectBagPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.frames_dir = config.output.cache_dir / "frames"
        self.masks_dir = config.output.cache_dir / "masks"
        self.overlays_dir = config.output.cache_dir / "overlays"
        self.flows_dir = config.output.cache_dir / "flows"
        self.labeled_pcd_dir = config.output.cache_dir / "labeled_pcd"
        self.point_images_dir = config.output.cache_dir / "point_image"
        self.image_stamps: list[int] = []
        self.image_paths: list[Path] = []
        self.camera_poses: list[TimedMessage] = []
        self.lidar_poses: list[TimedMessage] = []
        self.pointclouds: list[TimedPoints] = []
        self.target: Target | None = None
        self.geometry_flow_frames = 0
        self.farneback_flow_frames = 0

    def run(self) -> None:
        self._validate()
        self._prepare_cache()
        started = time.perf_counter()
        self._extract_frames_and_poses()
        self._segment_frames()
        stats = self._rewrite_bag()
        stats["geometric_flow_frames"] = self.geometry_flow_frames
        stats["farneback_flow_fallback_frames"] = self.farneback_flow_frames
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
        if self.config.output.save_flows:
            self.flows_dir.mkdir(parents=True, exist_ok=True)
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
            topics.pointcloud,
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
                    count += 1
                if topic == topics.image_pose:
                    self.camera_poses.append(timed)
                if topic == topics.pointcloud_pose:
                    self.lidar_poses.append(timed)
                if topic == topics.pointcloud:
                    points = voxel_downsample(
                        xyz_view(message),
                        self.config.segmentation.geometry_flow_voxel_size_m,
                    )
                    self.pointclouds.append(TimedPoints(stamp_ns, points))
        self.camera_poses.sort(key=lambda item: item.stamp_ns)
        self.lidar_poses.sort(key=lambda item: item.stamp_ns)
        self.pointclouds.sort(key=lambda item: item.stamp_ns)
        if not self.image_paths:
            raise RuntimeError(f"No images found on topic {topics.image}")
        if not self.camera_poses or not self.lidar_poses:
            raise RuntimeError("Camera or lidar odometry topic is empty")
        if self.config.segmentation.geometry_flow_enabled and not self.pointclouds:
            print("Warning: pointcloud topic is empty; geometry flow will fall back to Farneback")

    def _segment_frames(self) -> None:
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
        seed_mask = sam.predict(seed_image, self.target, self.target.center_point)
        self._save_mask(seed, seed_image, seed_mask, self.config.segmentation.mask_dilate_px)
        self._save_flow(seed, np.zeros((*seed_image.shape[:2], 2), dtype=np.float32))

        self._segment_direction(sam, seed + 1, len(self.image_paths), 1, seed)
        self._segment_direction(sam, seed - 1, -1, -1, seed)

    def _segment_direction(
        self, sam: SamSegmenter, start: int, stop: int, step: int, seed: int
    ) -> None:
        if start == stop:
            return
        previous_index = seed
        previous_image = self._read_frame(previous_index)
        previous_mask = self._load_mask(seed)
        point = mask_centroid(previous_mask)
        interval = self.config.segmentation.sam_interval

        for index in tqdm(range(start, stop, step), desc=f"Segmenting {'forward' if step > 0 else 'backward'}"):
            image = self._read_frame(index)
            geometric_result = self._geometric_warp_mask(
                previous_index, index, previous_mask
            )
            if geometric_result is None:
                propagated, flow = warp_mask(
                    previous_image,
                    image,
                    previous_mask,
                    self.config.segmentation.flow_smoothing_sigma_px,
                )
                self.farneback_flow_frames += 1
                self._save_flow(
                    index,
                    flow,
                    self.config.segmentation.flow_visual_min_magnitude_px,
                )
            else:
                propagated, geometric_flow = geometric_result
                self.geometry_flow_frames += 1
                background_flow = dense_optical_flow(
                    previous_image,
                    image,
                    self.config.segmentation.flow_smoothing_sigma_px,
                )
                self._save_flow(
                    index,
                    background_flow,
                    self.config.segmentation.flow_visual_min_magnitude_px,
                    target_mask=propagated,
                    target_flow=geometric_flow,
                )
            use_sam = abs(index - seed) % interval == 0
            if use_sam:
                try:
                    prompt_point = mask_centroid(propagated) if self.target.mode == "one" else None
                    sam_mask = sam.predict(image, self.target, prompt_point or point)
                    mask = self._constrain_sam_mask(sam_mask, propagated)
                except Exception as error:
                    print(f"\nWarning: SAM failed at frame {index}, using optical flow: {error}")
                    mask = propagated
            else:
                mask = propagated
            dilate_px = (
                self.config.segmentation.mask_dilate_px
                if use_sam
                else self.config.segmentation.flow_mask_dilate_px
            )
            self._save_mask(index, image, mask, dilate_px)
            previous_index, previous_image, previous_mask = (
                index,
                image,
                self._load_mask(index),
            )
            point = mask_centroid(previous_mask) or point

    def _geometric_warp_mask(
        self, previous_index: int, current_index: int, previous_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Propagate a target mask from static 3-D points when they are available."""
        options = self.config.segmentation
        if not options.geometry_flow_enabled:
            return None
        tolerance_s = self.config.point_filter.time_tolerance_s
        previous_stamp = self.image_stamps[previous_index]
        current_stamp = self.image_stamps[current_index]
        pointcloud = nearest(self.pointclouds, previous_stamp, tolerance_s)
        previous_camera = nearest(self.camera_poses, previous_stamp, tolerance_s)
        current_camera = nearest(self.camera_poses, current_stamp, tolerance_s)
        if pointcloud is None or previous_camera is None or current_camera is None:
            return None
        lidar_pose = nearest(self.lidar_poses, pointcloud.stamp_ns, tolerance_s)
        if lidar_pose is None:
            return None
        forward_flow = geometric_forward_flow(
            pointcloud.points,
            lidar_to_camera(previous_camera.message, lidar_pose.message),
            lidar_to_camera(current_camera.message, lidar_pose.message),
            previous_mask,
            self.config.camera,
            self.config.point_filter.min_depth_m,
            self.config.point_filter.max_depth_m,
            options.geometry_flow_min_seed_points,
        )
        if forward_flow is None:
            return None
        return forward_warp_mask_with_flow(previous_mask, forward_flow)

    def _constrain_sam_mask(self, sam_mask: np.ndarray, flow_prior: np.ndarray) -> np.ndarray:
        margin = self.config.segmentation.sam_refine_margin_px
        if margin <= 0 or not flow_prior.any():
            return sam_mask
        size = 2 * margin + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        prior_band = cv2.dilate(flow_prior.astype(np.uint8), kernel).astype(bool)
        constrained = sam_mask & prior_band
        if int(constrained.sum()) >= self.config.segmentation.min_mask_area_px:
            return constrained
        return flow_prior

    def _save_mask(self, index: int, image: np.ndarray, mask: np.ndarray, dilate_px: int) -> None:
        mask = postprocess_mask(mask, dilate_px)
        path = self.masks_dir / f"{index:06d}.png"
        if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Failed to save mask: {path}")
        if self.config.output.save_overlays:
            write_frame(self.overlays_dir / f"{index:06d}.jpg", mask_overlay(image, mask))

    def _save_flow(
        self,
        index: int,
        flow: np.ndarray,
        min_magnitude_px: float = 0.0,
        target_mask: np.ndarray | None = None,
        target_flow: np.ndarray | None = None,
    ) -> None:
        if not self.config.output.save_flows:
            return
        path = self.flows_dir / f"{index:06d}.png"
        visualization = flow_to_color(flow, min_magnitude_px)
        if target_mask is not None and target_flow is not None:
            # Keep sub-pixel geometry flow visible on the target while the rest
            # of the image uses thresholded Farneback flow to suppress noise.
            target_visualization = flow_to_color(target_flow, 0.0)
            visualization[target_mask] = target_visualization[target_mask]
        if not cv2.imwrite(str(path), visualization):
            raise RuntimeError(f"Failed to save dense optical-flow visualization: {path}")

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
                        object_labeled_points += int(object_labels.sum())
                        export_name = f"{point_count:06d}"
                        if self.config.output.save_labeled_pcd:
                            transform_world_lidar = pose_matrix(lidar_pose.message)
                            points_world = transform_points(points, transform_world_lidar)
                            write_labeled_pcd(
                                self.labeled_pcd_dir / f"{export_name}.pcd",
                                points_world,
                                object_labels,
                            )
                            labeled_pcd_count += 1
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
                    output_points += int(keep.sum())
                    destination.write(topic, filter_pointcloud(message, keep), bag_time)
                    point_count += 1
                    continue

                destination.write(topic, message, bag_time)

        return {
            "images": image_count,
            "masks": image_count,
            "flow_visualizations": len(self.image_paths)
            if self.config.output.save_flows
            else 0,
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
