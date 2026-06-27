from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Topics:
    image: str
    image_pose: str
    pointcloud: str
    pointcloud_pose: str
    mask: str = "/proxy_model/object_mask"


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "pinhole"
    distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def matrix(self):
        import numpy as np

        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Qwen:
    enabled: bool = True
    model: str = "qwen3-vl-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_s: float = 60.0
    max_retries: int = 2
    image_max_side: int = 1280
    label: str = ""
    mode: str = "one"
    center_point: tuple[float, float] | None = None


@dataclass(frozen=True)
class Segmentation:
    prompt: str
    seed_frame: int = 0
    sam_interval: int = 5
    sam_agent_root: Path = Path("/home/yc/SAM-AGENT")
    checkpoint: Path = Path("/home/yc/SAM-AGENT/weights/sam3.pt")
    device: str = "cuda"
    confidence_threshold: float = 0.3
    mask_dilate_px: int = 2
    flow_mask_dilate_px: int = 0
    sam_refine_margin_px: int = 12
    min_mask_area_px: int = 100
    qwen: Qwen = field(default_factory=Qwen)


@dataclass(frozen=True)
class PointFilter:
    min_depth_m: float = 0.05
    max_depth_m: float = 100.0
    pixel_stride: int = 2
    depth_tolerance_m: float = 0.15
    depth_tolerance_ratio: float = 0.0
    edge_band_px: int = 0
    edge_depth_tolerance_m: float = 0.03
    time_tolerance_s: float = 0.02
    multiview_window: int = 2
    multiview_min_views: int = 2
    multiview_ratio: float = 0.9
    cluster_voxel_size_m: float = 0.02
    cluster_radius_m: float = 0.06
    cluster_min_points: int = 20
    keep_largest_cluster: bool = True


@dataclass(frozen=True)
class Output:
    cache_dir: Path
    jpeg_quality: int = 95
    save_masks: bool = True
    save_overlays: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class AppConfig:
    input_bag: Path
    output_bag: Path
    topics: Topics
    camera: Camera
    segmentation: Segmentation
    point_filter: PointFilter
    output: Output


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required configuration field: {key}")
    return data[key]


def load_config(path: str | Path) -> AppConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent

    def resolve(value: str | Path) -> Path:
        p = Path(value).expanduser()
        return (base / p).resolve() if not p.is_absolute() else p.resolve()

    topics_raw = _required(raw, "topics")
    camera_raw = _required(raw, "camera")
    seg_raw = _required(raw, "segmentation")
    qwen_raw = seg_raw.get("qwen", {})
    point_raw = raw.get("point_filter", {})
    output_raw = raw.get("output", {})

    distortion = tuple(float(v) for v in camera_raw.get("distortion", [0, 0, 0, 0, 0]))
    center = qwen_raw.get("center_point")
    center_point = tuple(float(v) for v in center) if center is not None else None

    return AppConfig(
        input_bag=resolve(_required(raw, "input_bag")),
        output_bag=resolve(_required(raw, "output_bag")),
        topics=Topics(
            image=_required(topics_raw, "image"),
            image_pose=_required(topics_raw, "image_pose"),
            pointcloud=_required(topics_raw, "pointcloud"),
            pointcloud_pose=_required(topics_raw, "pointcloud_pose"),
            mask=str(topics_raw.get("mask", "/proxy_model/object_mask")),
        ),
        camera=Camera(
            width=int(_required(camera_raw, "width")),
            height=int(_required(camera_raw, "height")),
            fx=float(_required(camera_raw, "fx")),
            fy=float(_required(camera_raw, "fy")),
            cx=float(_required(camera_raw, "cx")),
            cy=float(_required(camera_raw, "cy")),
            distortion_model=str(camera_raw.get("distortion_model", "pinhole")),
            distortion=distortion,
        ),
        segmentation=Segmentation(
            prompt=str(_required(seg_raw, "prompt")),
            seed_frame=int(seg_raw.get("seed_frame", 0)),
            sam_interval=max(1, int(seg_raw.get("sam_interval", 5))),
            sam_agent_root=resolve(seg_raw.get("sam_agent_root", "/home/yc/SAM-AGENT")),
            checkpoint=resolve(seg_raw.get("checkpoint", "/home/yc/SAM-AGENT/weights/sam3.pt")),
            device=str(seg_raw.get("device", "cuda")),
            confidence_threshold=float(seg_raw.get("confidence_threshold", 0.3)),
            mask_dilate_px=max(0, int(seg_raw.get("mask_dilate_px", 2))),
            flow_mask_dilate_px=max(0, int(seg_raw.get("flow_mask_dilate_px", 0))),
            sam_refine_margin_px=max(0, int(seg_raw.get("sam_refine_margin_px", 12))),
            min_mask_area_px=max(1, int(seg_raw.get("min_mask_area_px", 100))),
            qwen=Qwen(
                enabled=bool(qwen_raw.get("enabled", True)),
                model=str(qwen_raw.get("model", "qwen3-vl-plus")),
                base_url=str(
                    qwen_raw.get(
                        "base_url",
                        "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                ),
                api_key_env=str(qwen_raw.get("api_key_env", "DASHSCOPE_API_KEY")),
                timeout_s=float(qwen_raw.get("timeout_s", 60.0)),
                max_retries=int(qwen_raw.get("max_retries", 2)),
                image_max_side=int(qwen_raw.get("image_max_side", 1280)),
                label=str(qwen_raw.get("label", "")),
                mode=str(qwen_raw.get("mode", "one")),
                center_point=center_point,
            ),
        ),
        point_filter=PointFilter(
            min_depth_m=float(point_raw.get("min_depth_m", 0.05)),
            max_depth_m=float(point_raw.get("max_depth_m", 100.0)),
            pixel_stride=max(1, int(point_raw.get("pixel_stride", 2))),
            depth_tolerance_m=max(0.0, float(point_raw.get("depth_tolerance_m", 0.15))),
            depth_tolerance_ratio=max(0.0, float(point_raw.get("depth_tolerance_ratio", 0.0))),
            edge_band_px=max(0, int(point_raw.get("edge_band_px", 0))),
            edge_depth_tolerance_m=max(
                0.0, float(point_raw.get("edge_depth_tolerance_m", 0.03))
            ),
            time_tolerance_s=max(0.0, float(point_raw.get("time_tolerance_s", 0.02))),
            multiview_window=max(0, int(point_raw.get("multiview_window", 2))),
            multiview_min_views=max(1, int(point_raw.get("multiview_min_views", 2))),
            multiview_ratio=min(1.0, max(0.0, float(point_raw.get("multiview_ratio", 0.9)))),
            cluster_voxel_size_m=max(
                1e-4, float(point_raw.get("cluster_voxel_size_m", 0.02))
            ),
            cluster_radius_m=max(0.0, float(point_raw.get("cluster_radius_m", 0.06))),
            cluster_min_points=max(1, int(point_raw.get("cluster_min_points", 20))),
            keep_largest_cluster=bool(point_raw.get("keep_largest_cluster", True)),
        ),
        output=Output(
            cache_dir=resolve(output_raw.get("cache_dir", ".proxy_model_cache")),
            jpeg_quality=int(output_raw.get("jpeg_quality", 95)),
            save_masks=bool(output_raw.get("save_masks", True)),
            save_overlays=bool(output_raw.get("save_overlays", True)),
            overwrite=bool(output_raw.get("overwrite", False)),
        ),
    )
