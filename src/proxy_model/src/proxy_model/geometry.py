from __future__ import annotations

import cv2
import numpy as np

from .config import Camera, PointFilter


def pose_matrix(odometry: object) -> np.ndarray:
    p = odometry.pose.pose.position
    q = odometry.pose.pose.orientation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Odometry contains a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = (p.x, p.y, p.z)
    return transform


def lidar_to_camera(camera_pose: object, lidar_pose: object) -> np.ndarray:
    return np.linalg.inv(pose_matrix(camera_pose)) @ pose_matrix(lidar_pose)


def points_in_mask(
    points_lidar: np.ndarray,
    transform_camera_lidar: np.ndarray,
    mask: np.ndarray,
    camera: Camera,
    options: PointFilter,
) -> np.ndarray:
    count = len(points_lidar)
    keep = np.zeros(count, dtype=bool)
    finite = np.isfinite(points_lidar).all(axis=1)
    if not finite.any():
        return keep

    source_indices = np.flatnonzero(finite)
    points = points_lidar[finite]
    points_camera = (
        transform_camera_lidar[:3, :3] @ points.T
    ).T + transform_camera_lidar[:3, 3]
    depth = points_camera[:, 2]
    valid_depth = (depth >= options.min_depth_m) & (depth <= options.max_depth_m)
    if not valid_depth.any():
        return keep

    source_indices = source_indices[valid_depth]
    points_camera = points_camera[valid_depth]
    depth = depth[valid_depth]
    distortion = np.asarray(camera.distortion, dtype=np.float64)
    if camera.distortion_model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            points_camera.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            camera.matrix,
            distortion[:4],
        )
    else:
        projected, _ = cv2.projectPoints(
            points_camera,
            np.zeros(3),
            np.zeros(3),
            camera.matrix,
            distortion,
        )
    uv = np.rint(projected.reshape(-1, 2)).astype(np.int64)
    u, v = uv[:, 0], uv[:, 1]
    h, w = mask.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not inside.any():
        return keep

    source_indices = source_indices[inside]
    u, v, depth = u[inside], v[inside], depth[inside]
    in_mask = mask[v, u]
    if not in_mask.any():
        return keep

    edge = None
    if options.edge_band_px > 0:
        size = 2 * options.edge_band_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        core = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        edge = mask & ~core

    source_indices = source_indices[in_mask]
    u, v, depth = u[in_mask], v[in_mask], depth[in_mask]
    stride = options.pixel_stride
    cell_u, cell_v = u // stride, v // stride
    cells_w = (w + stride - 1) // stride
    cell_id = cell_v * cells_w + cell_u
    nearest_depth = np.full(((h + stride - 1) // stride) * cells_w, np.inf)
    np.minimum.at(nearest_depth, cell_id, depth)
    tolerance = np.maximum(
        options.depth_tolerance_m, nearest_depth[cell_id] * options.depth_tolerance_ratio
    )
    if edge is not None:
        edge_hit = edge[v, u]
        tolerance[edge_hit] = np.minimum(tolerance[edge_hit], options.edge_depth_tolerance_m)
    near_surface = depth <= nearest_depth[cell_id] + tolerance
    keep[source_indices[near_surface]] = True
    return keep


def largest_cluster_keep(
    points: np.ndarray, keep: np.ndarray, options: PointFilter
) -> np.ndarray:
    if not options.keep_largest_cluster or options.cluster_radius_m <= 0:
        return keep

    selected = np.flatnonzero(keep)
    if selected.size < options.cluster_min_points:
        return keep

    pts = points[selected]
    finite = np.isfinite(pts).all(axis=1)
    if finite.sum() < options.cluster_min_points:
        output = np.zeros_like(keep)
        output[selected[finite]] = True
        return output

    selected = selected[finite]
    pts = pts[finite]
    voxel = max(options.cluster_voxel_size_m, options.cluster_radius_m)
    cells = np.floor(pts / voxel).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(i)

    parent = np.arange(len(pts), dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    radius2 = options.cluster_radius_m * options.cluster_radius_m
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    for i, cell in enumerate(cells):
        base = tuple(cell)
        for offset in offsets:
            neighbor = (base[0] + offset[0], base[1] + offset[1], base[2] + offset[2])
            for j in buckets.get(neighbor, []):
                if j <= i:
                    continue
                if float(np.sum((pts[i] - pts[j]) ** 2)) <= radius2:
                    union(i, j)

    roots = np.array([find(i) for i in range(len(pts))], dtype=np.int32)
    unique, counts = np.unique(roots, return_counts=True)
    largest = unique[np.argmax(counts)]
    if counts.max() < options.cluster_min_points:
        return np.zeros_like(keep)

    output = np.zeros_like(keep)
    output[selected[roots == largest]] = True
    return output


def project_points(
    points_lidar: np.ndarray,
    transform_camera_lidar: np.ndarray,
    camera: Camera,
    options: PointFilter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(points_lidar)
    valid_source = np.zeros(count, dtype=bool)
    uv = np.zeros((count, 2), dtype=np.int64)
    depth_all = np.full(count, np.inf, dtype=np.float64)

    finite = np.isfinite(points_lidar).all(axis=1)
    if not finite.any():
        return valid_source, uv, depth_all

    source_indices = np.flatnonzero(finite)
    points = points_lidar[finite]
    points_camera = (
        transform_camera_lidar[:3, :3] @ points.T
    ).T + transform_camera_lidar[:3, 3]
    depth = points_camera[:, 2]
    valid_depth = (depth >= options.min_depth_m) & (depth <= options.max_depth_m)
    if not valid_depth.any():
        return valid_source, uv, depth_all

    source_indices = source_indices[valid_depth]
    points_camera = points_camera[valid_depth]
    depth = depth[valid_depth]
    distortion = np.asarray(camera.distortion, dtype=np.float64)
    if camera.distortion_model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            points_camera.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            camera.matrix,
            distortion[:4],
        )
    else:
        projected, _ = cv2.projectPoints(
            points_camera,
            np.zeros(3),
            np.zeros(3),
            camera.matrix,
            distortion,
        )
    uv[source_indices] = np.rint(projected.reshape(-1, 2)).astype(np.int64)
    depth_all[source_indices] = depth
    valid_source[source_indices] = True
    return valid_source, uv, depth_all


def points_mask_consistency(
    points_lidar: np.ndarray,
    view_transforms: list[np.ndarray],
    masks: list[np.ndarray],
    camera: Camera,
    options: PointFilter,
) -> np.ndarray:
    if not view_transforms or not masks:
        return np.zeros(len(points_lidar), dtype=bool)

    visible = np.zeros(len(points_lidar), dtype=np.int32)
    hit = np.zeros(len(points_lidar), dtype=np.int32)
    for transform, mask in zip(view_transforms, masks):
        valid, uv, _ = project_points(points_lidar, transform, camera, options)
        h, w = mask.shape
        u, v = uv[:, 0], uv[:, 1]
        inside = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        visible += inside.astype(np.int32)
        hit += (inside & mask[v.clip(0, h - 1), u.clip(0, w - 1)]).astype(np.int32)

    enough_views = visible >= options.multiview_min_views
    ratio = np.zeros(len(points_lidar), dtype=np.float32)
    np.divide(hit, visible, out=ratio, where=visible > 0)
    return enough_views & (ratio >= options.multiview_ratio)
