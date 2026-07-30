from __future__ import annotations

import cv2
import numpy as np

from .config import Camera, PointFilter


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Keep one finite point per voxel without adding a point-cloud dependency."""
    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        return points.reshape(0, 3)
    cells = np.floor(points / voxel_size_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to an N x 3 point array."""
    points = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def points_projected_in_mask(
    points_lidar: np.ndarray,
    transform_camera_lidar: np.ndarray,
    mask: np.ndarray,
    camera: Camera,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return direct 2-D mask labels and projected pixels for every lidar point.

    Labels are deliberately based only on whether a valid projected point lands
    inside the mask.  Later multiview, depth-tolerance, and clustering filters
    do not alter these exported semantic labels.
    """
    points_camera = transform_points(points_lidar, transform_camera_lidar)
    projected, _, valid = _project_camera_points(
        points_camera, camera, min_depth_m, max_depth_m
    )
    labels = np.zeros(len(points_lidar), dtype=bool)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return labels, projected
    pixels = np.rint(projected[valid_indices]).astype(np.int32)
    h, w = mask.shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    inside_indices = valid_indices[inside]
    inside_pixels = pixels[inside]
    labels[inside_indices] = mask[inside_pixels[:, 1], inside_pixels[:, 0]]
    return labels, projected


def geometric_forward_flow(
    points_lidar: np.ndarray,
    previous_camera_lidar: np.ndarray,
    current_camera_lidar: np.ndarray,
    source_mask: np.ndarray,
    camera: Camera,
    min_depth_m: float,
    max_depth_m: float,
    min_seed_points: int,
) -> np.ndarray | None:
    """Build dense forward flow inside a mask from projected static 3-D points.

    The input points are observed in the lidar frame at the source time.  They
    are projected into both cameras, so this works for arbitrary static object
    geometry rather than assuming a planar target.
    """
    source_mask = np.asarray(source_mask, dtype=bool)
    h, w = source_mask.shape
    points = np.asarray(points_lidar, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if finite.sum() < min_seed_points:
        return None
    points = points[finite]
    previous_points = transform_points(points, previous_camera_lidar)
    current_points = transform_points(points, current_camera_lidar)
    previous_uv, previous_depth, previous_valid = _project_camera_points(
        previous_points, camera, min_depth_m, max_depth_m
    )
    current_uv, current_depth, current_valid = _project_camera_points(
        current_points, camera, min_depth_m, max_depth_m
    )
    valid = previous_valid & current_valid
    if valid.sum() < min_seed_points:
        return None

    previous_uv = previous_uv[valid]
    current_uv = current_uv[valid]
    previous_depth = previous_depth[valid]
    current_depth = current_depth[valid]
    previous_px = np.rint(previous_uv).astype(np.int32)
    current_px = np.rint(current_uv).astype(np.int32)
    inside = (
        (previous_px[:, 0] >= 0)
        & (previous_px[:, 0] < w)
        & (previous_px[:, 1] >= 0)
        & (previous_px[:, 1] < h)
        & (current_px[:, 0] >= 0)
        & (current_px[:, 0] < w)
        & (current_px[:, 1] >= 0)
        & (current_px[:, 1] < h)
    )
    if not inside.any():
        return None
    previous_uv = previous_uv[inside]
    current_uv = current_uv[inside]
    previous_depth = previous_depth[inside]
    current_depth = current_depth[inside]
    previous_px = previous_px[inside]
    current_px = current_px[inside]
    in_mask = source_mask[previous_px[:, 1], previous_px[:, 0]]
    if in_mask.sum() < min_seed_points:
        return None
    previous_uv = previous_uv[in_mask]
    current_uv = current_uv[in_mask]
    previous_depth = previous_depth[in_mask]
    current_depth = current_depth[in_mask]
    previous_px = previous_px[in_mask]
    current_px = current_px[in_mask]

    # Retain the nearest visible lidar point for each source pixel.
    cell = previous_px[:, 1] * w + previous_px[:, 0]
    order = np.lexsort((previous_depth, cell))
    ordered_cell = cell[order]
    keep = np.empty(len(order), dtype=bool)
    keep[0] = True
    keep[1:] = ordered_cell[1:] != ordered_cell[:-1]
    selected = order[keep]
    if len(selected) < min_seed_points:
        return None

    seed_flow = np.zeros((h, w, 2), dtype=np.float32)
    seed_valid = np.zeros((h, w), dtype=bool)
    u, v = previous_px[selected, 0], previous_px[selected, 1]
    seed_flow[v, u] = (current_uv[selected] - previous_uv[selected]).astype(np.float32)
    seed_valid[v, u] = True
    return _fill_flow_inside_mask(seed_flow, seed_valid, source_mask)


def forward_warp_mask_with_flow(
    source_mask: np.ndarray, forward_flow: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-warp a mask and return dense current-to-source flow for display."""
    source_mask = np.asarray(source_mask, dtype=bool)
    h, w = source_mask.shape
    y, x = np.nonzero(source_mask)
    destination_x = np.rint(x + forward_flow[y, x, 0]).astype(np.int32)
    destination_y = np.rint(y + forward_flow[y, x, 1]).astype(np.int32)
    inside = (
        (destination_x >= 0)
        & (destination_x < w)
        & (destination_y >= 0)
        & (destination_y < h)
    )
    if not inside.any():
        return np.zeros_like(source_mask), np.zeros((h, w, 2), dtype=np.float32)

    source_x, source_y = x[inside], y[inside]
    destination_x, destination_y = destination_x[inside], destination_y[inside]
    output_mask = np.zeros_like(source_mask)
    output_mask[destination_y, destination_x] = True
    output_mask = cv2.morphologyEx(
        output_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)

    backward_seed = np.zeros((h, w, 2), dtype=np.float32)
    backward_valid = np.zeros((h, w), dtype=bool)
    backward_seed[destination_y, destination_x] = -forward_flow[source_y, source_x]
    backward_valid[destination_y, destination_x] = True
    backward_flow = _fill_flow_inside_mask(backward_seed, backward_valid, output_mask)
    return output_mask, backward_flow


def _project_camera_points(
    points_camera: np.ndarray,
    camera: Camera,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (depth >= min_depth_m) & (
        depth <= max_depth_m
    )
    projected = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    if not valid.any():
        return projected, depth, valid
    distortion = np.asarray(camera.distortion, dtype=np.float64)
    if camera.distortion_model == "fisheye":
        uv, _ = cv2.fisheye.projectPoints(
            points_camera[valid].reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            camera.matrix,
            distortion[:4],
        )
    else:
        uv, _ = cv2.projectPoints(
            points_camera[valid],
            np.zeros(3),
            np.zeros(3),
            camera.matrix,
            distortion,
        )
    projected[valid] = uv.reshape(-1, 2)
    return projected, depth, valid


def _fill_flow_inside_mask(
    seed_flow: np.ndarray, seed_valid: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor flow completion independently for every mask component."""
    h, w = mask.shape
    output = np.zeros((h, w, 2), dtype=np.float32)
    component_count, components = cv2.connectedComponents(mask.astype(np.uint8))
    for component in range(1, component_count):
        region = components == component
        seeds = seed_valid & region
        if not seeds.any():
            continue
        distance_input = np.ones((h, w), dtype=np.uint8)
        distance_input[seeds] = 0
        _, labels = cv2.distanceTransformWithLabels(
            distance_input,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        lookup = np.zeros((int(labels.max()) + 1, 2), dtype=np.float32)
        lookup[labels[seeds]] = seed_flow[seeds]
        output[region] = lookup[labels[region]]
    return output


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
