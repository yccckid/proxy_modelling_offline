#!/usr/bin/env python3
"""Put a GS-SDF 2D-Gaussian PLY into an object-centred PCA coordinate frame.

The exported GS-SDF PLY stores quaternion fields in W, X, Y, Z order.  This
script computes an object frame from the Gaussian centres, then applies the
same rigid world-to-object transform to every centre and Gaussian orientation.
Scales, opacity, and SH coefficients are deliberately copied unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_vertex_ply(path: Path) -> tuple[bytes, np.ndarray, bytes]:
    """Read a binary-little-endian PLY with vertex data as its first element.

    GS-SDF's ``gs.ply`` has only this vertex element.  Any bytes after the
    vertex records are retained verbatim so the reader is safe for compatible
    files containing trailing elements as well.
    """
    with path.open("rb") as file:
        header_lines: list[bytes] = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError("PLY header ended before 'end_header'.")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        payload = file.read()

    if not header_lines or header_lines[0].strip() != b"ply":
        raise ValueError("Input is not a PLY file.")
    if not any(line.strip() == b"format binary_little_endian 1.0" for line in header_lines):
        raise ValueError("Only 'binary_little_endian 1.0' PLY files are supported.")

    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    current: tuple[str, int, list[tuple[str, str]]] | None = None
    for raw in header_lines:
        tokens = raw.decode("ascii").strip().split()
        if not tokens:
            continue
        if tokens[0] == "element":
            if len(tokens) != 3:
                raise ValueError(f"Invalid element declaration: {raw!r}")
            current = (tokens[1], int(tokens[2]), [])
            elements.append(current)
        elif tokens[0] == "property" and current is not None:
            if len(tokens) != 3 or tokens[1] == "list":
                raise ValueError("Only scalar PLY properties are supported.")
            current[2].append((tokens[1], tokens[2]))

    if not elements or elements[0][0] != "vertex":
        raise ValueError("The first PLY element must be 'vertex'.")
    _, count, properties = elements[0]
    if not properties:
        raise ValueError("PLY vertex element has no properties.")
    try:
        dtype = np.dtype([(name, "<" + PLY_DTYPES[type_name]) for type_name, name in properties])
    except KeyError as error:
        raise ValueError(f"Unsupported PLY scalar type: {error.args[0]}") from error
    vertex_bytes = count * dtype.itemsize
    if len(payload) < vertex_bytes:
        raise ValueError("PLY payload is shorter than its vertex declaration.")
    vertices = np.frombuffer(payload[:vertex_bytes], dtype=dtype, count=count).copy()
    return b"".join(header_lines), vertices, payload[vertex_bytes:]


def write_vertex_ply(path: Path, header: bytes, vertices: np.ndarray, trailing: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(header)
        file.write(vertices.tobytes())
        file.write(trailing)


def normalize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1)
    valid = norms > 1e-12
    result[valid] /= norms[valid, None]
    # A zero quaternion cannot define an orientation.  Treat it as identity
    # instead of producing NaNs in the exported PLY.
    result[~valid] = np.array([1.0, 0.0, 0.0, 0.0])
    return result


def quaternions_to_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    q = normalize_quaternions_wxyz(quaternions)
    w, x, y, z = q.T
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def matrices_to_quaternions_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to WXYZ quaternions, vectorised."""
    matrix = np.asarray(matrices, dtype=np.float64)
    q = np.empty((len(matrix), 4), dtype=np.float64)
    m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
    trace = m00 + m11 + m22

    positive = trace > 0.0
    scale = np.empty(len(matrix), dtype=np.float64)
    scale[positive] = 2.0 * np.sqrt(np.maximum(trace[positive] + 1.0, 0.0))
    q[positive, 0] = 0.25 * scale[positive]
    q[positive, 1] = (m21[positive] - m12[positive]) / scale[positive]
    q[positive, 2] = (m02[positive] - m20[positive]) / scale[positive]
    q[positive, 3] = (m10[positive] - m01[positive]) / scale[positive]

    x_largest = ~positive & (m00 > m11) & (m00 > m22)
    scale[x_largest] = 2.0 * np.sqrt(np.maximum(1.0 + m00[x_largest] - m11[x_largest] - m22[x_largest], 0.0))
    q[x_largest, 0] = (m21[x_largest] - m12[x_largest]) / scale[x_largest]
    q[x_largest, 1] = 0.25 * scale[x_largest]
    q[x_largest, 2] = (m01[x_largest] + m10[x_largest]) / scale[x_largest]
    q[x_largest, 3] = (m02[x_largest] + m20[x_largest]) / scale[x_largest]

    y_largest = ~positive & ~x_largest & (m11 > m22)
    scale[y_largest] = 2.0 * np.sqrt(np.maximum(1.0 + m11[y_largest] - m00[y_largest] - m22[y_largest], 0.0))
    q[y_largest, 0] = (m02[y_largest] - m20[y_largest]) / scale[y_largest]
    q[y_largest, 1] = (m01[y_largest] + m10[y_largest]) / scale[y_largest]
    q[y_largest, 2] = 0.25 * scale[y_largest]
    q[y_largest, 3] = (m12[y_largest] + m21[y_largest]) / scale[y_largest]

    z_largest = ~positive & ~x_largest & ~y_largest
    scale[z_largest] = 2.0 * np.sqrt(np.maximum(1.0 + m22[z_largest] - m00[z_largest] - m11[z_largest], 0.0))
    q[z_largest, 0] = (m10[z_largest] - m01[z_largest]) / scale[z_largest]
    q[z_largest, 1] = (m02[z_largest] + m20[z_largest]) / scale[z_largest]
    q[z_largest, 2] = (m12[z_largest] + m21[z_largest]) / scale[z_largest]
    q[z_largest, 3] = 0.25 * scale[z_largest]
    return normalize_quaternions_wxyz(q)


def pca_object_frame(points: np.ndarray, origin_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return origin_W, R_WO, and decreasing PCA eigenvalues.

    ``R_WO`` has object axes as columns expressed in world coordinates.  Thus
    ``R_OW = R_WO.T`` maps world-coordinate vectors to object coordinates.
    """
    pca_mean = points.mean(axis=0)
    if origin_mode == "centroid":
        origin = pca_mean
    elif origin_mode == "bbox":
        origin = 0.5 * (points.min(axis=0) + points.max(axis=0))
    else:  # argparse makes this unreachable; preserve a useful error for APIs.
        raise ValueError(f"Unknown origin mode: {origin_mode}")

    covariance = (points - pca_mean).T @ (points - pca_mean) / len(points)
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = axes[:, order]

    # Make the otherwise arbitrary PCA signs reproducible: each axis first
    # points toward its largest-magnitude world-coordinate component.  Then
    # fix the third one if necessary to retain a right-handed object frame.
    for column in range(3):
        dominant = np.argmax(np.abs(axes[:, column]))
        if axes[dominant, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    return origin, axes, eigenvalues


def load_pca_selection(path: Path, total: int) -> np.ndarray:
    selection = np.load(path, allow_pickle=False)
    if selection.ndim != 1:
        raise ValueError("--pca-mask must be a one-dimensional .npy array.")
    if selection.dtype == np.bool_:
        if len(selection) != total:
            raise ValueError("Boolean --pca-mask length must equal the number of Gaussians.")
        return selection
    if not np.issubdtype(selection.dtype, np.integer):
        raise ValueError("--pca-mask must contain booleans or integer Gaussian indices.")
    mask = np.zeros(total, dtype=bool)
    if np.any(selection < 0) or np.any(selection >= total):
        raise ValueError("--pca-mask contains an out-of-range Gaussian index.")
    mask[selection] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input GS-SDF gs.ply (binary_little_endian).")
    parser.add_argument("--output", required=True, type=Path, help="Output PLY in the PCA object frame.")
    parser.add_argument("--transform-output", type=Path, help="JSON file for the world/object transform (default: next to output).")
    parser.add_argument("--origin", choices=("centroid", "bbox"), default="centroid",
                        help="Object-frame origin: mean Gaussian centre (default) or bounding-box centre.")
    parser.add_argument("--pca-mask", type=Path,
                        help="Optional .npy bool mask or integer indices used only to estimate the object PCA frame.")
    args = parser.parse_args()

    header, vertices, trailing = read_vertex_ply(args.input)
    required = ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3")
    missing = [name for name in required if name not in vertices.dtype.names]
    if missing:
        raise ValueError(f"Input is not a GS-SDF Gaussian PLY; missing properties: {', '.join(missing)}")

    centres_world = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)
    finite = np.isfinite(centres_world).all(axis=1)
    selection = finite.copy()
    if args.pca_mask is not None:
        selection &= load_pca_selection(args.pca_mask, len(vertices))
    if selection.sum() < 3:
        raise ValueError("At least three finite selected Gaussian centres are required for PCA.")

    origin_world, rotation_world_from_object, eigenvalues = pca_object_frame(centres_world[selection], args.origin)
    rotation_object_from_world = rotation_world_from_object.T
    centres_object = (centres_world - origin_world) @ rotation_world_from_object
    centres_object[~finite] = centres_world[~finite]
    vertices["x"], vertices["y"], vertices["z"] = (centres_object[:, axis] for axis in range(3))

    old_quaternions = np.column_stack([vertices[f"rot_{axis}"] for axis in range(4)])
    old_rotations = quaternions_to_matrices_wxyz(old_quaternions)
    new_rotations = rotation_object_from_world[None, :, :] @ old_rotations
    new_quaternions = matrices_to_quaternions_wxyz(new_rotations)
    for axis in range(4):
        vertices[f"rot_{axis}"] = new_quaternions[:, axis]

    write_vertex_ply(args.output, header, vertices, trailing)
    transform_path = args.transform_output or args.output.with_suffix(".transform.json")
    world_to_object = np.eye(4)
    world_to_object[:3, :3] = rotation_object_from_world
    world_to_object[:3, 3] = -rotation_object_from_world @ origin_world
    object_to_world = np.eye(4)
    object_to_world[:3, :3] = rotation_world_from_object
    object_to_world[:3, 3] = origin_world
    metadata = {
        "description": "Object frame obtained from GS centre PCA; matrices use column vectors.",
        "input_ply": str(args.input),
        "output_ply": str(args.output),
        "origin_mode": args.origin,
        "origin_world": origin_world.tolist(),
        "pca_axes_world_columns": rotation_world_from_object.tolist(),
        "pca_eigenvalues_descending": eigenvalues.tolist(),
        "world_to_object": world_to_object.tolist(),
        "object_to_world": object_to_world.tolist(),
        "quaternion_order": "wxyz",
        "pca_gaussian_count": int(selection.sum()),
        "total_gaussian_count": int(len(vertices)),
        "note": "Only centres and rotations are transformed. Scales, opacity, and SH coefficients are unchanged.",
    }
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    transform_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote object-frame PLY: {args.output}")
    print(f"Wrote transform JSON: {transform_path}")
    print(f"Origin in world frame: {origin_world.tolist()}")
    print(f"PCA eigenvalues: {eigenvalues.tolist()}")


if __name__ == "__main__":
    main()
