from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class TimedMessage:
    stamp_ns: int
    bag_time_ns: int
    message: object


def message_stamp_ns(message: object, bag_time: object) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and stamp.to_nsec() > 0:
        return int(stamp.to_nsec())
    return int(bag_time.to_nsec())


def nearest(items: list[TimedMessage], stamp_ns: int, tolerance_s: float) -> TimedMessage | None:
    if not items:
        return None
    import bisect

    stamps = [item.stamp_ns for item in items]
    pos = bisect.bisect_left(stamps, stamp_ns)
    candidates = items[max(0, pos - 1) : min(len(items), pos + 1)]
    result = min(candidates, key=lambda item: abs(item.stamp_ns - stamp_ns))
    return result if abs(result.stamp_ns - stamp_ns) <= tolerance_s * 1e9 else None


def decode_image(message: object) -> np.ndarray:
    msg_type = getattr(message, "_type", "")
    if msg_type == "sensor_msgs/CompressedImage":
        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    elif msg_type == "sensor_msgs/Image":
        from cv_bridge import CvBridge

        image = CvBridge().imgmsg_to_cv2(message, desired_encoding="bgr8")
    else:
        raise TypeError(f"Unsupported image message type: {msg_type}")
    if image is None:
        raise ValueError("Failed to decode image message")
    return image


def encode_masked_image(message: object, image_bgr: np.ndarray, quality: int) -> object:
    output = copy.deepcopy(message)
    msg_type = getattr(message, "_type", "")
    if msg_type == "sensor_msgs/CompressedImage":
        ok, encoded = cv2.imencode(
            ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            raise RuntimeError("Failed to encode masked JPEG")
        output.format = "bgr8; jpeg compressed bgr8"
        output.data = encoded.tobytes()
        return output
    if msg_type == "sensor_msgs/Image":
        from cv_bridge import CvBridge

        encoded = CvBridge().cv2_to_imgmsg(image_bgr, encoding="bgr8")
        encoded.header = copy.deepcopy(message.header)
        return encoded
    raise TypeError(f"Unsupported image message type: {msg_type}")


def encode_mask_image(reference: object, mask: np.ndarray) -> object:
    from cv_bridge import CvBridge

    image = (mask.astype(np.uint8) * 255)
    message = CvBridge().cv2_to_imgmsg(image, encoding="mono8")
    message.header = copy.deepcopy(reference.header)
    return message


def xyz_view(message: object) -> np.ndarray:
    offsets = {field.name: field.offset for field in message.fields}
    missing = {"x", "y", "z"} - offsets.keys()
    if missing:
        raise ValueError(f"PointCloud2 lacks fields: {sorted(missing)}")
    byteorder = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [f"{byteorder}f4"] * 3,
            "offsets": [offsets["x"], offsets["y"], offsets["z"]],
            "itemsize": message.point_step,
        }
    )
    count = int(message.width * message.height)
    values = np.frombuffer(message.data, dtype=dtype, count=count)
    return np.column_stack((values["x"], values["y"], values["z"])).astype(
        np.float64, copy=False
    )


def filter_pointcloud(message: object, keep: np.ndarray) -> object:
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    count = int(message.width * message.height)
    if keep.size != count:
        raise ValueError(f"Point mask has {keep.size} entries, expected {count}")
    records = np.frombuffer(message.data, dtype=np.uint8).reshape(count, message.point_step)
    selected = np.ascontiguousarray(records[keep])
    output = copy.deepcopy(message)
    output.height = 1
    output.width = int(selected.shape[0])
    output.row_step = output.width * output.point_step
    output.data = selected.tobytes()
    output.is_dense = bool(message.is_dense)
    return output


def write_frame(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(image)
    output[mask] = image[mask]
    return output


def mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(image)
    color[..., 1] = 255
    overlay[mask] = cv2.addWeighted(image[mask], 0.45, color[mask], 0.55, 0)
    return overlay
