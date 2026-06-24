from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass

from PIL import Image

from .config import Qwen


SYSTEM_PROMPT = """You identify the object requested by the user for image segmentation.
Return only strict JSON:
{"label":"short English noun phrase","mode":"all|one","center_point":[x,y]|null}
Use mode "one" when the user refers to one specific instance, otherwise "all".
For mode "one", center_point is normalized to [0,1] and must lie inside that object.
The label must describe appearance/category but must not contain spatial words."""


@dataclass(frozen=True)
class Target:
    label: str
    mode: str
    center_point: tuple[int, int] | None
    raw: str = ""


def resolve_target(image: Image.Image, prompt: str, config: Qwen) -> Target:
    if not config.enabled:
        if not config.label:
            raise ValueError("segmentation.qwen.label is required when Qwen is disabled")
        point = _configured_point(config.center_point, image.size)
        return Target(config.label, config.mode, point)

    api_key = os.getenv(config.api_key_env, "")
    if not api_key:
        raise ValueError(
            f"Environment variable {config.api_key_env} is empty. "
            "Set the Qwen API key or disable Qwen and configure label/mode."
        )
    from openai import OpenAI

    sent = _resize(image, config.image_max_side)
    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_s,
        max_retries=config.max_retries,
    )
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(sent)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=256,
    )
    raw = (response.choices[0].message.content or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Qwen did not return JSON: {raw}")
    data = json.loads(match.group(0))
    label = str(data["label"]).strip()
    mode = str(data.get("mode", "all")).lower()
    if mode not in {"all", "one"}:
        mode = "all"
    point = _configured_point(data.get("center_point"), image.size)
    if mode == "one" and point is None:
        raise ValueError(f"Qwen selected one instance without center_point: {raw}")
    return Target(label, mode, point, raw)


def _configured_point(
    value: tuple[float, float] | list[float] | None, size: tuple[int, int]
) -> tuple[int, int] | None:
    if value is None:
        return None
    width, height = size
    x, y = float(value[0]), float(value[1])
    if max(abs(x), abs(y)) <= 1.5:
        x, y = x * width, y * height
    return (
        int(min(width - 1, max(0, round(x)))),
        int(min(height - 1, max(0, round(y)))),
    )


def _resize(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0 or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC
    )


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=90)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
