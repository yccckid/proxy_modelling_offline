#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from proxy_model.config import load_config
from proxy_model.pipeline import ObjectBagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an object-only ROS bag for GS-SDF."
    )
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument(
        "--masks-dir",
        default="",
        help="Optional COLMAP masks directory to populate from cached masks, e.g. data/scene/masks.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    ObjectBagPipeline(config).run()
    if args.masks_dir:
        masks_dir = Path(args.masks_dir).expanduser().resolve()
        masks_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted((config.output.cache_dir / "masks").glob("*.png")):
            shutil.copy2(src, masks_dir / src.name)
        print(f"Copied masks to {masks_dir}")


if __name__ == "__main__":
    main()
