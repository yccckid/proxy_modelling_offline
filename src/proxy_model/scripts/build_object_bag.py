#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    ObjectBagPipeline(load_config(args.config)).run()


if __name__ == "__main__":
    main()
