#!/usr/bin/env python3
"""CLI wrapper for `xvla_baseline.data.split_lerobot_episodes`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xvla_baseline.data.split_lerobot_episodes import main


if __name__ == "__main__":
    main()

