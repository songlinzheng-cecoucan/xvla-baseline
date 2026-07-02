#!/usr/bin/env python3
"""CLI wrapper for `xvla_baseline.training.train_smolvla_lora`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xvla_baseline.training.train_smolvla_lora import main


if __name__ == "__main__":
    main()

