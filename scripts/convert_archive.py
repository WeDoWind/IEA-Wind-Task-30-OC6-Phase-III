#!/usr/bin/env python3
"""Entry point for converting the source archives into the archive layer. See `--help`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oc6_phase3.build.cli import convert_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(convert_main())
