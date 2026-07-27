#!/usr/bin/env python3
"""Create reviewable EEE records and Hugging Face eval-result previews."""

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_export.exporter import main  # noqa: E402


if __name__ == "__main__":
    main()
