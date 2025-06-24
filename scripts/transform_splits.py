#!/usr/bin/env python3
"""Execute numeric transformation for all processed splits."""
# Uses DEBUG_MODE from config to fail-fast on missing files

import sys
from FIT_python.transform_wrapper import TransformWrapper
from FIT_python.config import DEBUG_MODE


def main() -> int:
    wrapper = TransformWrapper()
    try:
        wrapper.transform_all()
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

