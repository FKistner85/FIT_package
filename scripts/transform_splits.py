#!/usr/bin/env python3
"""Execute numeric transformation for all processed splits."""

import sys
from FIT_python.transform_wrapper import TransformWrapper


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

