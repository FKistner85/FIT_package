#!/usr/bin/env python3
"""Transform all split datasets to NumPy format."""

from FIT_python.transform_wrapper import TransformWrapper


def main() -> int:
    wrapper = TransformWrapper()
    return wrapper.transform_all()


if __name__ == "__main__":
    import sys
    sys.exit(main())
