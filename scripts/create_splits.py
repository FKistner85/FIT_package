#!/usr/bin/env python3
"""Create train/test splits for all raw datasets."""

from FIT_python.pipeline.split_wrapper import SplitWrapper


def main() -> int:
    wrapper = SplitWrapper()
    return wrapper.split_all()


if __name__ == "__main__":
    import sys
    sys.exit(main())
