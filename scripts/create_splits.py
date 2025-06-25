#!/usr/bin/env python3
"""Create train/test splits for all raw datasets."""

from FIT_python.splits_wrapper import SplitsWrapper


def main() -> int:
    wrapper = SplitsWrapper()
    return wrapper.split_all()


if __name__ == "__main__":
    import sys
    sys.exit(main())
