#!/usr/bin/env python3
"""Simple feature selection on scaled datasets."""

from FIT_python.old_files.feature_wrapper import FeatureSelector


def main() -> int:
    wrapper = FeatureSelector()
    return wrapper.select_all()


if __name__ == "__main__":
    import sys
    sys.exit(main())
