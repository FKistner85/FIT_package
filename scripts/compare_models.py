#!/usr/bin/env python3
"""Very small model comparison using scaled splits."""

from FIT_python.model_comparator import LegacyModelComparator


def main() -> int:
    comparator = LegacyModelComparator()
    return comparator.compare_all()


if __name__ == "__main__":
    import sys
    sys.exit(main())
