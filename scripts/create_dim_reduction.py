#!/usr/bin/env python3
"""Apply PCA on feature selected datasets."""

from FIT_python.old_files.dim_reduction_wrapper import DimReducer


def main() -> int:
    wrapper = DimReducer()
    return wrapper.reduce_all()


if __name__ == "__main__":
    import sys
    sys.exit(main())
