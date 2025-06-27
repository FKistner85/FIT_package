#!/usr/bin/env python3
"""Scale all train/test splits using configured scalers."""

from FIT_python.old_files.scaler_wrapper import ScalerWrapper


def main() -> int:
    wrapper = ScalerWrapper()
    return wrapper.scale_all()


if __name__ == "__main__":
    import sys
    sys.exit(main())
