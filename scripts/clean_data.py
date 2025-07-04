#!/usr/bin/env python3
"""Simple cleaning step copying raw CSV files to Parquet."""
from FIT_python.data_import_wrapper import DataImportWrapper


def main() -> int:
    wrapper = DataImportWrapper()
    return wrapper.clean_all()


if __name__ == "__main__":
    import sys
    sys.exit(main())
