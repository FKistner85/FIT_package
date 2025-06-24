#!/usr/bin/env python3
# scripts/scale_splits.py

"""Scale all train/test splits using configured scalers."""

from pathlib import Path
from FIT_python.config import SPLITS_DIR, PROCESSED_DIR
from FIT_python.scaler_wrapper import ScalerWrapper

def main():
    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Instantiate wrapper (use default scaler from config)
    scaler = ScalerWrapper()
    scaler.scale_all()
    print("All datasets scaled and saved in", PROCESSED_DIR)

if __name__ == "__main__":
    main()
