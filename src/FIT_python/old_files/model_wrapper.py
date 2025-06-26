#!/usr/bin/env python3
"""Wrapper class to orchestrate model training across datasets using model_utils."""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

from FIT_python.config import NUMERIC_DIR, RESULTS_DATA_DIR, MODELS, METRICS, GLOBAL_RANDOM_SEED, DEBUG_MODE
from FIT_python.model_utils import train_and_evaluate
from FIT_python.hyperparameter_wrapper import HyperparameterWrapper

class ModelWrapper:
    def __init__(self):
        self.models = MODELS
        self.metrics = METRICS
        self.tuner = HyperparameterWrapper()
        self.seed = GLOBAL_RANDOM_SEED
        self.output_dir = RESULTS_DATA_DIR
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def run_all(self) -> int:
        try:
            for ds_folder in NUMERIC_DIR.iterdir():
                if not ds_folder.is_dir():
                    continue
                name = ds_folder.name
                logging.info("Processing dataset: %s", name)
                results = train_and_evaluate(
                    ds_folder,
                    self.models,
                    self.metrics,
                    self.tuner,
                    self.seed,
                    DEBUG_MODE,
                )
                # save or handle results per dataset...
            logging.info("All datasets processed.")
        except Exception as e:
            logging.error("Error during model training: %s", e)
            if DEBUG_MODE:
                raise
            return 1
        return 0

def main():
    wrapper = ModelWrapper()
    sys.exit(wrapper.run_all())

if __name__ == "__main__":
    main()
