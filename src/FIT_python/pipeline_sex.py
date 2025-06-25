#!/usr/bin/env python3
"""Run sklearn pipeline specifically for the 'sex' target."""

import sys
import logging
from FIT_python.pipeline_wrapper import PipelineWrapper
from FIT_python.config import DEBUG_MODE

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    target = "sex"
    logging.info("Starting pipeline for target: %s", target)
    wrapper = PipelineWrapper(target)
    try:
        results = wrapper.run_pipeline(filter_na=True)
        if results:
            import pandas as pd
            df = pd.DataFrame(results)[["dataset", "model", "cv_score", "test_score"]]
            logging.info("\n%s", df)
    except Exception as e:
        logging.error("Pipeline failed for target %s: %s", target, e)
        if DEBUG_MODE:
            raise
        return 1
    logging.info("Pipeline completed successfully for target: %s", target)
    return 0

if __name__ == "__main__":
    sys.exit(main())
