# rcv_sampling.py

## Overview
Generates a recaptured control variation (RCV) dataset excluding given indices.

## Key Components
- generate_rcv

### generate_rcv
Creates a reference DataFrame by removing the indices used in a current
comparison and labelling the remaining entries as `RCV`. This ensures that
subsequent projections use data unseen in the comparison itself. All other
columns are preserved so the output can feed directly into feature selection and
reduction steps.

## References
- https://pandas.pydata.org/docs/

## Assumptions and Limitations
Replaces identifiers with "RCV" for use as reference.
