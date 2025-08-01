# rcv_sampling.py

## Overview
Generates a recaptured control variation (RCV) dataset excluding given IDs.

## Key Components
- generate_rcv

### generate_rcv
Creates a reference DataFrame from the traiing set  renameing entries in the individual_id and trail colum `RCV`. This ensures that
subsequent projections use data unseen in the comparison itself. All other
columns are preserved so the output can feed directly into feature selection and
reduction steps.


## Assumptions and Limitations
Replaces identifiers with "RCV" for use as reference.
