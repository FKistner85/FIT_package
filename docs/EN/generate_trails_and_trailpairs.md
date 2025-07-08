# generate_trails_and_trailpairs.py

## Overview
Creates trail segments and all pairwise comparisons with metadata.

## Key Components
- generate_pairwise_comparisons_from_df

### generate_pairwise_comparisons_from_df
Takes a raw footprint DataFrame and creates all valid trail combinations for the
given individuals. Trails are sampled in non‑overlapping chunks and compared
both within and across individuals while keeping track of metadata such as
`same_individual` and `same_sex`. The function performs stratified
cross‑validation to balance trail sizes. Large datasets may generate many pairs
which increases memory usage and runtime.

## References
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

## Assumptions and Limitations
Uses random sampling of chunks; results include summary statistics.
