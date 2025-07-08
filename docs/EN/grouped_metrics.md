# grouped_metrics.py

## Overview
Functions computing per-individual accuracy statistics.

## Key Components
- individual_accuracies
- individual_majority_stats

### individual_accuracies
Calculates accuracy separately for each individual and returns the mean values
for female, male and overall balanced accuracy. Supply arrays of true labels,
predicted labels and matching individual IDs. Results can reveal whether a model
performs unevenly across subjects but may be noisy if some individuals have few
samples.

### individual_majority_stats
Reports how many individuals are predicted correctly more than half of the time
versus those that are not. This highlights systematic mistakes on certain
subjects. Small groups can skew the percentage and the function does not convey
how wrong predictions are distributed within each individual.

## References
- https://pandas.pydata.org/docs/

## Assumptions and Limitations
Group labels are provided alongside predictions.
