# Coordinate Reprojection

The annotation GUI applies rotations and zoom operations to the loaded
image.  Every step is stored as a `QTransform` matrix so that the exact
sequence can be reproduced later.

## Stored parameters

* `rotation_points` – pairs of points defining each rotation step
* `scale_points` – user defined scale bars
* `transforms` – list of 3x3 transformation matrices in the order they
  were applied

These values are written to the annotation JSON together with the
landmark coordinates.

## Usage

The `TransformationPipeline` class in `FIT_python.utils.transformations`
can apply a sequence of transforms to images or coordinates and also
invert the sequence.  When existing annotations are loaded the stored
matrices are read and used to back‑project landmarks to their original
positions.
