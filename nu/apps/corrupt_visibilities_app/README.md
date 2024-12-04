# Corrupt Visibilities App

This Stimela app corrupts model visibilities in a Measurement Set with random phase-only gains and writes the result to the `DATA` column.

## Usage

```bash

$ python3 run.py ms=/path/to/measurement_set.ms seed=42 noise_std=0.4

```

### Parameters

- `ms` (File, required): Path to the Measurement Set.
- `seed` (int, optional): Random seed for reproducibility (default: 42).
- `noise_std` (float, optional): Corrupts the noise with independent and identically distributed noise within the set standard deviation (default: 0.0).
