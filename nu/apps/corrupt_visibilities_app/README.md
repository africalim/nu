# Corrupt Visibilities App

This Stimela app corrupts model visibilities in a Measurement Set with random phase-only gains and writes the result to the `DATA` column.

## Usage

### Parameters

- `ms` (File, required): Path to the Measurement Set.
- `seed` (int, optional): Random seed for reproducibility (default: 42).
