#!/usr/bin/env python3

import os
import sys
import numpy as np
import dask.array as da
from daskms import xds_from_ms, xds_to_table
import click
from scabha.schema_utils import clickify_parameters
from omegaconf import OmegaConf
import logging
from pathlib import Path

# Load the schema for parameters
schema_path = Path(__file__).parent.parent / "recipes/corrupt.yaml"
schemas = OmegaConf.load(schema_path)

@click.command("corrupt_visibilities")
@clickify_parameters(schemas.cabs.get("corrupt_visibilities"))
def main(**kwargs):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CorruptVisibilities")

    # Extract parameters
    ms_path = kwargs.get("ms")
    seed = kwargs.get("seed", 42)
    noise_std = kwargs.get("noise_std", 0.0)

    # Check if MS exists
    if not os.path.exists(ms_path):
        logger.error(f"Measurement Set not found at {ms_path}")
        sys.exit(1)

    # Set random seed
    np.random.seed(seed)

    # Open the Measurement Set
    logger.info(f"Opening Measurement Set: {ms_path}")
    xds_list = xds_from_ms(ms_path, columns=["MODEL_DATA", "ANTENNA1", "ANTENNA2"], chunks={"row": 100000})

    if not xds_list:
        logger.error("Failed to read Measurement Set.")
        sys.exit(1)

    processed_xds = []
    for xds in xds_list:
        # Ensure MODEL_DATA exists
        if not hasattr(xds, "MODEL_DATA"):
            logger.error("MODEL_DATA column not found in the Measurement Set.")
            sys.exit(1)

        model_data = xds.MODEL_DATA.data  # Dask array
        nrows, nchannels, npols = model_data.shape

        antenna1 = xds.ANTENNA1.data
        antenna2 = xds.ANTENNA2.data

        # Compute the number of antennas
        max_antenna = int(max(antenna1.max().compute(), antenna2.max().compute()))
        n_antennas = max_antenna + 1

        logger.info(f"Number of antennas: {n_antennas}")
        logger.info(f"Number of channels: {nchannels}")
        logger.info(f"Number of polarizations: {npols}")

        # Generate random phase-only gains for each antenna
        random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(n_antennas, nchannels, npols)))

        # Convert antenna indices to NumPy arrays for indexing
        antenna1_np = antenna1.compute()
        antenna2_np = antenna2.compute()

        # Apply gains to visibilities
        logger.info("Applying random phase-only gains to visibilities.")

        gains_antenna1 = random_phases[antenna1_np]
        gains_antenna2 = np.conj(random_phases[antenna2_np])

        # Convert gains to Dask arrays with appropriate chunks
        gains_antenna1_da = da.from_array(gains_antenna1, chunks=model_data.chunks)
        gains_antenna2_da = da.from_array(gains_antenna2, chunks=model_data.chunks)

        corrupted_data = model_data * gains_antenna1_da * gains_antenna2_da

        # Add noise if noise_std > 0
        if noise_std > 0:
            logger.info(f"Adding i.i.d. noise with standard deviation {noise_std} to visibilities.")
            noise_shape = corrupted_data.shape
            noise_real = da.random.normal(0, noise_std, size=noise_shape, chunks=corrupted_data.chunks)
            noise_imag = da.random.normal(0, noise_std, size=noise_shape, chunks=corrupted_data.chunks)
            noise = noise_real + 1j * noise_imag
            corrupted_data += noise

        # Update the dataset with corrupted DATA
        xds = xds.assign(DATA=(("row", "chan", "corr"), corrupted_data))
        processed_xds.append(xds)

    # Write the updated datasets back to the MS
    logger.info("Writing corrupted data to DATA column.")
    xds_to_table(processed_xds, ms_path, columns=["DATA"])

    logger.info("Corruption process completed successfully.")

