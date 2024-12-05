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
schema_path = os.path.join(os.path.dirname(__file__), "../cabs/solve.yaml")
schemas = OmegaConf.load(schema_path)

@click.command("solve")
@clickify_parameters(schemas.cabs.get("solve"))
def main(**kwargs):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Calibrate: ")

    # Extract parameters
    ms_path = kwargs.get("ms")

    # Check if MS exists
    if not os.path.exists(ms_path):
        logger.error(f"Measurement Set not found at {ms_path}")
        sys.exit(1)

    # Open the Measurement Set
    logger.info(f"Opening Measurement Set: {ms_path}")
    xds_list = xds_from_ms(ms_path, columns=["DATA", "MODEL_DATA", "ANTENNA1", "ANTENNA2"])

    if not xds_list:
        logger.error("Failed to read Measurement Set.")
        sys.exit(1)

    processed_xds = []
    for xds in xds_list:
        # Ensure MODEL_DATA exists
        if not hasattr(xds, "MODEL_DATA"):
            logger.error("MODEL_DATA column not found in the Measurement Set.")
            sys.exit(1)

        # Ensure DATA exists
        if "DATA" not in xds.data_vars:
            logger.error("DATA column not found in the Measurement Set.")
            sys.exit(1)

        data = xds.DATA.data  # Dask array
        model_data = xds.MODEL_DATA.data  # Dask array

        # Get antenna indices
        antenna1 = xds.ANTENNA1.data
        antenna2 = xds.ANTENNA2.data

        # Compute the number of antennas
        max_antenna = int(max(antenna1.max().compute(), antenna2.max().compute()))
        n_antennas = max_antenna + 1

        logger.info(f"Number of antennas: {n_antennas}")

        # Convert antenna indices to NumPy arrays for indexing
        antenna1_np = antenna1.compute()
        antenna2_np = antenna2.compute()

        nrows, nchannels, npols = data.shape
        logger.info(f"Number of rows: {nrows}")
        logger.info(f"Number of channels: {nchannels}")
        logger.info(f"Number of polarizations: {npols}")

        # Compute per-baseline phase differences
        logger.info("Computing per-baseline phase differences.")
        model_data_conj = da.conj(model_data)
        ratio = data * model_data_conj
        # Handle zeros in model data to avoid NaNs
        model_abs = da.abs(model_data)
        zero_mask = model_abs == 0.0
        ratio = da.where(zero_mask, 1.0 + 0j, ratio)
        delta_phi_ij = da.angle(ratio)

        # Initialize phi_i array
        phi = np.zeros((n_antennas, nchannels, npols))

        # Solve for phi_i per channel and polarization
        logger.info("Solving for per-antenna phases.")

        for chan in range(nchannels):
            for pol in range(npols):
                # Collect delta_phi_ij for all baselines at this channel and pol
                delta_phi = delta_phi_ij[:, chan, pol].compute()

                # Build the design matrix A
                nrows = len(delta_phi)
                A = np.zeros((nrows, n_antennas))
                A[np.arange(nrows), antenna1_np] = 1.0
                A[np.arange(nrows), antenna2_np] = -1.0

                # Fix the phase of the reference antenna (antenna 0)
                A_reduced = A[:, 1:]
                delta_phi_reduced = delta_phi - A[:, 0] * 0.0  # phi_0 = 0

                # Solve for phi[1:] using least squares
                phi_reduced, residuals, rank, s = np.linalg.lstsq(A_reduced, delta_phi_reduced, rcond=None)
                phi_i = np.zeros(n_antennas)
                phi_i[0] = 0.0  # Reference antenna phase set to zero
                phi_i[1:] = phi_reduced

                # Store phi_i for this channel and polarization
                phi[:, chan, pol] = phi_i

        # Compute gains G_i = exp(1j * phi_i)
        gains = np.exp(1j * phi)

        # Apply gains to data
        logger.info("Applying gains to data.")

        # Map antenna indices to gains
        gains_antenna1 = gains[antenna1_np]
        gains_antenna2 = np.conj(gains[antenna2_np])

        # Convert gains to Dask arrays with appropriate chunks
        gains_antenna1_da = da.from_array(gains_antenna1, chunks=(data.chunks[0], data.chunks[1], data.chunks[2]))
        gains_antenna2_da = da.from_array(gains_antenna2, chunks=(data.chunks[0], data.chunks[1], data.chunks[2]))

        corrected_data = data / (gains_antenna1_da * gains_antenna2_da)

        # Update the dataset with corrected data
        xds = xds.assign(CORRECTED_DATA=(("row", "chan", "corr"), corrected_data))
        processed_xds.append(xds)

    # Write the updated datasets back to the MS
    logger.info(f"Writing corrected data to CORRECTED_DATA column.")
    da.compute(xds_to_table(processed_xds, ms_path, columns=["CORRECTED_DATA"]))
    logger.info("Phase-only calibration completed successfully.")
