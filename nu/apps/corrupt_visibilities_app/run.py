#!/usr/bin/env python3

import os
import sys
import numpy as np
from casacore.tables import table
import argparse
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Corrupt visibilities with random phase-only gains.')
    parser.add_argument('--ms', required=True, help='Path to the Measurement Set')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--noise-std', required=False, type=float, default=0.0, help='Standard deviation of i.i.d. noise to add to visibilities')
    return parser.parse_args()

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('CorruptVisibilities')

    # Parse command-line arguments
    args = parse_arguments()
    ms_path = args.ms
    seed = args.seed
    noise_std = args.noise_std

    # Check if MS exists
    if not os.path.exists(ms_path):
        logger.error(f"Measurement Set not found at {ms_path}")
        sys.exit(1)

    # Set random seed
    np.random.seed(seed)

    # Open the Measurement Set
    logger.info(f"Opening Measurement Set: {ms_path}")
    ms = table(ms_path, readonly=False)

    # Check for MODEL_DATA column
    if 'MODEL_DATA' not in ms.colnames():
        logger.error("MODEL_DATA column not found in the Measurement Set.")
        ms.close()
        sys.exit(1)

    # Read MODEL_DATA
    model_data = ms.getcol('MODEL_DATA')
    nrows, nchannels, npols = model_data.shape

    # Read antenna information
    antenna1 = ms.getcol('ANTENNA1')
    antenna2 = ms.getcol('ANTENNA2')
    n_antennas = max(antenna1.max(), antenna2.max()) + 1

    logger.info(f"Number of antennas: {n_antennas}")
    logger.info(f"Number of channels: {nchannels}")
    logger.info(f"Number of polarizations: {npols}")

    # Generate random phase-only gains for each antenna
    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, (n_antennas, nchannels, npols)))

    # Apply gains to visibilities
    logger.info("Applying random phase-only gains to visibilities.")
    gains_antenna1 = random_phases[antenna1]
    gains_antenna2 = random_phases[antenna2].conj()

    corrupted_data = model_data * gains_antenna1 * gains_antenna2

    # Add noise if noise_std > 0
    if noise_std > 0:
        logger.info(f"Adding i.i.d. noise with standard deviation {noise_std} to visibilities.")
        noise_real = np.random.normal(0, noise_std, corrupted_data.shape)
        noise_imag = np.random.normal(0, noise_std, corrupted_data.shape)
        noise = noise_real + 1j * noise_imag
        corrupted_data += noise
    # Ensure that the std is non-negative
    if noise_std < 0:
        logger.error("Noise standard deviation cannot be negative.")
        ms.close()
        sys.exit(1)

    # Write corrupted data to DATA column
    logger.info("Writing corrupted data to DATA column.")
    if 'DATA' not in ms.colnames():
        # Copy column description from MODEL_DATA
        desc = ms.getcoldesc('MODEL_DATA')
        ms.addcols(desc, 'DATA')

    ms.putcol('DATA', corrupted_data)

    # Close the Measurement Set
    ms.close()
    logger.info("Corruption process completed successfully.")

if __name__ == '__main__':
    main()
