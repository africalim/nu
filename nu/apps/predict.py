#!/usr/bin/env python
import os
import click
import numpy as np

from omegaconf import OmegaConf
from scabha.schema_utils import clickify_parameters

import dask
import dask.array as da
from daskms import xds_from_table, xds_to_table

from africanus.rime import wsclean_predict
from africanus.coordinates import radec_to_lm

def predict_source(msname: str, ra: float = None, dec: float = None, flux: float = 1.0, column: str = "MODEL_DATA") -> None:
    """
    Populates a measurement set (MS) with visibilities for a point source.

    Args:
        msname (str): Path to the measurement set.
        ra (float, optional): Right Ascension of the source in degrees. Defaults to the phase center if not provided.
        dec (float, optional): Declination of the source in degrees. Defaults to the phase center if not provided.
        flux (float, optional): Flux of the source. Defaults to 1.0.
        column (str, optional): Column name where the visibilities will be stored. Defaults to "MODEL_DATA".

    Returns:
        None
    """
    # Read the measurement set and extract relevant information
    field_name = "::".join((msname, "FIELD"))
    spwname = "::".join((msname, "SPECTRAL_WINDOW"))
    xds = xds_from_table(msname)[0]
    uvw = xds['UVW'].values
    spwds = xds_from_table(spwname)[0]
    fieldds = xds_from_table(field_name)[0]
    frequency = spwds['CHAN_FREQ'].values.squeeze()
    phase_center = fieldds['PHASE_DIR'].values.squeeze()[0]
    gauss_shape = np.zeros((1, 3))  # Not relevant for point source

    # Handle default RA and DEC values
    if ra is None:
        ra = phase_center[0]
    if dec is None:
        dec = phase_center[1]
    # Convert RA and DEC to LM coordinates
    lm = radec_to_lm(np.array([[ra, dec]]))
    # Prepare input parameters for wsclean_predict
    source_type = 'POINT'
    coeffs = np.zeros((1, 0))  # No polynomial coefficients for point source
    log_poly = np.zeros(1, dtype=bool)
    ref_freq = frequency[0]

    # Use Dask to parallelize the visibility prediction
    uvws = da.from_array(uvw, chunks=(10000, 3))
    lms = da.from_array(lm)
    visibilities = wsclean_predict(
        uvws.compute(), lms.compute(),
        np.array([source_type]),
        np.array([flux], dtype=np.float64),
        np.array(coeffs, dtype=np.float64),
        np.array(log_poly, dtype=np.float64),
        np.array([ref_freq]),
        gauss_shape,
        frequency)
    dask_visibilities = da.from_array(visibilities,
            chunks=(10000, visibilities.shape[1], visibilities.shape[2]))
    # Update the measurement set with the predicted visibilities
    # Assign visibilities to MODEL_DATA array on the dataset
    xds = xds.assign(**{column: (("row", "chan", "corr"), dask_visibilities)})
    # Write to the table
    da.compute(xds_to_table(xds, msname, [column]))
    print(f"Successfully populated measurement set '{msname}' with column '{column}'.")
    return visibilities

schemas = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../cabs/predict.yml"))

@click.command('predict')
@clickify_parameters(schemas.cabs.get('predict'))
def main(**kwargs):
    """
    CLI entry point for the predict app.

    This app populates a measurement set (MS) with visibilities for a single point source.

    Args:
        **kwargs: Command-line arguments including:
            msname (str): Path to the measurement set.
            ra (float): Right Ascension of the source in degrees.
            dec (float): Declination of the source in degrees.
            flux (float): Flux of the source.
            column (str): Column name where the visibilities will be stored.

    Returns:
        None
    """
    predict_source(**kwargs)
