#!/usr/bin/env python
import os
import click
import numpy as np

from omegaconf import OmegaConf
from scabha.schema_utils import clickify_parameters


import Tigger
from collections import namedtuple

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
    pol_name = "::".join((msname, "POLARIZATION"))
    xds = xds_from_table(msname)[0]
    uvw = xds['UVW'].values
    spwds = xds_from_table(spwname)[0]
    fieldds = xds_from_table(field_name)[0]
    frequency = spwds['CHAN_FREQ'].values.squeeze()
    phase_center = fieldds['PHASE_DIR'].values.squeeze()[0]
    pols = xds_from_table(pol_name)[0]
    corr_shape = pols.NUM_CORR.values[0]
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
    vis = wsclean_predict(
        uvws.compute(), lms.compute(),
        np.array([source_type]),
        np.array([flux], dtype=np.float64),
        np.array(coeffs, dtype=np.float64),
        np.array(log_poly, dtype=np.float64),
        np.array([ref_freq]),
        gauss_shape,
        frequency)

    if corr_shape == 1:
        visibilities = vis
    elif corr_shape == 2:
        visibilities = da.concatenate([vis, vis], axis=2)
    elif corr_shape == 4:
        zeros = da.zeros_like(vis)
        visibilities = da.concatenate([vis, zeros, zeros, vis], axis=2)
    visibilities = visibilities.rechunk({0:10000, 2: corr_shape})

    # Update the measurement set with the predicted visibilities
    # Assign visibilities to MODEL_DATA array on the dataset
    xds = xds.assign(**{column: (("row", "chan", "corr"), visibilities)})
    # Write to the table
    da.compute(xds_to_table(xds, msname, [column]))
    print(f"Successfully populated measurement set '{msname}' with column '{column}'.")
    return visibilities


def parse_sky_model(filename: str, chunks: tuple[int, ...] | int) -> dict[str, namedtuple]:
    """
    Parses a Tigger sky model file and organizes source data.

    Parameters
    ----------
    filename : str
        Path to the sky model file.
    chunks : tuple of ints or int
        Chunk size for Dask arrays.

    Returns
    -------
    source_data : dict
        A dictionary containing source data:
        {'point': Point, 'gauss': Gauss}, where
        - Point and Gauss are namedtuples containing:
          - radec: RA/Dec coordinates
          - stokes: Stokes parameters (I, Q, U, V)
          - spi: Spectral index
          - ref_freq: Reference frequency
          - shape: Gaussian shape parameters (for 'gauss' sources only)
    """
    sky_model = Tigger.load(filename, verbose=False)

    # Placeholder for sources with no spectrum
    _empty_spectrum = object()

    # Data containers for point and Gaussian sources
    point_radec, point_stokes, point_spi, point_ref_freq = [], [], [], []
    gauss_radec, gauss_stokes, gauss_spi, gauss_ref_freq, gauss_shape = [], [], [], [], []

    for source in sky_model.sources:
        ra, dec = source.pos.ra, source.pos.dec
        typecode = source.typecode.lower()

        # Stokes parameters
        I, Q, U, V = source.flux.I, source.flux.Q, source.flux.U, source.flux.V

        # Spectrum and reference frequency
        spectrum = getattr(source, "spectrum", _empty_spectrum) or _empty_spectrum
        ref_freq = getattr(spectrum, "freq0", sky_model.freq0)

        # Spectral index (SPI)
        spi = [[getattr(spectrum, "spi", 0)] * 4]

        if typecode == "gau":  # Gaussian sources
            emaj, emin, pa = source.shape.ex, source.shape.ey, source.shape.pa
            gauss_radec.append([ra, dec])
            gauss_stokes.append([I, Q, U, V])
            gauss_spi.append(spi)
            gauss_ref_freq.append(ref_freq)
            gauss_shape.append([emaj, emin, pa])
        elif typecode == "pnt":  # Point sources
            point_radec.append([ra, dec])
            point_stokes.append([I, Q, U, V])
            point_spi.append(spi)
            point_ref_freq.append(ref_freq)
        else:
            raise ValueError(f"Unknown source morphology: {typecode}")

    # Define namedtuples for organized storage
    Point = namedtuple("Point", ["radec", "stokes", "spi", "ref_freq"])
    Gauss = namedtuple("Gauss", ["radec", "stokes", "spi", "ref_freq", "shape"])

    # Populate the source_data dictionary with Dask arrays
    source_data = {}
    if point_radec:
        source_data["point"] = Point(
            radec=da.from_array(point_radec, chunks=(chunks, -1)),
            stokes=da.from_array(point_stokes, chunks=(chunks, -1)),
            spi=da.from_array(point_spi, chunks=(chunks, 1, -1)),
            ref_freq=da.from_array(point_ref_freq, chunks=chunks),
        )
    if gauss_radec:
        source_data["gauss"] = Gauss(
            radec=da.from_array(gauss_radec, chunks=(chunks, -1)),
            stokes=da.from_array(gauss_stokes, chunks=(chunks, -1)),
            spi=da.from_array(gauss_spi, chunks=(chunks, 1, -1)),
            ref_freq=da.from_array(gauss_ref_freq, chunks=chunks),
            shape=da.from_array(gauss_shape, chunks=(chunks, -1)),
        )

    return source_data

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
