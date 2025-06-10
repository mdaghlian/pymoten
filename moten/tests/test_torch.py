import numpy as np
import torch
import math

from moten.core import mk_3d_gabor
from moten.core_torch import mk_3d_gabor_TORCH

def test_mk_3d_gabor_identity():
    """
    Tests that mk_3d_gabor_TORCH and mk_3d_gabor produce identical outcomes.
    """
    # Define common parameters for the gabor filters
    vhsize = (128, 128)
    stimulus_fps = 60
    params = {
        'vhsize': vhsize,
        'stimulus_fps': stimulus_fps,
        'aspect_ratio': 1.0,
        'filter_temporal_width': 40,
        'centerh': 0.5,
        'centerv': 0.5,
        'direction': 30.0,
        'spatial_freq': 8.0,
        'spatial_env': 0.2,
        'temporal_freq': 1.5,
        'temporal_env': 0.25,
        'spatial_phase_offset': math.pi / 4,
    }

    # Generate output from the PyTorch function
    torch_output = mk_3d_gabor_TORCH(**params, device='cpu')

    # Generate output from the NumPy function
    numpy_output = mk_3d_gabor(**params)

    # Compare the outputs for each returned component
    # We use numpy.testing.assert_allclose for numerical stability
    # and to handle potential small floating point differences.
    rtol = 1e-6  # Relative tolerance
    atol = 1e-8  # Absolute tolerance

    # spatial_gabor_sin
    np.testing.assert_allclose(torch_output[0].numpy(), numpy_output[0], rtol=rtol, atol=atol,
                               err_msg="spatial_gabor_sin mismatch")

    # spatial_gabor_cos
    np.testing.assert_allclose(torch_output[1].numpy(), numpy_output[1], rtol=rtol, atol=atol,
                               err_msg="spatial_gabor_cos mismatch")

    # temporal_gabor_sin
    np.testing.assert_allclose(torch_output[2].numpy(), numpy_output[2], rtol=rtol, atol=atol,
                               err_msg="temporal_gabor_sin mismatch")

    # temporal_gabor_cos
    np.testing.assert_allclose(torch_output[3].numpy(), numpy_output[3], rtol=rtol, atol=atol,
                               err_msg="temporal_gabor_cos mismatch")

    print("All Gabor components match between PyTorch and NumPy implementations!")
