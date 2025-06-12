import numpy as np
import torch
import math
import time

from moten.core import mk_3d_gabor
from moten.core_torch import mk_3d_gabor_TORCH
from moten.pyramids import MotionEnergyPyramid


def compare_gabor_outputs(params, device='cpu', rtol=1e-5, atol=1e-5):
    """
    Compare outputs of NumPy and PyTorch gabor generators for given params.
    Returns max absolute differences for each component and timings.
    """
    # PyTorch version
    start_t = time.time()
    torch_out = mk_3d_gabor_TORCH(**params, device=device)
    dt_torch = time.time() - start_t

    # NumPy version
    start_n = time.time()
    numpy_out = mk_3d_gabor(**params)
    dt_numpy = time.time() - start_n

    # compute max differences
    diffs = [
        np.max(np.abs(torch_out[i].cpu().numpy() - numpy_out[i]))
        for i in range(len(numpy_out))
    ]
    return {
        'times': {'torch': dt_torch, 'numpy': dt_numpy},
        'max_diffs': diffs
    }


def test_gabor_parametric():
    """
    Runs compare_gabor_outputs over a grid of filter parameters.
    Logs benchmarks and max diffs.
    """
    vhsize = (128, 128)
    stimulus_fps = 60

    # Define parameter ranges
    spatial_freqs = [2.0, 4.0, 8.0, 16.0]
    temporal_freqs = [0.5, 1.5, 3.0]
    directions = [0.0, 45.0, 90.0, 135.0]

    base_params = {
        'vhsize': vhsize,
        'stimulus_fps': stimulus_fps,
        'aspect_ratio': 1.0,
        'filter_temporal_width': 40,
        'centerh': 0.5,
        'centerv': 0.5,
        'spatial_env': 0.2,
        'temporal_env': 0.25,
        'spatial_phase_offset': math.pi / 4,
    }

    results = []
    for sf in spatial_freqs:
        for tf in temporal_freqs:
            for dir_deg in directions:
                params = base_params.copy()
                params.update({'spatial_freq': sf,
                               'temporal_freq': tf,
                               'direction': dir_deg})
                res = compare_gabor_outputs(params)
                results.append({
                    'spatial_freq': sf,
                    'temporal_freq': tf,
                    'direction': dir_deg,
                    'time_torch': res['times']['torch'],
                    'time_numpy': res['times']['numpy'],
                    'max_diff_spatial_sin': res['max_diffs'][0],
                    'max_diff_spatial_cos': res['max_diffs'][1],
                    'max_diff_temporal_sin': res['max_diffs'][2],
                    'max_diff_temporal_cos': res['max_diffs'][3],
                })

    # Print summary
    print("Parametric Gabor Comparison Results:")
    for r in results:
        print(f"SF={r['spatial_freq']}, TF={r['temporal_freq']}, Dir={r['direction']} -> "
              f"Torch {r['time_torch']:.4f}s, NumPy {r['time_numpy']:.4f}s, "
              f"Diffs [S_sin={r['max_diff_spatial_sin']:.2e}, "
              f"S_cos={r['max_diff_spatial_cos']:.2e}, "
              f"T_sin={r['max_diff_temporal_sin']:.2e}, "
              f"T_cos={r['max_diff_temporal_cos']:.2e}]")


def test_project_stimulus_perf():
    """
    Compare project_stimulus performance and outputs for NumPy vs Torch.
    """
    print("Running project_stimulus performance and correctness test...")
    vhsize = (128, 128)
    stimulus_fps = 60
    num_frames = 20
    num_stim = 10

    # generate random stimuli
    stim_list = [
        np.random.uniform(0, 100, size=(num_frames, *vhsize))
        for _ in range(num_stim)
    ]

    pyramid = MotionEnergyPyramid(stimulus_vhsize=vhsize, stimulus_fps=stimulus_fps)

    # NumPy
    start_np = time.time()
    out_np = [pyramid.project_stimulus(stim, use_torch=False) for stim in stim_list]
    t_np = time.time() - start_np

    # Torch
    start_t = time.time()
    out_torch = pyramid.project_stimulus(stim_list, use_torch=True)
    t_torch = time.time() - start_t

    speedup = t_np / t_torch if t_torch > 0 else float('inf')
    print(f"Torch is {speedup:.2f}x faster than NumPy ({t_torch:.4f}s vs {t_np:.4f}s)")

    # correctness
    max_diff = 0.0
    for i in range(num_stim):
        arr_t = out_torch[i].cpu().numpy()
        arr_n = out_np[i]
        max_diff = max(max_diff, np.max(np.abs(arr_t - arr_n)))

    print(f"Max abs difference: {max_diff:.2e}")


if __name__ == "__main__":
    test_gabor_parametric()
    test_project_stimulus_perf()
