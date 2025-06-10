'''
'''
#
# Adapted from MATLAB code written by S. Nishimoto (see Nishimoto, et al., 2011).
# Anwar O. Nunez-Elizalde (Jan, 2016)
#
# Updates:
#  Anwar O. Nunez-Elizalde (Apr, 2020)
#  M. Daghlian - copied from "core.py", adapting to work with tf

import itertools
from PIL import Image

import numpy as np
import torch
import math
from moten.utils import (DotDict,
                         iterator_func,
                         log_compress_TORCH,
                         sqrt_sum_squares_TORCH,
                         pointwise_square_TORCH,
                         )

##############################
#
##############################
def project_stimulus_NOT_FLAT_TORCH(
    stimulus,
    filters,
    quadrature_combination,
    output_nonlinearity,
    vhsize=None,
    dtype=torch.float32,
    device='cuda',
):
    """Compute motion energy filter responses to a single stimulus using PyTorch, matching NumPy exactly."""
    # convert to torch tensor
    if isinstance(stimulus, torch.Tensor):
        stim = stimulus.to(device=device, dtype=dtype)
    else:
        stim = torch.from_numpy(stimulus).to(device=device, dtype=dtype)

    # reshape to (n_images, pixels)
    if stim.ndim == 3:
        n_images, vdim, hdim = stim.shape
        stim = stim.reshape(n_images, -1)
        vhsize = (vdim, hdim)
    else:
        n_images, pixels = stim.shape
        assert vhsize is not None, "vhsize must be provided for 2D stimuli"
        vdim, hdim = vhsize
        assert vdim * hdim == pixels

    n_filters = len(filters)
    responses = torch.zeros((n_images, n_filters), device=device, dtype=dtype)

    for i, params in enumerate(filters):
        sg0, sg90, tg0, tg90 = mk_3d_gabor_TORCH(vhsize, **params, device=device)
        ch_sin, ch_cos = dotdelay_frames_TORCH(sg0, sg90, tg0, tg90, stim)
        comb = quadrature_combination(ch_sin, ch_cos)
        out = output_nonlinearity(comb)
        responses[:, i] = out

    return responses



def project_stimulus_TORCH(
    stimulus,
    filters,
    quadrature_combination=sqrt_sum_squares_TORCH,
    output_nonlinearity=log_compress_TORCH,
    vhsize=(),
    dtype=torch.float32,
    device='cuda',
    masklimit=0.001,
):
    '''Compute motion energy filter responses for multiple stimulus sets using PyTorch.

    Parameters
    ----------
    stimuli_list : list of (np.ndarray or torch.Tensor)
        Each element is a stimulus array of shape (nimages, vdim, hdim) or (nimages, vdim*hdim).
    filters : list of dict
        Filter parameter dictionaries for mk_3d_gabor_TORCH.
    quadrature_combination : callable
    output_nonlinearity : callable
    vhsize : tuple
        Required if any stimulus is already 2D.
    dtype, device : as before

    Returns
    -------
    responses_list : list of torch.Tensor
        Each tensor has shape (nimages_i, nfilters).
    '''
    # ===== Convert and flatten all stimuli =====
    flat_list = []
    shapes = []  # to reconstruct per-set outputs
    if not isinstance(stimulus, list):
        stimulus = [stimulus]
    print(len(stimulus))
    for stim in stimulus:
        # to tensor
        if isinstance(stim, np.ndarray):
            stim = torch.from_numpy(stim)
        stim = stim.to(device=device, dtype=dtype)
        # reshape
        if stim.ndim == 3:
            n, v, h = stim.shape
            stim = stim.reshape(n, -1)
            vhsize = (v, h)
        assert stim.ndim == 2, "Each stimulus must be 2D or 3D"
        flat_list.append(stim)
        shapes.append(stim.shape[0])
    
    # concatenate into one big batch: (N_total, P)
    batch = torch.cat(flat_list, dim=0)
    N_total, P = batch.shape
    vdim, hdim = vhsize
    assert vdim * hdim == P, "vhsize mismatch"
    F = len(filters)

    # ===== Precompute filter banks =====
    # Spatial filters: (F, P)
    spatial_sin = []
    spatial_cos = []
    temporal_sin = []
    temporal_cos = []
    for params in filters:
        sg0, sg90, tg0, tg90 = mk_3d_gabor_TORCH(vhsize, **params, device=device)
        spatial_sin.append(sg0.reshape(-1)) # flatten as in dotspatial_frames...
        spatial_cos.append(sg90.reshape(-1))
        temporal_sin.append(tg0)
        temporal_cos.append(tg90)
    spatial_sin = torch.stack(spatial_sin, dim=0)   # (F, P)
    spatial_cos = torch.stack(spatial_cos, dim=0)   # (F, P)
    temporal_sin = torch.stack(temporal_sin, dim=0) # (F, T)
    temporal_cos = torch.stack(temporal_cos, dim=0) # (F, T)

    # ===== Spatial projection =====    
    # -> following dotspatial_frames 
    gabors = torch.stack([spatial_sin, spatial_cos], dim=0)  # shape: (2, F, P)
    mask = gabors.abs().sum(0) > masklimit
    masked_gabors = gabors * mask[np.newaxis,...]
    batch_T = batch.T # P x nframes 

    # Batched matrix multiplication
    gabor_prod_raw = masked_gabors @ batch_T
    print(gabor_prod_raw.shape)
    gsin = gabor_prod_raw[0,:,:]
    gcos = gabor_prod_raw[1,:,:]
    bloop
    print(mask)
    print(mask.shape)
    batch_stack = batch.unsqueeze(-1).expand(-1, -1, F)
    print(batch_stack.T.shape)
    gabor_prod = (gabors[:,mask].squeeze() @ batch[:,mask.T].squeeze()).T
    print(gabor_prod.shape)

    # Mask spatial
    # spatial_sin *= mask
    # spatial_cos *= mask
    # gabor
    bloop
    gsin = batch @ spatial_sin.T  # (N_total, F)
    gcos = batch @ spatial_cos.T  # (N_total, F)

    # ===== Temporal filtering =====
    # expand dims
    gsin = gsin.unsqueeze(-1)   # (N_total, F, 1)
    gcos = gcos.unsqueeze(-1)
    outs = gsin * temporal_cos.unsqueeze(0) + gcos * temporal_sin.unsqueeze(0)
    outc = -gsin * temporal_sin.unsqueeze(0) + gcos * temporal_cos.unsqueeze(0)

    # align delays
    T = temporal_sin.shape[1]
    tdxc = int(torch.ceil(torch.tensor(T/2.0)).item())
    delays = torch.arange(T, device=device) - tdxc + 1
    aligned_outs = torch.zeros_like(outs)
    aligned_outc = torch.zeros_like(outc)
    for t, d in enumerate(delays.tolist()):
        if d > 0:
            aligned_outs[d:, :, t] = outs[:-d, :, t]
            aligned_outc[d:, :, t] = outc[:-d, :, t]
        elif d < 0:
            aligned_outs[:d, :, t] = outs[-d:, :, t]
            aligned_outc[:d, :, t] = outc[-d:, :, t]
        else:
            aligned_outs[:, :, t] = outs[:, :, t]      
            aligned_outc[:, :, t] = outc[:, :, t]        

    channel_sin = aligned_outs.sum(dim=2)  # (N_total, F)
    channel_cos = aligned_outc.sum(dim=2)

    # Combine + nonlinearity
    combined = quadrature_combination(channel_sin, channel_cos)
    responses = output_nonlinearity(combined)  # (N_total, F)

    # ===== Split back into list =====
    responses_list = []
    idx = 0
    for n in shapes:
        responses_list.append(responses[idx:idx+n])
        idx += n

    return responses_list

##############################
# core functionality
##############################

def mk_3d_gabor_TORCH(vhsize,
                stimulus_fps,
                aspect_ratio='auto',
                filter_temporal_width='auto',
                centerh=0.5,
                centerv=0.5,
                direction=45.0,
                spatial_freq=16.0,
                spatial_env=0.3,
                temporal_freq=2.0,
                temporal_env=0.3,
                spatial_phase_offset=0.0,
                device='cpu'):
    '''Same function as described, but fully in PyTorch.'''
    
    vdim, hdim = vhsize

    if aspect_ratio == 'auto':
        aspect_ratio = hdim / float(vdim)

    if filter_temporal_width == 'auto':
        filter_temporal_width = int(stimulus_fps * (2/3.))

    assert math.isclose(filter_temporal_width, int(filter_temporal_width)), \
        "filter_temporal_width should be an integer or close to one."
    filter_temporal_width = int(filter_temporal_width)

    # create coordinate grids
    dh = torch.linspace(0, aspect_ratio, hdim, device=device, )
    dv = torch.linspace(0, 1, vdim, device=device, )
    dt = torch.linspace(0, 1, filter_temporal_width+1, device=device, )[:-1] # endpoint=False
    # *** M Daghlian - decided to propogate this forward...
    # AN: Actually, `dt` should include endpoint.    
    # Currently, the center of the filter width is +(1./fps)/2.
    # However, this would break backwards compatibility.
    # TODO: Allow for `dt_endpoint` as an argument
    # and set default to False.
    
    
    ihs, ivs = torch.meshgrid(dh, dv, indexing='xy')  

    # Compute frequencies
    theta_rad = direction / 180.0 * math.pi
    fh = -spatial_freq * math.cos(theta_rad) * 2 * math.pi
    fv = spatial_freq * math.sin(theta_rad) * 2 * math.pi
    ft = temporal_freq*(filter_temporal_width/stimulus_fps)*2*math.pi

    # Spatial Gaussian envelope
    spatial_gaussian = torch.exp(-((ihs - centerh) ** 2 + (ivs - centerv) ** 2) / (2 * spatial_env ** 2))

    # Spatial grating
    phase = (ihs - centerh) * fh + (ivs - centerv) * fv + spatial_phase_offset
    spatial_grating_sin = torch.sin(phase)
    spatial_grating_cos = torch.cos(phase)

    # Spatial Gabor
    spatial_gabor_sin = spatial_gaussian * spatial_grating_sin
    spatial_gabor_cos = spatial_gaussian * spatial_grating_cos

    # Temporal Gaussian and grating
    temporal_gaussian = torch.exp(-((dt - 0.5) ** 2) / (2 * temporal_env ** 2))
    temporal_grating_sin = torch.sin((dt - 0.5) * ft)
    temporal_grating_cos = torch.cos((dt - 0.5) * ft)

    temporal_gabor_sin = temporal_gaussian * temporal_grating_sin
    temporal_gabor_cos = temporal_gaussian * temporal_grating_cos

    return spatial_gabor_sin, spatial_gabor_cos, temporal_gabor_sin, temporal_gabor_cos

def dotspatial_frames_TORCH(spatial_gabor_sin, spatial_gabor_cos,
                      stimulus,
                      masklimit=0.001):
    '''Dot the spatial gabor filters with the stimulus using PyTorch

    Parameters
    ----------
    spatial_gabor_sin : torch.Tensor, (vdim, hdim)
    spatial_gabor_cos : torch.Tensor, (vdim, hdim)
        Spatial Gabor quadrature pair
    stimulus : torch.Tensor, (nimages, vdim*hdim)
        Movie frames with spatial dimension flattened
    masklimit : float
        Threshold to find the non-zero filter region

    Returns
    -------
    channel_sin : torch.Tensor, (nimages,)
    channel_cos : torch.Tensor, (nimages,)
        Filter responses for each frame
    '''
    # Flatten the spatial filters into vectors
    gabor_sin_flat = spatial_gabor_sin.flatten()
    gabor_cos_flat = spatial_gabor_cos.flatten()

    # Stack into shape (2, vdim*hdim)
    gabors = torch.stack([gabor_sin_flat, gabor_cos_flat], dim=0)  # shape: (2, pixels)

    # Create a mask to keep only meaningful (non-negligible) filter elements
    mask = torch.sum(torch.abs(gabors), dim=0) > masklimit  # shape: (pixels,)

    # Apply mask to gabors and stimulus
    gabors_masked = gabors[:, mask]  # shape: (2, num_masked_pixels)
    stimulus_masked = stimulus[:, mask]  # shape: (nimages, num_masked_pixels)

    # Matrix multiplication: (nimages x pixels) @ (pixels x 2) -> (nimages x 2)
    gabor_prod = stimulus_masked @ gabors_masked.T  # shape: (nimages, 2)

    # Split into sin and cos responses
    gabor_sin = gabor_prod[:, 0]
    gabor_cos = gabor_prod[:, 1]

    return gabor_sin, gabor_cos


def dotdelay_frames_TORCH_OLD(spatial_gabor_sin, spatial_gabor_cos,
                    temporal_gabor_sin, temporal_gabor_cos,
                    stimulus,
                    masklimit=0.001):
    '''Convolve the motion energy filter with a stimulus in PyTorch

    Parameters
    ----------
    spatial_gabor_sin : torch.Tensor, (vdim, hdim)
    spatial_gabor_cos : torch.Tensor, (vdim, hdim)
        Spatial gabor quadrature pair

    temporal_gabor_sin : torch.Tensor, (T,)
    temporal_gabor_cos : torch.Tensor, (T,)
        Temporal gabor quadrature pair

    stimulus : torch.Tensor, (nimages, vdim*hdim)
        Flattened movie frames

    masklimit : float
        Threshold for masking near-zero filter elements

    Returns
    -------
    channel_sin, channel_cos : torch.Tensor, (nimages,)
        Motion‐energy filter responses
    '''
    # 1) get the spatial filter outputs per frame
    gabor_sin, gabor_cos = dotspatial_frames_TORCH(
        spatial_gabor_sin, spatial_gabor_cos, stimulus, masklimit=masklimit
    )
    # shape (nimages,) each
    nimages = gabor_sin.shape[0]

    # 2) pack into (nimages, 2)
    gabor_prod = torch.stack([gabor_sin, gabor_cos], dim=1)

    # 3) pack temporal filters into (2, T)
    temporal = torch.stack([temporal_gabor_sin, temporal_gabor_cos], dim=0)  # (2, T)
    T = temporal.shape[1]

    # 4) compute the cross‐quadrature delay outputs
    #    outs[i,t] = sin_i * cos_temporal[t] + cos_i * sin_temporal[t]
    #    outc[i,t] = -sin_i * sin_temporal[t] + cos_i * cos_temporal[t]
    sin_i = gabor_prod[:, 0:1]           # (nimages,1)
    cos_i = gabor_prod[:, 1:2]           # (nimages,1)
    sin_t = temporal[0:1, :]             # (1,T)
    cos_t = temporal[1:2, :]             # (1,T)

    outs = sin_i @ cos_t + cos_i @ sin_t     # (nimages, T)
    outc = -sin_i @ sin_t + cos_i @ cos_t     # (nimages, T)

    # 5) roll each column by its delay to align in time
    #    delays = arange(T) - ceil(T/2) + 1
    tdxc = int(torch.ceil(torch.tensor(T / 2.0)).item())
    delays = torch.arange(T, device=outs.device) - tdxc + 1

    nouts = torch.zeros_like(outs)
    noutc = torch.zeros_like(outc)
    for t in range(T):
        d = int(delays[t].item())
        if d == 0:
            nouts[:, t] = outs[:, t]
            noutc[:, t] = outc[:, t]
        elif d > 0:
            nouts[d:, t] = outs[:-d, t]
            noutc[d:, t] = outc[:-d, t]
        else:  # d < 0
            nouts[:d, t] = outs[-d:, t]
            noutc[:d, t] = outc[-d:, t]

    # 6) sum across the temporal axis to get final channels
    channel_sin = nouts.sum(dim=1)   # (nimages,)
    channel_cos = noutc.sum(dim=1)   # (nimages,)

    return channel_sin, channel_cos

