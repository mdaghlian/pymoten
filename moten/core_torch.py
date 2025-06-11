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
def project_stimulus_TORCH(
    stimulus,
    filters,
    quadrature_combination=sqrt_sum_squares_TORCH,
    output_nonlinearity=log_compress_TORCH,
    vhsize=(),
    dtype=torch.float32,
    device='cpu',
    masklimit=0.001,
    max_batch_size=5000, # New parameter: Maximum number of frames per batch
):
    '''
    Wrapper around project_stimulus_TORCH to handle large stimuli by processing them in chunks.

    Parameters
    ----------
    stimulus : (np.ndarray or torch.Tensor) or list of (np.ndarray or torch.Tensor)
        Each element is a stimulus array of shape (nimages, vdim, hdim) or (nimages, vdim*hdim).
        Can be a single stimulus or a list of stimuli.
    filters : list of dict
        Filter parameter dictionaries for mk_3d_gabor_TORCH.
    quadrature_combination : callable
    output_nonlinearity : callable
    vhsize : tuple
        Required if any stimulus is already 2D.
    dtype, device : as before
    masklimit : float
    max_batch_size : int, optional
        The maximum number of frames to process in a single batch.
        If None, the function will process all stimuli in one go (like the original).
        Set this to a value that fits your GPU memory.

    Returns
    -------
    responses_list : list of torch.Tensor
        Each tensor has shape (nimages_i, nfilters), corresponding to the input stimuli.
    '''
    # Ensure stimulus is a list of tensors for consistent iteration
    if not isinstance(stimulus, list):
        stimulus = [stimulus]

    all_responses = []
    current_chunk = []
    current_chunk_total_frames = 0

    for stim_idx, stim in enumerate(stimulus):
        # Convert to tensor (if numpy) and get shape
        if isinstance(stim, np.ndarray):
            stim_tensor = torch.from_numpy(stim)
        else:
            stim_tensor = stim

        # Determine number of frames for the current stimulus
        num_frames_in_stim = stim_tensor.shape[0] if stim_tensor.ndim == 3 else stim_tensor.shape[0]

        # If max_batch_size is set and adding this stimulus would exceed it,
        # process the current chunk before adding the new stimulus.
        # Ensure current_chunk is not empty to avoid processing an empty batch.
        if max_batch_size is not None and \
           (current_chunk_total_frames + num_frames_in_stim > max_batch_size) and \
           current_chunk:
            print(f"Processing chunk with {current_chunk_total_frames} frames...")
            chunk_responses_list = project_stimulus_TORCH(
                stimulus=current_chunk,
                filters=filters,
                quadrature_combination=quadrature_combination,
                output_nonlinearity=output_nonlinearity,
                vhsize=vhsize, # Pass vhsize to the inner function
                dtype=dtype,
                device=device,
                masklimit=masklimit,
            )
            all_responses.extend(chunk_responses_list)
            # Reset chunk for the next batch
            current_chunk = []
            current_chunk_total_frames = 0

        # Add current stimulus to the chunk
        current_chunk.append(stim_tensor)
        current_chunk_total_frames += num_frames_in_stim

    # Process any remaining stimulus in the last chunk
    if current_chunk:
        print(f"Processing final chunk with {current_chunk_total_frames} frames...")
        chunk_responses_list = project_stimulus_batch_TORCH(
            stimulus=current_chunk,
            filters=filters,
            quadrature_combination=quadrature_combination,
            output_nonlinearity=output_nonlinearity,
            vhsize=vhsize, # Pass vhsize to the inner function
            dtype=dtype,
            device=device,
            masklimit=masklimit,
        )
        all_responses.extend(chunk_responses_list)

    return all_responses


def project_stimulus_batch_TORCH(
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
    channel_sin, channel_cos = dotdelay_frames_TORCH(
        spatial_gabor_sin = spatial_sin, 
        spatial_gabor_cos = spatial_cos,
        temporal_gabor_cos = temporal_cos,
        temporal_gabor_sin= temporal_sin,
        batch=batch, 
        shapes=shapes,
        masklimit=masklimit, 
    )

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
                      batch,
                      masklimit=0.001):
    '''Dot the spatial gabor filters with the stimulus using PyTorch

    Parameters
    ----------
    spatial_gabor_sin : torch.Tensor, F x P
    spatial_gabor_cos : torch.Tensor, F x P
        Spatial Gabor quadrature pair
    batch : torch.Tensor, (nframes x P)
        Movie frames with spatial dimension flattened
    masklimit : float
        Threshold to find the non-zero filter region

    Returns
    -------
    gsin : torch.Tensor, F x nimages 
    gcos : torch.Tensor, F x nimages
        Filter responses for each frame
    '''
    # ===== Spatial projection =====    
    # -> following dotspatial_frames 
    gabors = torch.stack([spatial_gabor_sin, spatial_gabor_cos], dim=0)  # shape: (2, F, P)
    mask = gabors.abs().sum(0) > masklimit
    masked_gabors = gabors * mask[np.newaxis,...]
    batch_T = batch.T # P x nframes 

    # Batched matrix multiplication
    gabor_prod_raw = masked_gabors @ batch_T
    gsin = gabor_prod_raw[0,:,:]
    gcos = gabor_prod_raw[1,:,:]

    return gsin, gcos
import torch.nn.functional as F
def dotdelay_frames_TORCH(
    spatial_gabor_sin,
    spatial_gabor_cos,
    temporal_gabor_sin,
    temporal_gabor_cos,
    batch,
    shapes, # <-- Added 'shapes' parameter
    masklimit=0.001,
):
    '''Compute motion energy filter responses for a batch of stimulus frames using PyTorch,
    respecting individual movie boundaries during temporal convolution.

    This function performs spatial convolution on the full batch, then splits
    the responses by original movie shape to perform independent temporal
    convolutions, and finally concatenates the results.

    Parameters
    ----------
    spatial_gabor_sin : torch.Tensor, (num_filters, P)
        Stacked spatial sine Gabor filters, where num_filters is the number of filters,
        and P is the flattened spatial dimension (vdim*hdim).
    spatial_gabor_cos : torch.Tensor, (num_filters, P)
        Stacked spatial cosine Gabor filters, quadrature pair to spatial_gabor_sin.
    temporal_gabor_sin : torch.Tensor, (num_filters, T)
        Stacked temporal sine Gabor filters, where T is the temporal filter width.
    temporal_gabor_cos : torch.Tensor, (num_filters, T)
        Stacked temporal cosine Gabor filters, quadrature pair to temporal_gabor_sin.
    batch : torch.Tensor, (N_total, P)
        Concatenated stimulus frames, where N_total is the total number of frames
        across all movies, and P is the flattened spatial dimension.
    shapes : list of int
        A list of integers, where each integer is the number of frames in an
        original movie. This is used to reconstruct the movie boundaries for
        temporal convolution.
    masklimit : float-like, optional
        Threshold for masking out small filter values in spatial convolution. Defaults to 0.001.

    Returns
    -------
    channel_sin : torch.Tensor, (N_total, num_filters)
        The sine component of the spatio-temporal filter response for each
        frame and each gabor channel, combined across all movies.
    channel_cos : torch.Tensor, (N_total, num_filters)
        The cosine component of the spatio-temporal filter response for each
        frame and each gabor channel, combined across all movies.
    '''
    # Extract dimensions
    num_filters, P = spatial_gabor_sin.shape  # num_filters: num_filters, P: spatial_pixels
    N_total = batch.shape[0]        # N_total: total_frames
    T = temporal_gabor_sin.shape[1] # T: temporal_filter_width

    # --- 1. Vectorized Spatial Convolution ---
    # Apply mask: Sum absolute values of sine and cosine components for each spatial filter.
    spatial_mask = (torch.abs(spatial_gabor_sin) + torch.abs(spatial_gabor_cos)) > masklimit
    
    # Apply mask to filters
    masked_spatial_sin = spatial_gabor_sin * spatial_mask
    masked_spatial_cos = spatial_gabor_cos * spatial_mask

    # Perform dot product: (N_total, P) @ (P, num_filters) -> (N_total, num_filters)
    # This computes spatial responses for all frames and all filters simultaneously.
    spatial_response_sin = batch @ masked_spatial_sin.T
    spatial_response_cos = batch @ masked_spatial_cos.T

    # --- 2. Temporal Convolution (Correlation) per Movie ---
    # The previous NumPy `delays` loop implemented a form of correlation.
    # For `F.conv1d` to perform correlation, the kernel needs to be reversed.
    temporal_gabor_sin_rev = torch.flip(temporal_gabor_sin, dims=[-1]) # (num_filters, T)
    temporal_gabor_cos_rev = torch.flip(temporal_gabor_cos, dims=[-1]) # (num_filters, T)

    # Reshape temporal filters for `conv1d`: (num_filters, T) -> (num_filters, 1, T)
    # `groups=num_filters` makes `conv1d` apply each kernel to its corresponding input channel.
    kernel_cos = temporal_gabor_cos_rev.unsqueeze(1) # (num_filters, 1, T)
    kernel_sin = temporal_gabor_sin_rev.unsqueeze(1) # (num_filters, 1, T)

    # Calculate padding for 'same' mode correlation: output length equals input length (N_total)
    padding = (T - 1) // 2

    # Split spatial responses back into individual movie segments
    # Each element in these lists will be (N_movie_i, num_filters)
    spatial_sin_split = torch.split(spatial_response_sin, shapes, dim=0)
    spatial_cos_split = torch.split(spatial_response_cos, shapes, dim=0)

    convolved_sin_list = []
    convolved_cos_list = []

    # Loop through each movie's spatial responses to apply temporal convolution independently
    for i in range(len(shapes)):
        current_movie_sin = spatial_sin_split[i] # Shape (N_movie_i, num_filters)
        current_movie_cos = spatial_cos_split[i] # Shape (N_movie_i, num_filters)

        # Reshape current movie's spatial responses for `conv1d`:
        # From (N_movie_i, num_filters) to (1, num_filters, N_movie_i)
        # The .T transposes to (num_filters, N_movie_i), then unsqueeze(0) adds batch dim.
        input_sin_for_conv = current_movie_sin.T.unsqueeze(0) # (1, num_filters, N_movie_i)
        input_cos_for_conv = current_movie_cos.T.unsqueeze(0) # (1, num_filters, N_movie_i)

        # Perform the 1D convolutions for each term for the current movie
        # Output shape: (1, num_filters, N_movie_i)
        conv_sin_cos = F.conv1d(input_sin_for_conv, kernel_cos, padding=padding, groups=num_filters)
        conv_cos_sin = F.conv1d(input_cos_for_conv, kernel_sin, padding=padding, groups=num_filters)
        conv_sin_sin = F.conv1d(input_sin_for_conv, kernel_sin, padding=padding, groups=num_filters)
        conv_cos_cos = F.conv1d(input_cos_for_conv, kernel_cos, padding=padding, groups=num_filters)

        # Update: Fixed squeezing and transposing after conv1d
        # The result of conv1d is (1, num_filters, N_movie_i)
        # Squeeze the batch dimension: (num_filters, N_movie_i)
        # Transpose: (N_movie_i, num_filters)
        convolved_sin_list.append((conv_sin_cos.squeeze(0) + conv_cos_sin.squeeze(0)).T)
        convolved_cos_list.append((conv_cos_cos.squeeze(0) - conv_sin_sin.squeeze(0)).T)

    # Concatenate results from all movies back into a single tensor
    channel_sin = torch.cat(convolved_sin_list, dim=0)
    channel_cos = torch.cat(convolved_cos_list, dim=0)

    return channel_sin, channel_cos

# def dotdelay_frames_TORCH(spatial_gabor_sin, spatial_gabor_cos,
#                     temporal_gabor_sin, temporal_gabor_cos,
#                     batch,
#                     masklimit=0.001):
#     '''Convolve the motion energy filter with a stimulus in PyTorch

#     Parameters
#     ----------
#     spatial_gabor_sin : torch.Tensor, F x P
#     spatial_gabor_cos : torch.Tensor, F x P
#         Spatial Gabor quadrature pair
    
#     temporal_gabor_sin : torch.Tensor, (T,)
#     temporal_gabor_cos : torch.Tensor, (T,)
#         Temporal gabor quadrature pair

#     batch : torch.Tensor, (nframes x P)
#         Movie frames with spatial dimension flattened
#     masklimit : float
#         Threshold to find the non-zero filter region

#     Returns
#     -------
#     channel_sin, channel_cos : torch.Tensor, (nimages,)
#         Motion‐energy filter responses
#     '''
#     # 1) get the spatial filter outputs per frame
#     gabor_sin, gabor_cos = dotspatial_frames_TORCH(
#         spatial_gabor_sin, spatial_gabor_cos, batch, masklimit=masklimit
#     )
    
#     # shape (nimages,) each
#     nimages = gabor_sin.shape[0]

#     # 2) pack into (nimages, 2)
#     gabor_prod = torch.stack([gabor_sin, gabor_cos], dim=1)

#     # 3) pack temporal filters into (2, T)
#     temporal = torch.stack([temporal_gabor_sin, temporal_gabor_cos], dim=0)  # (2, T)
#     T = temporal.shape[1]

#     # 4) compute the cross‐quadrature delay outputs
#     #    outs[i,t] = sin_i * cos_temporal[t] + cos_i * sin_temporal[t]
#     #    outc[i,t] = -sin_i * sin_temporal[t] + cos_i * cos_temporal[t]
#     sin_i = gabor_prod[:, 0:1]           # (nimages,1)
#     cos_i = gabor_prod[:, 1:2]           # (nimages,1)
#     sin_t = temporal[0:1, :]             # (1,T)
#     cos_t = temporal[1:2, :]             # (1,T)

#     outs = sin_i @ cos_t + cos_i @ sin_t     # (nimages, T)
#     outc = -sin_i @ sin_t + cos_i @ cos_t     # (nimages, T)

#     # 5) roll each column by its delay to align in time
#     #    delays = arange(T) - ceil(T/2) + 1
#     tdxc = int(torch.ceil(torch.tensor(T / 2.0)).item())
#     delays = torch.arange(T, device=outs.device) - tdxc + 1

#     nouts = torch.zeros_like(outs)
#     noutc = torch.zeros_like(outc)
#     for t in range(T):
#         d = int(delays[t].item())
#         if d == 0:
#             nouts[:, t] = outs[:, t]
#             noutc[:, t] = outc[:, t]
#         elif d > 0:
#             nouts[d:, t] = outs[:-d, t]
#             noutc[d:, t] = outc[:-d, t]
#         else:  # d < 0
#             nouts[:d, t] = outs[-d:, t]
#             noutc[:d, t] = outc[-d:, t]

#     # 6) sum across the temporal axis to get final channels
#     channel_sin = nouts.sum(dim=1)   # (nimages,)
#     channel_cos = noutc.sum(dim=1)   # (nimages,)

#     return channel_sin, channel_cos

