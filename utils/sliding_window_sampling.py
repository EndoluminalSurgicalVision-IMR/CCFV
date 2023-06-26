import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union
import torch
import torch.nn.functional as F



from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    fall_back_tuple,
    look_up_option
)

import numpy as np


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: list):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        self.handlers = {layer: None for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.handlers[layer_id] = layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self._features[layer_id] = input
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features
    
    def remove_handler(self):
        for handler in self.handlers.values():
            handler.remove()

def ms_sliding_window_sampling(
    layers: list,
    sample_num: dict,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    overlap: float = 0.0,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:

    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i])
                       for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(
        padding_mode, PytorchPadMode), value=cval)

    scan_interval = _get_scan_interval(
        image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(
                valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(
        importance_map, torch.Tensor, device, compute_dtype)[0]  # type: ignore
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(
        torch.float32), min=min_non_zero).to(compute_dtype)

    # for each patch
    labels = labels.as_tensor()
    feat_extractor = FeatureExtractor(predictor, layers)
    sample_dict = {layer:{j:[] for j in range(len(torch.unique(labels)))} for layer in layers}
    res_dict = {layer:{j:[] for j in range(len(torch.unique(labels)))} for layer in layers}

    for layer in layers:
        for lb in torch.unique(labels):
            lb_idx = torch.nonzero(labels==lb, as_tuple=False).numpy()
            rand_idx = np.random.choice(len(lb_idx), min(len(lb_idx), sample_num[layer]), replace=False)
            sample_dict[layer][int(lb)] = lb_idx[rand_idx]
    # import pdb; pdb.set_trace()
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):  ## progress=Fasle
        slice_range = range(slice_g, min(
            slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1),
             slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat(
            [convert_data_type(inputs[win_slice], torch.Tensor)[0]
             for win_slice in unravel_slice]
        ).to(sw_device)
        
        features = feat_extractor(window_data)
        feat_extractor.remove_handler()

        out_shape = roi_size
        for layer in layers:
            seg_prob_out = features[layer][0]
            if seg_prob_out.shape != out_shape:
                seg_prob_out = F.interpolate(seg_prob_out, size=out_shape, mode='trilinear')
            for idx, prob_out in zip(unravel_slice, seg_prob_out): 
                for lb in sample_dict[layer].keys():
                    visited = np.array([])
                    for i in range(len(sample_dict[layer][lb])):
                        sample_idx = sample_dict[layer][lb][i]
                        if idx[2].start <= sample_idx[2] < idx[2].stop and \
                            idx[3].start <= sample_idx[3] < idx[3].stop and \
                                idx[4].start <= sample_idx[4] < idx[4].stop:
                            point_feat = prob_out[:, sample_idx[2]-idx[2].start, sample_idx[3]-idx[3].start, sample_idx[4]-idx[4].start].cpu().numpy()
                            res_dict[layer][lb].append(point_feat)
                            visited = np.append(visited, i)
                    sample_dict[layer][lb] = np.delete(sample_dict[layer][lb], visited.astype(np.int64), axis=0)          
    for decoder in res_dict.keys():
        for lb in res_dict[decoder].keys():
            res_dict[decoder][lb] = np.array(res_dict[decoder][lb])

    return res_dict