import numpy as np
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def scaling(dataset, sigma=0.1):
    factor = np.random.normal(loc=1., scale=sigma, size=(
        dataset.shape[0], dataset.shape[2]))
    data_scaled = np.multiply(dataset, factor[:, np.newaxis, :])
    return data_scaled

def rotation(dataset):
    flip = np.random.choice([-1, 1], size=(dataset.shape[0], dataset.shape[2]))
    rotate_axis = np.arange(dataset.shape[2])
    np.random.shuffle(rotate_axis)
    data_rotation = flip[:, np.newaxis, :] * dataset[:, :, rotate_axis]
    return data_rotation

def permutation(dataset, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(dataset.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(dataset.shape[0]))
    data_permute = np.zeros_like(dataset)
    for i, pat in enumerate(dataset):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(
                    dataset.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            data_permute[i] = pat[warp]
        else:
            data_permute[i] = pat
    return data_permute

def magnitude_warp(dataset, sigma=0.2, knot=4):
    orig_steps = np.arange(dataset.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(
        dataset.shape[0], knot + 2, dataset.shape[2]))
    warp_steps = (np.ones(
        (dataset.shape[2], 1)) * (np.linspace(0, dataset.shape[1] - 1., num=knot + 2))).T
    data_m_Warp = np.zeros_like(dataset)
    for i, pat in enumerate(dataset):
        warper = np.array([CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in
                           range(dataset.shape[2])]).T
        data_m_Warp[i] = pat * warper
    return data_m_Warp

def time_warp(dataset, sigma=0.2, knot=4):
    orig_steps = np.arange(dataset.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(
        dataset.shape[0], knot + 2, dataset.shape[2]))
    warp_steps = (np.ones(
        (dataset.shape[2], 1)) * (np.linspace(0, dataset.shape[1] - 1., num=knot + 2))).T
    data_t_Warp = np.zeros_like(dataset)
    for i, pat in enumerate(dataset):
        for dim in range(dataset.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(
                orig_steps)
            scale = (dataset.shape[1] - 1) / time_warp[-1]
            data_t_Warp[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, dataset.shape[1] - 1),
                                               pat[:, dim]).T
    return data_t_Warp

def window_slice(dataset, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio * dataset.shape[1]).astype(int)
    if target_len >= dataset.shape[1]:
        return dataset
    starts = np.random.randint(
        low=0, high=dataset.shape[1] - target_len, size=(dataset.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    data_w_Slice = np.zeros_like(dataset)
    for i, pat in enumerate(dataset):
        for dim in range(dataset.shape[2]):
            data_w_Slice[i, :, dim] = np.interp(np.linspace(0, target_len, num=dataset.shape[1]),
                                                np.arange(target_len), pat[starts[i]:ends[i], dim]).T
    return data_w_Slice

def window_warp(dataset, window_ratio=0.1, scales=[0.5, 2.]):
    warp_scales = np.random.choice(scales, dataset.shape[0])
    warp_size = np.ceil(window_ratio * dataset.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
    window_starts = np.random.randint(low=1, high=dataset.shape[1] - warp_size - 1,
                                      size=(dataset.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
    data_w_Warp = np.zeros_like(dataset)
    for i, pat in enumerate(dataset):
        for dim in range(dataset.shape[2]):
            start_seg = pat[:window_starts[i], dim]
            window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])),
                                   window_steps, pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            data_w_Warp[i, :, dim] = np.interp(np.arange(dataset.shape[1]),
                                               np.linspace(0, dataset.shape[1] - 1., num=warped.size), warped).T
    return data_w_Warp

def augment_dataset(dataset):
    data_scaled = scaling(dataset)
    data_rotation = rotation(dataset)
    data_permute = permutation(dataset)
    data_m_Warp = magnitude_warp(dataset)
    data_t_Warp = time_warp(dataset)
    data_w_Slice = window_slice(dataset)
    data_w_Warp = window_warp(dataset)

    augmented_dataset = np.concatenate(
        [dataset, data_scaled, data_rotation, data_permute, data_m_Warp, data_t_Warp, data_w_Slice, data_w_Warp])

    return augmented_dataset
