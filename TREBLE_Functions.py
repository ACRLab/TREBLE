"""
Created on 5/20/24

@author: belaaguilar
"""
import numpy as np
import pandas as pd


def split_with_overlap(vec,
                       seg_length,
                       overlap):
    starts = np.arange(0, len(vec), seg_length-overlap)
    ends = [min(start + seg_length, len(vec)) for start in starts]
    peaks = [vec[start:end].reset_index(drop=True) for start, end in zip(starts, ends)]
    return peaks


def get_windows(features,
                window_size=10,
                step_size=1,
                name=None):
    # Get windows
    peaks1 = split_with_overlap(features, window_size, window_size - step_size)

    # Clean and combine into matrix
    peaks1 = peaks1[:len(peaks1) - window_size]
    df_list = []
    for df in peaks1:
        ndf = df.unstack().to_frame().T
        ndf.columns = [f'{col[0]}_{col[1] + 1}' for col in ndf.columns]
        df_list.append(ndf.transpose())
    peaks1 = pd.concat(df_list, axis=1)

    if name is not None:
        peaks1.columns = [f"{name}_{i}" for i in range(peaks1.shape[1])]
    else:
        peaks1.columns = features.index[:peaks1.shape[1]]
    return peaks1


def bin_umap(layout, n_bins):
    n = n_bins

    # Split x into n bins
    step = (max(layout.iloc[:, 0]) - min(layout.iloc[:, 0])) / n
    x1 = np.arange(min(layout.iloc[:, 0]), max(layout.iloc[:, 0]) + step, step)

    xnew = [np.argmin(abs(x - x1)) + 1 for x in layout.iloc[:, 0]]
    layout['xnew'] = xnew

    # Split y into n bins
    step = step = (max(layout.iloc[:, 1]) - min(layout.iloc[:, 1])) / n
    y1 = np.arange(min(layout.iloc[:, 1]), max(layout.iloc[:, 1]) + step, step)

    ynew = [np.argmin(abs(y - y1)) + 1 for y in layout.iloc[:, 1]]
    layout['ynew'] = ynew

    # Paste xy to get unique bin combos (this will be input to sling shot as 'clusters')
    xy_new = [f"{x}_{y}" for x, y in zip(xnew, ynew)]
    layout['xy_new'] = xy_new

    # Convert coordinates to numeric, sort first
    m = set(xy_new)
    m = sorted(m, key=lambda v: int(v.split('_')[1]))
    m = sorted(m, key=lambda v: int(v.split('_')[0]))

    # Get names
    names_m = {name: i + 1 for i, name in enumerate(m)}
    names_xy_new = [names_m[name] for name in xy_new]

    # Get vector of coords
    layout["coords"] = names_xy_new

    # Return
    l = {'layout': layout, 'new_coords': xy_new}

    return l


if __name__ == '__main__':
    pass
