import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_sizes(table_path):
    sizes_tmp = '_sizes.npy'
    if os.path.exists(sizes_tmp):
        sizes = np.load(sizes_tmp)
    else:
        table = pd.read_csv(table_path, sep='\t')
        sizes = table['n_pixels'].values
        np.save(sizes_tmp, sizes)
    return sizes


def to_volume(sizes, resolution=[]):
    return sizes


def compute_size_histogram(table_path, out_path,
                           lower_percentile=1, upper_percentile=99):
    sizes = _load_sizes(table_path)
    thresh_low = np.percentile(sizes, lower_percentile)
    thresh_high = np.percentile(sizes, upper_percentile)

    sizes = sizes[sizes > thresh_low]
    sizes = sizes[sizes < thresh_high]

    volumes = to_volume(sizes)

    fig, ax = plt.subplots(3)
    _, bins, _ = ax[0].hist(volumes)
    ax[0].xlabel = 'bouton-volume [cubic microns]'

    size_range = bins[1]
    volumes_small = volumes[volumes < size_range]
    volumes_large = volumes[volumes > size_range]

    ax[1].hist(volumes_small)
    ax[1].xlabel = 'bouton-volume [cubic microns]'

    ax[2].hist(volumes_large)
    ax[2].xlabel = 'bouton-volume [cubic microns]'

    plt.savefig(out_path)


if __name__ == '__main__':
    table_path = '../data/0.0.0/tables/sbem-adult-1-lgn-boutons/default.csv'
    out_path = '../size_histogram.png'
    compute_size_histogram(table_path, out_path)
