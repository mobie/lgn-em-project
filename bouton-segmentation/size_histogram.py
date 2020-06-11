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


def to_radius(sizes, resolution=.02):
    return sizes
    radius = (.75 * sizes / np.pi) ** (1. / 3)
    return radius


def compute_size_histogram(table_path, out_path=None,
                           lower_percentile=1, upper_percentile=99):
    sizes = _load_sizes(table_path)
    thresh_low = np.percentile(sizes, lower_percentile)
    thresh_high = np.percentile(sizes, upper_percentile)

    print("Initial number of objects:", len(sizes))
    sizes = sizes[sizes > thresh_low]
    sizes = sizes[sizes < thresh_high]
    print("After thresholds:", len(sizes))

    radii = to_radius(sizes)

    fig, ax = plt.subplots(3)
    _, bins, _ = ax[0].hist(radii, bins=32)
    ax[0].set_title("Full Size Histogram")

    size_range = bins[1]
    radii_small = radii[radii < size_range]
    radii_large = radii[radii > size_range]

    ax[1].hist(radii_small, bins=32)
    ax[1].set_title("Histogram of the first bin")

    ax[2].hist(radii_large, bins=32)
    ax[2].set_title("Histogram of the other bins")
    ax[2].set_xlabel('radius [micron meter]')

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


if __name__ == '__main__':
    table_path = '../data/0.0.0/tables/sbem-adult-1-lgn-boutons/default.csv'
    out_path = '../size_histogram.png'
    out_path = None
    compute_size_histogram(table_path, out_path)
