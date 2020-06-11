import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_sizes(table_path):
    table = pd.read_csv(table_path, sep='\t')
    sizes = table['n_pixels'].values
    return sizes


def to_diameter(sizes, resolution=0.02):
    diameter = 2 * (.75 * sizes / np.pi) ** (1. / 3)
    return diameter * resolution


def compute_size_histogram(table_path, out_path=None,
                           lower_percentile=0.5, upper_percentile=99.5):
    sizes = _load_sizes(table_path)
    if lower_percentile is not None:
        thresh_low = np.percentile(sizes, lower_percentile)
        sizes = sizes[sizes > thresh_low]
    if upper_percentile is not None:
        thresh_high = np.percentile(sizes, upper_percentile)
        sizes = sizes[sizes < thresh_high]

    diameter = to_diameter(sizes)

    fig, ax = plt.subplots(1)
    _, bins, _ = ax.hist(diameter, bins=24)
    ax.set_title("Histogram of bouton diameters")
    ax.set_xlabel('diameter [micrometer]')

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


if __name__ == '__main__':
    table_path = '../data/0.0.0/tables/sbem-adult-1-lgn-boutons/default.csv'
    out_path = '../bouton_size_histogram.png'
    # out_path = None
    compute_size_histogram(table_path, out_path)
