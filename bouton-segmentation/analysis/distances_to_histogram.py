import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_distances(dist_path='./distances.pkl'):
    with open(dist_path, 'rb') as f:
        distance_dict = pickle.load(f)
    distances = np.array(list(distance_dict.values()))
    return distances


def plot_single(distances):
    plt.hist(distances, bins=128)
    plt.xlabel('distance [micron]')
    plt.show()


def plot_multiple(distances, max_distances):
    fig, ax = plt.subplots(len(max_distances))
    for ii, max_dist in enumerate(max_distances):
        ax[ii].hist(distances[distances < max_dist], bins=64)
        ax[ii].set_title(f'max-distance={max_dist}')
    ax[-1].set_xlabel('distance [micron]')
    plt.show()


def distance_histogram(max_distances=None):
    # distances = 250 * np.random.rand(10000)
    distances = load_distances()
    if max_distances is None:
        plot_single(distances)
    else:
        plot_multiple(distances, max_distances)


if __name__ == '__main__':
    max_distances = [10, 25, 50, 100, 250]
    distance_histogram(max_distances)
