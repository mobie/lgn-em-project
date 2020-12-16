import os
import numpy as np
import vigra

from scipy.ndimage import binary_erosion
from elf.segmentation.features import (compute_rag,
                                       compute_boundary_mean_and_length,
                                       project_node_labels_to_pixels,
                                       compute_z_edge_mask)
from elf.segmentation.multicut import compute_edge_costs, multicut_kernighan_lin
from elf.segmentation.watershed import stacked_watershed
from elf.segmentation.utils import normalize_input, compute_maximum_label_overlap
from mipnet.utils.prediction import predict_with_halo, normalize

try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff


def get_prediction(raw, cache=False):
    model_path = '/g/kreshuk/pape/Work/mobie/lgn-em-datasets/training/neurons/networks/v1/Weights'
    tmp_path = './tmp_pred.npy'
    if os.path.exists(tmp_path) and cache:
        pred = np.load(tmp_path)
    else:
        pred = predict_with_halo(raw, model_path, gpus=[5, 6, 7],
                                 inner_block_shape=[32, 128, 128],
                                 outer_block_shape=[48, 192, 192],
                                 preprocess=normalize)
        pred = pred[1]
        if cache:
            np.save(tmp_path, pred)
    return pred


def refine_seg(raw, seeds, restrict_to_seeds=True, restrict_to_bb=False, return_intermediates=False):
    pred = get_prediction(raw, cache=False)

    n_threads = 1
    # make watershed
    ws, _ = stacked_watershed(pred, threshold=.5, sigma_seeds=1., n_threads=n_threads)
    rag = compute_rag(ws, n_threads=n_threads)
    edge_feats = compute_boundary_mean_and_length(rag, pred, n_threads=n_threads)
    edge_feats, edge_sizes = edge_feats[:, 0], edge_feats[:, 1]
    z_edges = compute_z_edge_mask(rag, ws)
    edge_costs = compute_edge_costs(edge_feats, beta=.4, weighting_scheme='xyz',
                                    edge_sizes=edge_sizes, z_edge_mask=z_edges)

    # make seeds and map them to edges
    bb = tuple(slice(sh // 2 - ha // 2, sh // 2 + ha // 2) for sh, ha in zip(pred.shape, seeds.shape))

    seeds[seeds < 0] = 0
    seeds = vigra.analysis.labelVolumeWithBackground(seeds.astype('uint32'))
    seed_ids = np.unique(seeds)
    seed_mask = binary_erosion(seeds, iterations=2)

    seeds_new = seeds.copy()
    seeds_new[~seed_mask] = 0
    seed_ids_new = np.unique(seeds_new)
    for seed_id in seed_ids:
        if seed_id in seed_ids_new:
            continue
        seeds_new[seeds == seed_id] = seed_id

    seeds_full = np.zeros(pred.shape, dtype=seeds.dtype)
    seeds_full[bb] = seeds
    seeds = seeds_full

    seed_labels = compute_maximum_label_overlap(ws, seeds, ignore_zeros=True)

    edge_ids = rag.uvIds()
    labels_u = seed_labels[edge_ids[:, 0]]
    labels_v = seed_labels[edge_ids[:, 1]]

    seed_mask = np.logical_and(labels_u != 0, labels_v != 0)
    same_seed = np.logical_and(seed_mask, labels_u == labels_v)
    diff_seed = np.logical_and(seed_mask, labels_u != labels_v)

    max_att = edge_costs.max() + .1
    max_rep = edge_costs.min() - .1
    edge_costs[same_seed] = max_att
    edge_costs[diff_seed] = max_rep

    # run multicut
    node_labels = multicut_kernighan_lin(rag, edge_costs)
    if restrict_to_seeds:
        seed_nodes = np.unique(node_labels[seed_labels > 0])
        node_labels[~np.isin(node_labels, seed_nodes)] = 0
        vigra.analysis.relabelConsecutive(node_labels, out=node_labels)

    seg = project_node_labels_to_pixels(rag, node_labels, n_threads=n_threads)

    if restrict_to_bb:
        bb_mask = np.zeros(seg.shape, dtype='bool')
        bb_mask[bb] = 1
        seg[~bb_mask] = 0

    if return_intermediates:
        return pred, ws, seeds, seg
    else:
        return seg


def update_hmap(raw, hmap, invert):
    if invert:
        intensities = normalize_input(raw.max() - raw)
    else:
        intensities = normalize_input(raw)
    return normalize_input(intensities * hmap)


def em_hmap(raw, sigma, sigma2=None, invert=True):
    """ This heightmap works well for 2d boundaries in EM.
    """
    hmap = normalize_input(ff.gaussianGradientMagnitude(raw, sigma))
    sigma2 = sigma if sigma2 is None else sigma2
    hmap = ff.hessianOfGaussianEigenvalues(hmap, sigma2)[..., 0]
    hmap = update_hmap(raw, hmap, invert=invert)
    return hmap


def boundaries_to_seg(raw, seg):
    hmap = np.zeros(raw.shape, dtype='float32')
    for z in range(hmap.shape[0]):
        hmap[z] = em_hmap(normalize_input(raw[z]), sigma=2)

    seg[seg == -1] = 0
    seg = seg.astype('uint32')

    seeds = vigra.analysis.labelVolumeWithBackground(seg)
    seg, _ = vigra.analysis.watershedsNew(hmap, seeds=seeds)

    return seg
