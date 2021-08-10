import argparse
import json
import os

import h5py
import numpy as np
import z5py
import elf.segmentation as eseg
from skimage.transform import resize


MODEL_PATH = '../../training/neurons/networks/v1/Weights'
POSITIONS = [
    "[239.10152517632784,178.5252449269954,20.687220124184286]",
    "[232.99255836936175,178.68605970682748,20.521753592280277]",
    "[220.01249384420944,172.2340655804881,20.521753592280266]",
    "[207.44771381666328,159.29009167425187,19.91725674105999]",
    "[135.20085158147674,141.61755033363426,15.858557291516016]",
    "[115.98011685727037,140.0018695817759,15.858557291516014]"
]
NEW_POSITIONS = [
    "[157.94482772391055,150.66224714510327,51.19999999999992]",
    "[148.4514211450999,151.33652425312084,51.20555163508739]",
    "[149.1307794076376,141.1183531285991,50.71289208318241]",
    "[138.8755539533484,121.19535996485543,51.307762494108054]"
]

EXTRA_POS = "[107.13705156898774,107.70803124959056,14.835337077736645]"


def to_bb(coordinate, resolution, halo):
    coord = [int(coord / res) for coord, res in zip(coordinate, resolution)]
    bb = tuple(
        slice(co - ha, co + ha) for co, ha in zip(coord, halo)
    )
    return bb


def predict(raw, name):
    cache_path = './prediction_cache'
    os.makedirs(cache_path, exist_ok=True)
    cache_path = os.path.join(cache_path, f'{name}.h5')
    if os.path.exists(cache_path):
        with h5py.File(cache_path, 'r') as f:
            pred = f['data'][:]
        return pred

    from mipnet.utils.prediction import predict_with_halo, normalize
    inner_block_shape = (24, 196, 196)
    outer_block_shape = (32, 256, 256)
    pred = predict_with_halo(raw, MODEL_PATH, gpus=[0, 1],
                             inner_block_shape=inner_block_shape,
                             outer_block_shape=outer_block_shape,
                             preprocess=normalize)
    pred = np.max(pred[:3], axis=0)
    assert pred.ndim == 3
    with h5py.File(cache_path, 'a') as f:
        f.create_dataset('data', data=pred, compression='gzip')
    return pred


def segment(raw, name, betas):
    pred = predict(raw, name)
    ws, max_id = eseg.watershed.stacked_watershed(pred, threshold=0.2, sigma_seeds=2.)
    rag = eseg.features.compute_rag(ws, max_id + 1)
    feats = eseg.features.compute_boundary_mean_and_length(rag, pred)
    costs, sizes = feats[:, 0], feats[:, 1]
    z_edges = eseg.features.compute_z_edge_mask(rag, ws)

    segs = {}
    for beta in betas:
        costs = eseg.multicut.compute_edge_costs(costs, edge_sizes=sizes,
                                                 z_edge_mask=z_edges, beta=beta)
        seg = eseg.multicut.multicut_kernighan_lin(rag, costs)
        seg = eseg.features.project_node_labels_to_pixels(rag, seg)
        segs[str(beta)] = seg
    return segs, ws


def create_example_data(args):
    out_folder = './training_data2'
    os.makedirs(out_folder, exist_ok=True)

    raw_path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    raw_key = 'setup0/timepoint0/s0'

    coordinate = json.loads(args.coordinate)[::-1]  # we reverse the coordinate because it comes from MoBIE
    # get the coordinate and load the raw data
    resolution = [0.04, 0.01, 0.01]
    bb = to_bb(coordinate, resolution, args.halo)
    f = z5py.File(raw_path, 'r')
    ds = f[raw_key]
    raw = ds[bb]

    # load the bouton segmentation
    scale_factor = [1, 2, 2]
    bb1 = tuple(
        slice(b.start // sf, b.stop // sf) for b, sf in zip(bb, scale_factor)
    )
    bouton_path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-boutons.n5'
    bouton_key = 'setup0/timepoint0/s0'
    f = z5py.File(bouton_path, 'r')
    ds = f[bouton_key]
    boutons = ds[bb1]
    boutons = resize(boutons, raw.shape, order=0, preserve_range=True).astype(boutons.dtype)
    assert boutons.shape == raw.shape

    # predict neurite instance segmentation
    axes = ['z', 'y', 'x']
    loc = ''.join(f'{ax}{b.start}' for ax, b in zip(axes, bb)) + '-' +\
        ''.join(f'{ax}{b.stop}' for ax, b in zip(axes, bb))
    betas = [0.5, 0.6, 0.7]
    neuron_segmentations, ws = segment(raw, loc, betas)

    # clefts (placeholder!)
    # clefts = ''

    if args.debug:
        import napari
        with napari.gui_qt():
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(boutons)
            for name, seg in neuron_segmentations.items():
                v.add_labels(seg, name=name)
    else:
        name = f'training_data_{loc}.h5'
        out_path = os.path.join(out_folder, name)
        if os.path.exists(out_path):
            print("Dataset", out_path, "exists already")
            return
        with h5py.File(out_path, 'a') as f_out:
            f_out.attrs['coordinate'] = coordinate
            f_out.attrs['start'] = [b.start for b in bb]
            f_out.create_dataset('raw', data=raw, compression='gzip')
            f_out.create_dataset('boutons', data=boutons, compression='gzip')
            f_out.create_dataset('neurites/watershed', data=ws, compression='gzip')
            for beta in betas:
                f_out.create_dataset(f'neurites/seg_{beta}',
                                     data=neuron_segmentations[str(beta)], compression='gzip')


def main(args):
    if args.coordinate is None:
        # for pos in POSITIONS:
        for pos in NEW_POSITIONS:
            args.coordinate = pos
            create_example_data(args)
    else:
        create_example_data(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coordinate', type=str, default=None)
    parser.add_argument('-d', '--debug', default=0, type=int)
    parser.add_argument('--halo', type=int, nargs='+', default=[32, 256, 256])
    args = parser.parse_args()
    main(args)
