import os

import numpy as np
import nifty.tools as nt
import z5py

from elf.segmentation.workflows import simple_multicut_workflow
from elf.segmentation.watershed import stacked_watershed
from mipnet.utils.prediction import predict_with_halo, normalize

ROOT = '/g/kreshuk/pape/Work/data/rompani/training_data/raw_data'


def predict_affs(raw):
    gpus = list(range(4))

    ckpt = '/g/kreshuk/pape/Work/mobie/lgn-em-datasets/training/neurons/networks/v2/Weights'
    inner_block_shape = [32, 128, 128]
    halo = [8, 64, 64]
    outer_block_shape = [ish + 2 * ha
                         for ish, ha in zip(inner_block_shape, halo)]

    affs = predict_with_halo(raw, ckpt, gpus,
                             inner_block_shape, outer_block_shape,
                             preprocess=normalize,
                             model_is_inferno=True)
    affs = affs[:3]
    return affs


def load_data(block_id):
    path = os.path.join(ROOT, 'train_data_%i.n5' % block_id)
    f = z5py.File(path)
    ds = f['raw']
    ds.n_threads = 8
    raw = ds[:]

    affs = predict_affs(raw)
    return raw, affs


def segment(affs, beta):
    boundaries = np.max(affs[1:], axis=0)
    ws, _ = stacked_watershed(boundaries, threshold=.5, sigma_seeds=1.)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    results = simple_multicut_workflow(affs,
                                       offsets=offsets,
                                       use_2dws=True,
                                       watershed=ws,
                                       multicut_solver='kernighan-lin',
                                       beta=beta,
                                       n_threads=8,
                                       weighting_scheme=None,
                                       return_intermediates=True)
    ws = results['watershed']
    node_labels = results['node_labels']
    return ws, node_labels


def make_data(block_id, beta):
    print("Run predictions ...")
    raw, affs = load_data(block_id)
    print("Run segmentation ...")
    seg, node_labels = segment(affs, beta)
    return raw, affs, seg, node_labels


def make_paintera(path):
    from paintera_tools import convert_to_paintera_format, downscale
    scale_factors = [[1, 2, 2], [1, 2, 2]]
    downscale(path, 'raw/s0', 'raw',
              scale_factors, scale_factors,
              tmp_folder='tmp_paintera_ds', target='local',
              max_jobs=8, resolution=[40, 10, 10])

    convert_to_paintera_format(path, 'raw', 'seg', 'paintera',
                               label_scale=0, resolution=[40, 10, 10],
                               target='local', tmp_folder='tmp_paintera',
                               max_jobs=8, max_threads=8,
                               assignment_path=path, assignment_key='node_labels')


def set_up_paintera(block_id, check):
    raw, boundaries, ws, node_labels = make_data(block_id, beta=.6)

    if check:
        import napari
        seg = nt.take(node_labels, ws)
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(raw)
            viewer.add_image(boundaries)
            viewer.add_labels(seg)
        return

    path = f'paintera_data/block{block_id}.n5'
    with z5py.File(path) as f:
        f.create_dataset('raw/s0', data=raw, compression='gzip', chunks=(32, 256, 256))
        ds = f.create_dataset('seg', data=ws, compression='gzip', chunks=(32, 256, 256))
        ds.attrs['maxId'] = int(ws.max())
        f.create_dataset('node_labels', data=node_labels)

    make_paintera(path)


if __name__ == '__main__':
    set_up_paintera(block_id=0, check=False)
