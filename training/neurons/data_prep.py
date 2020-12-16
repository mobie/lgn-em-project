import os
from glob import glob

import numpy as np
import h5py
from scipy.io import loadmat
from skimage.transform import rescale
from tqdm import tqdm

from seg_utils import boundaries_to_seg, refine_seg

ROOT = '/g/kreshuk/data/helmstaedter/training_data/neurons/original'
ROOT_OUT = '/g/kreshuk/data/helmstaedter/training_data/neurons/rompani'


def convert(path, resize=False):
    x = loadmat(path)
    raw = x['raw'].T
    seg = x['target'].T

    seg_full = np.zeros(raw.shape, dtype='uint32')
    bb = tuple(slice(sh // 2 - ha // 2, sh // 2 + ha // 2) for sh, ha in zip(seg_full.shape, seg.shape))
    seg = boundaries_to_seg(raw[bb], seg)
    seg_full[bb] = seg

    if resize:
        raw = rescale(raw, (0.75,  1., 1.))
        seg_full = rescale(seg_full, (0.75,  1., 1.), order=0, preserve_range=True,
                           anti_aliasing=False).astype(seg.dtype)

    return raw, seg_full


def check_data(idd=1, resize=False):
    import napari
    path = os.path.join(ROOT, f'{idd}.mat')
    raw, seg = convert(path, resize=resize)
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(seg)


def check_refined_seg(idd=1):
    import napari
    path = os.path.join(ROOT, f'{idd}.mat')
    x = loadmat(path)
    raw = x['raw'].T
    seg = x['target'].T

    bd, ws, seeds, seg = refine_seg(raw, seg, return_intermediates=True, restrict_to_bb=False)
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw)
        v.add_image(bd)
        v.add_labels(ws)
        v.add_labels(seeds)
        v.add_labels(seg)


def convert_all(resize=True):
    os.makedirs(ROOT_OUT, exist_ok=True)
    all_files = glob(os.path.join(ROOT, '*.mat'))
    for path in tqdm(all_files):
        fname = os.path.split(path)[-1]
        raw, seg = convert(path, resize=resize)
        out_path = os.path.join(ROOT_OUT, fname.replace('.mat', '.h5'))
        with h5py.File(out_path, 'a') as f:
            f.create_dataset('raw', data=raw, compression='gzip')
            f.create_dataset('labels', data=seg, compression='gzip')


def refine_all(restrict_to_bb):
    all_files = glob(os.path.join(ROOT, '*.mat'))
    out_key = 'labels_postprocessed'
    if restrict_to_bb:
        out_key += '/restricted'
    else:
        out_key += '/full'
    for path in tqdm(all_files):
        fname = os.path.split(path)[-1]
        x = loadmat(path)
        raw = x['raw'].T
        seg = x['target'].T
        seg = refine_seg(raw, seg, restrict_to_bb=restrict_to_bb)
        out_path = os.path.join(ROOT_OUT, fname.replace('.mat', '.h5'))
        with h5py.File(out_path, 'a') as f:
            f.create_dataset(out_key, data=seg, compression='gzip')


if __name__ == '__main__':
    refine_all(restrict_to_bb=False)
    refine_all(restrict_to_bb=True)
    # check_refined_seg()
    # check_data(resize=True)
    # convert_all()
