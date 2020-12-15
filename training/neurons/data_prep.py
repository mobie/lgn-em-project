import os
from glob import glob

import numpy as np
import h5py
import vigra
from scipy.io import loadmat
from skimage.transform import rescale
from elf.segmentation.utils import normalize_input
from tqdm import tqdm

try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff

ROOT = '/g/kreshuk/data/helmstaedter/training_data/neurons/original'
ROOT_OUT = '/g/kreshuk/data/helmstaedter/training_data/neurons/rompani'


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


if __name__ == '__main__':
    check_data(resize=True)
    # convert_all()
