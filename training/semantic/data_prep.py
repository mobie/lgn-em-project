import os
from glob import glob

import numpy as np
import h5py
from skimage.transform import rescale
from tqdm import tqdm

ROOT = '/g/kreshuk/data/helmstaedter/training_data/semantic/original'
ROOT_OUT = '/g/kreshuk/data/helmstaedter/training_data/semantic/rompani'


def convert(path, resize=False):
    with h5py.File(path, 'r') as f:
        raw = f['em'][:]
        seg = f['label'][:].astype('uint32')
    # map 0 to background
    seg -= 1

    seg_full = -1 * np.ones(raw.shape, dtype='int32')
    bb = tuple(slice(sh // 2 - ha // 2, sh // 2 + ha // 2) for sh, ha in zip(seg_full.shape, seg.shape))
    seg_full[bb] = seg

    if resize:
        raw = rescale(raw, (0.75,  1., 1.))
        seg_full = rescale(seg_full, (0.75,  1., 1.), order=0, preserve_range=True,
                           anti_aliasing=False).astype(seg.dtype)

    return raw, seg_full


def check_data(idd=1, resize=False):
    import napari
    path = os.path.join(ROOT, f'cube-{idd}.hdf5')
    raw, seg = convert(path, resize=resize)
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(seg)


def convert_all(resize=True):
    os.makedirs(ROOT_OUT, exist_ok=True)
    all_files = glob(os.path.join(ROOT, '*.hdf5'))
    for path in tqdm(all_files):
        fname = os.path.split(path)[-1]
        raw, seg = convert(path, resize=resize)
        out_path = os.path.join(ROOT_OUT, fname.replace('.hdf5', '.h5'))
        with h5py.File(out_path, 'a') as f:
            f.create_dataset('raw', data=raw, compression='gzip')
            f.create_dataset('labels', data=seg, compression='gzip')


if __name__ == '__main__':
    check_data(idd=2, resize=False)
    # convert_all()
