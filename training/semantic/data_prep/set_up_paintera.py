import os

import z5py
from skimage.measure import label

from mipnet.utils.prediction import predict_with_halo, normalize

ROOT = '/g/kreshuk/pape/Work/data/rompani/training_data/raw_data'


def predict_ves(raw):
    gpus = list(range(4))

    ckpt = '/g/kreshuk/pape/Work/mobie/lgn-em-datasets/training/semantic/networks/v1/Weights'
    inner_block_shape = [32, 128, 128]
    halo = [8, 64, 64]
    outer_block_shape = [ish + 2 * ha
                         for ish, ha in zip(inner_block_shape, halo)]

    ves = predict_with_halo(raw, ckpt, gpus,
                            inner_block_shape, outer_block_shape,
                            preprocess=normalize,
                            model_is_inferno=True)
    ves = ves[:3]
    return ves


def load_data(block_id):
    path = os.path.join(ROOT, 'train_data_%i.n5' % block_id)
    f = z5py.File(path)
    ds = f['raw']
    ds.n_threads = 8
    raw = ds[:]

    ves = predict_ves(raw)
    ves = ves[1]
    return raw, ves


def segment(ves, threshold):
    binary = ves > threshold
    seg = label(binary)
    seg += 1
    return seg


def make_data(block_id, threshold):
    print("Run predictions ...")
    raw, ves = load_data(block_id)
    print("Run segmentation ...")
    seg = segment(ves, threshold)
    return raw, ves, seg


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
                               max_jobs=8, max_threads=8)


def set_up_paintera(block_id, check):
    raw, ves, seg = make_data(block_id, threshold=.5)

    if check:
        import napari
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(raw)
            viewer.add_image(ves)
            viewer.add_labels(seg)
        return

    path = f'paintera_data/block{block_id}.n5'
    with z5py.File(path) as f:
        f.create_dataset('raw/s0', data=raw, compression='gzip', chunks=(32, 256, 256))
        ds = f.create_dataset('seg', data=seg, compression='gzip', chunks=(32, 256, 256))
        ds.attrs['maxId'] = int(seg.max())

    make_paintera(path)


if __name__ == '__main__':
    set_up_paintera(block_id=0, check=False)
