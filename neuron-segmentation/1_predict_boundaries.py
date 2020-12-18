import os

import z5py
import numpy as np
from mipnet.models.unet import UNet2d
from mipnet.utils.prediction import predict_with_halo

from elf.segmentation.utils import normalize_input
from elf.wrapper import RoiWrapper

# if we need much larger volumes, we need to use the inference workflow from cluster_tools,
# but it would need to be adapted quite a bit to support the different pre/postprocessing
# import luigi
# from cluster_tools.inference import InferenceLocal, InferenceSlurm

from common import get_bounding_box, BLOCK_SHAPE

MODEL_PATH = os.path.join('/g/kreshuk/pape/Work/my_projects/super_embeddings/experiments/lgn/models_supervised-2d',
                          'lr0.0001_use-affs1.state')


def predict_boundaries(gpus):
    model_kwargs = dict(
        in_channels=1, out_channels=8,
        initial_features=32, depth=4,
        pad_convs=True, activation='Sigmoid'
    )
    ckpt = (
        UNet2d,
        model_kwargs,
        MODEL_PATH,
        'model'
    )

    inner_block_shape = (1, 1024, 1024)
    halo = (0, 64, 64)
    outer_block_shape = tuple(ish + 2 * ha for ish, ha in zip(inner_block_shape, halo))

    def preprocess(inp):
        inp = normalize_input(inp)
        return inp[0]

    def to_boundaries(pred):
        # TODO the proper way would be to also mirror affinities according to their offset
        # to boundaries
        pred = np.max(pred[:4], axis=0)
        pred = normalize_input(pred)
        return pred[None, None]

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f_in = z5py.File(path, 'r')
    ds_raw = f_in['setup0/timepoint0/s0']

    # path for temporary data
    out_path = './data.n5'
    f_out = z5py.File(out_path, 'a')
    ds_pred = f_out.require_dataset('boundaries', shape=ds_raw.shape, dtype='float32', compression='gzip',
                                    chunks=tuple(BLOCK_SHAPE))

    bb = get_bounding_box(intersect_with_blocking=True)

    ds_raw = RoiWrapper(ds_raw, bb)
    ds_pred = RoiWrapper(ds_pred, bb)

    predict_with_halo(ds_raw, ckpt, gpus=gpus, inner_block_shape=inner_block_shape,
                      outer_block_shape=outer_block_shape, preprocess=preprocess,
                      postprocess=to_boundaries, output=ds_pred)


def check_predictions():
    import napari
    bb = get_bounding_box(scale=0)

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s0']
    ds.n_threads = 8
    raw = ds[bb]

    path = './data.n5'
    f = z5py.File(path, 'r')
    ds = f['boundaries']
    ds.n_threads = 8
    pred = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_image(pred)


if __name__ == '__main__':
    gpus = list(range(8))
    predict_boundaries(gpus)
    # check_predictions()
