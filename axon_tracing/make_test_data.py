import os
import h5py
import z5py

from skimage.transform import resize
from mipnet.models.unet import UNet2d
from mipnet.utils.prediction import predict_with_halo
from elf.segmentation.utils import normalize_input

PATH = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
BOUTON_PATH = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-boutons.n5'
MODEL_PATH = os.path.join('/g/kreshuk/pape/Work/my_projects/super_embeddings/experiments/lgn',
                          'proofread/z100/imws_saved_state.torch')


def run_prediction(raw):
    model_kwargs = dict(
        in_channels=1, out_channels=8,
        initial_features=32, depth=4,
        pad_convs=True, activation='Sigmoid'
    )
    ckpt = (
        UNet2d,
        model_kwargs,
        MODEL_PATH
    )

    block_shape = (1,) + raw.shape[1:]

    def preprocess(inp):
        inp = normalize_input(inp)
        return inp[0]

    def postprocess(inp):
        return inp[:, None]

    gpus = [0]
    pred = predict_with_halo(raw, ckpt, gpus=gpus, inner_block_shape=block_shape,
                             outer_block_shape=block_shape, preprocess=preprocess,
                             postprocess=postprocess)
    return pred


def make_test_data(halo=[50, 512, 512]):
    with z5py.File(PATH, 'r') as f:
        ds = f['setup0/timepoint0/s0']
        shape = ds.shape

        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]

    pred = run_prediction(raw)

    with h5py.File('./test_data.h5', 'a') as f:
        f.create_dataset('raw', data=raw, compression='gzip')
        f.create_dataset('affinities', data=pred, compression='gzip')


def add_boutons(scale_factor=[1, 2, 2], halo=[50, 512, 512]):
    with z5py.File(BOUTON_PATH, 'r') as f:
        ds = f['setup0/timepoint0/s0']
        shape = ds.shape

        bb = tuple(slice(sh // 2 - ha // sf, sh // 2 + ha // sf)
                   for sh, ha, sf in zip(shape, halo, scale_factor))
        seg = ds[bb]

    target_shape = tuple(sh * sf for sh, sf in zip(seg.shape, scale_factor))
    seg = resize(seg, target_shape, order=0, preserve_range=True).astype(seg.dtype)

    with h5py.File('./test_data.h5', 'a') as f:
        f.create_dataset('boutons', data=seg, compression='gzip')


if __name__ == '__main__':
    # make_test_data()
    add_boutons()
