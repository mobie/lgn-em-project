import h5py
import z5py

from mipnet.models.unet import UNet2d
from mipnet.utils.prediction import predict_with_halo
from elf.segmentation.utils import normalize_input

PATH = '/g/rompani/'
MODEL_PATH = '/g/kreshuk/pape/Work/my_projects'


def run_prediction(model, raw):
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


if __name__ == '__main__':
    make_test_data()
