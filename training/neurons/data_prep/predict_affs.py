import os
import torch
import z5py

from mipnet.models.unet import AnisotropicUNet
from mipnet.utils.prediction import predict_with_halo, normalize

DEVICE = torch.device('cuda')
OFFSETS = [
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    [-2, 0, 0], [0, -3, 0], [0, 0, -3],
    [-3, 0, 0], [0, -9, 0], [0, 0, -9],
    [-4, 0, 0], [0, -18, 0], [0, 0, -18]
]


def run_predction():
    gpus = [0, 1, 2, 3]

    f = z5py.File('../training_data/train_data_5.n5', 'a')
    ds_raw = f['raw']

    scale_factors = [
        [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]
    ]
    kwargs = dict(scale_factors=scale_factors,
                  in_channels=1, out_channels=len(OFFSETS),
                  initial_features=32, gain=2,
                  pad_convs=True)

    # This is the unsupervised pretrained model
    state_path = '../models_unsupervised-3d/lr0.0001_use-affs1_weight1.state'
    ckpt = (
        AnisotropicUNet,
        kwargs,
        state_path,
        'model'
    )

    name = os.path.split(state_path)[-1]
    ds_out = f.create_dataset(f'predictions_3d/{name}',
                              shape=(len(OFFSETS),) + ds_raw.shape,
                              chunks=(1, 32, 256, 256),
                              compression='gzip',
                              dtype='float32')

    inner_block_shape = [32, 256, 256]
    halo = [8, 32, 32]
    outer_block_shape = [ish + 2 * ha
                         for ish, ha in zip(inner_block_shape, halo)]

    predict_with_halo(ds_raw, ckpt, gpus,
                      inner_block_shape, outer_block_shape,
                      preprocess=normalize,
                      output=ds_out,
                      model_is_inferno=False)


def run_predction_segem():
    gpus = [5, 6, 7]

    f = z5py.File('../training_data/train_data_5.n5', 'a')
    ds_raw = f['raw']

    ckpt = '/g/kreshuk/pape/Work/mobie/lgn-em-datasets/training/neurons/networks/v4/Weights'
    name = 'segemV4'
    ds_out = f.require_dataset(f'predictions_3d/{name}',
                               shape=(9,) + ds_raw.shape,
                               chunks=(1, 32, 128, 128),
                               compression='gzip',
                               dtype='float32')

    inner_block_shape = [32, 128, 128]
    halo = [8, 64, 64]
    outer_block_shape = [ish + 2 * ha
                         for ish, ha in zip(inner_block_shape, halo)]

    predict_with_halo(ds_raw, ckpt, gpus,
                      inner_block_shape, outer_block_shape,
                      preprocess=normalize,
                      output=ds_out,
                      model_is_inferno=True)


if __name__ == '__main__':
    # run_predction()
    run_predction_segem()
