import os
import z5py
from pybdv.metadata import get_resolution

PATH = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
XML_PATH = PATH.replace('.n5', '.xml')

ROOT_OUT = '/g/kreshuk/pape/Work/data/rompani/neuron_training_data'

COORDINATES = [
    (218.49863205740556, 191.0356443719574, 51.199999999999996),
    (218.49863205740556, 191.0356443719574, 51.199999999999996),
    (269.772513233381, 130.86088996562503, 19.919380113308723),
    (305.22301761042115, 100.28898819616944, 19.91938011330871),
    (305.22301761042115, 100.28898819616944, 19.91938011330871),
    (369.24590458217625, 107.88298507704943, 19.17986461523694)
]


def get_bounding_box(bid, halo):
    resolution = get_resolution(XML_PATH, setup_id=0)

    coord = COORDINATES[bid][::-1]
    coord = [int(co / re) for co, re in zip(coord, resolution)]

    bb = tuple(slice(co - ha, co + ha) for co, ha in zip(coord, halo))
    return bb


def extract_block(bid, halo):
    out_path = os.path.join(ROOT_OUT, f'train_data_{bid}.n5')
    if os.path.exists(out_path):
        return

    bb = get_bounding_box(bid, halo)
    with z5py.File(PATH, 'r') as f:
        ds = f['setup0/timepoint0/s0']
        ds.n_threads = 8
        data = ds[bb]

    with z5py.File(out_path, 'a') as f:
        ds = f.create_dataset('raw', shape=data.shape, compression='gzip',
                              dtype=data.dtype, chunks=(32, 128, 128))
        ds.n_threads = 8
        ds[:] = data


def extract_all():
    halo = [64, 512, 512]
    for bid in range(len(COORDINATES)):
        extract_block(bid, halo)


if __name__ == '__main__':
    extract_all()
