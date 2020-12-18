# intersect neuron and vesicle segmentation to get boutons; apply size filter

from functools import partial

import numpy as np
import z5py
import nifty.tools as nt
from tqdm import tqdm

from concurrent import futures
from common import get_bounding_box, BLOCK_SHAPE


def intersect_segmentation(
    block_id,
    seg_a,
    seg_b,
    seg_out,
    blocking
):
    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

    sa = seg_a[bb]
    sb = seg_b[bb]

    seg = sa + sb
    mask = np.logical_and(sa != 0, sb != 0)
    seg[~mask] = 0

    seg_out[bb] = seg


def segment_boutons():
    path = './data.n5'
    with z5py.File(path, 'a') as f:
        seg_neurons = f['segmentation/multicut']
        seg_ves = f['segmentation/vesicles']

        ds_out = f.require_dataset(
            'segmentation/boutons',
            shape=seg_ves.shape,
            chunks=tuple(BLOCK_SHAPE),
            dtype='uint64',
            compression='gzip'
        )

        roi_begin, roi_end = get_bounding_box(intersect_with_blocking=True,
                                              return_as_lists=True)
        blocking = nt.blocking(roi_begin, roi_end, BLOCK_SHAPE)
        n_blocks = blocking.numberOfBlocks

        n_threads = 32
        func = partial(
            intersect_segmentation,
            seg_a=seg_neurons,
            seg_b=seg_ves,
            seg_out=ds_out,
            blocking=blocking
        )

        with futures.ThreadPoolExecutor(n_threads) as tp:
            list(tqdm(tp.map(func, range(n_blocks)), total=n_blocks))

    # TODO apply size filter


def check_labels():
    import napari
    bb = get_bounding_box(scale=0)

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s0']
    ds.n_threads = 8
    raw = ds[bb]

    path = './data.n5'
    f = z5py.File(path, 'r')
    ds = f['segmentation/boutons']
    ds.n_threads = 8
    seg = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


if __name__ == '__main__':
    # segment_boutons()
    check_labels()
