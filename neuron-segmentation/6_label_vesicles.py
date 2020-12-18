# connected components on vesicle pmap

import z5py
# FIXME this doe snot work fully, fix it and then implement in cluster-tools
from elf.wrapper import  ThresholdWrapper, RoiWrapper
from elf.parallel.label import label

from common import get_bounding_box, BLOCK_SHAPE


def label_vesicles():

    # TODO bounding box is too aggressive, but predictions is
    # also not good enough yet, need better training data
    threshold = 250

    path = './data.n5'
    with z5py.File(path, 'a') as f:
        ds = f['vesicles']
        ds = ThresholdWrapper(ds, threshold)

        out_key = 'segmentation/vesicles'
        ds_out = f.require_dataset(
            out_key,
            shape=ds.shape,
            compression='gzip',
            dtype='uint64',
            chunks = ds.chunks
        )

        bb = get_bounding_box(intersect_with_blocking=True)
        label(
            ds, ds_out,
            with_background=True,
            roi=bb,
            verbose=True,
            n_threads=32
        )


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
    ds = f['segmentation/vesicles']
    ds.n_threads = 8
    seg = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


if __name__ == '__main__':
    label_vesicles()
    # check_labels()
