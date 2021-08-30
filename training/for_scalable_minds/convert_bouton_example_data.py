import os
from glob import glob
import h5py
import napari

DATA_ROOT = '/g/rompani/pape/lgn/tranining_data/boutons/V4'


def check(ff):
    with h5py.File(ff, 'r') as f:
        raw = f['raw'][:]
        seg = f['labels/boutons'][:]
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(seg)


def check_all():
    files = glob(os.path.join(DATA_ROOT, "*.h5"))
    for ff in files:
        check(ff)


if __name__ == '__main__':
    check_all()
