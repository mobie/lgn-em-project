import h5py
import numpy as np
import napari
import z5py

from pybdv.metadata import get_resolution

HALO = [784, 784]
COORDINATES = [
    (295.9188628431715, 241.35664615304347, 77.80049931433228),
    (129.7387499739823, 141.67135557401096, 41.23916527933145),
    (258.75709143856176, 173.21744418844247, 16.517148949803328),
    (271.0004488975969, 167.93485288980978, 16.517148949803328),
    (310.3066877962358, 209.89255655695754, 50.91518782051068),
    (334.6262736647317, 308.7590046671666, 92.87863012198444)
]
OFFSETS = [
    [-2, 0, 4],
    [0, 1],
    [1],
    [1],
    [1],
    [1]
]

PATH = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
XML_PATH = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.xml'


def draw_mask():
    path = '/g/kreshuk/pape/Work/data/rompani/neuron_training_data/defects.h5'
    with h5py.File(path, 'r') as f:
        raw = f['defect_sections/raw'][:]
        if 'defect_sections/mask' in f:
            seg = f['defect_sections/mask'][:]
        else:
            seg = np.zeros(raw.shape, dtype='uint8')

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)

    seg = viewer.layers['seg'].data
    with h5py.File(path, 'a') as f:
        ds = f.require_dataset('defect_sections/mask', dtype=seg.dtype, shape=seg.shape, compression='gzip')
        ds[:] = seg


def extract_defects(view=False):
    resolution = get_resolution(
        XML_PATH, setup_id=0
    )

    defect_raw = []
    defect_id = 0
    for coord, offsets in zip(COORDINATES, OFFSETS):
        coord = coord[::-1]
        coord = [int(co / re) for co, re in zip(coord, resolution)]
        for off in offsets:
            bb = (slice(coord[0] + off, coord[0] + off + 1),) + tuple(slice(co - ha, co + ha)
                                                                      for co, ha in zip(coord[1:], HALO))
            with z5py.File(PATH, 'r') as f:
                ds = f['setup0/timepoint0/s0']
                raw = ds[bb]

            if view:
                with napari.gui_qt():
                    viewer = napari.Viewer()
                    viewer.add_image(raw)
            defect_id += 1

        defect_raw.append(raw)

    defect_raw = np.concatenate(defect_raw, axis=0)
    out_path = '/g/kreshuk/pape/Work/data/rompani/neuron_training_data/defects.h5'
    with h5py.File(out_path, 'a') as f:
        f.create_dataset('defect_sections/raw', data=defect_raw, compression='gzip')


def check():
    path = '/g/kreshuk/pape/Work/data/rompani/neuron_training_data/defects.h5'
    with h5py.File(path, 'r') as f:
        raw = f['defect_sections/raw'][:]
        mask = f['defect_sections/mask'][:]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(mask)


if __name__ == '__main__':
    # extract_defects(view=False)
    # draw_mask()
    check()
