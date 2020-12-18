import os
import numpy as np
import pandas as pd
import z5py
from common import get_bounding_box


def prepare_table(make_full_table=False):
    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-boutons.n5'
    bb = get_bounding_box(scale=1)

    # this might be an issue if we get too large halos
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s0']
    ds.n_threads = 8
    seg = ds[bb]

    bouton_ids = np.unique(seg)[1:]

    # make the annotation table
    bouton_table_dir = '/g/rompani/lgn-em-datasets/data/0.0.0/tables/sbem-adult-1-lgn-boutons'
    if make_full_table:
        default_table = os.path.join(bouton_table_dir, 'default.csv')
        default_table = pd.read_csv(default_table, sep='\t')

        label_ids = default_table['label_id'].values

        n_labels = len(label_ids)
        annotations = np.array(n_labels * ["None"])
        annotations[bouton_ids] = "unlabeled"

    else:
        label_ids = bouton_ids
        annotations = np.array(len(label_ids) * ["unlabeled"])

    out_table = np.concatenate([
        label_ids[:, None],
        annotations[:, None]
    ], axis=1)
    out_table = pd.DataFrame(out_table, columns=['label_id', 'annotation'])

    out_table_path = os.path.join(bouton_table_dir, 'bouton_annotations_v1.csv')
    out_table.to_csv(out_table_path, sep='\t', index=False)


def check_boutons():
    import napari
    bb = get_bounding_box(scale=1)

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-boutons.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s0']
    ds.n_threads = 8
    seg = ds[bb]

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s1']
    ds.n_threads = 8
    raw = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


if __name__ == '__main__':
    prepare_table()
    # check_boutons()
