import os
import json

import luigi
import numpy as np
import z5py

from cluster_tools.lifted_multicut import LiftedMulticutWorkflow
from cluster_tools.write import WriteLocal, WriteSlurm
from common import get_bounding_box, BLOCK_SHAPE

PATH = './data.n5'
TARGET = 'local'
MAX_JOBS = 16
TMP_FOLDER = './tmp_lifted_multicut'
CONFIG_FOLDER = os.path.join(TMP_FOLDER, 'configs')


def _make_global_config(configs):
    os.makedirs(CONFIG_FOLDER, exist_ok=True)

    roi_begin, roi_end = get_bounding_box(return_as_lists=True)

    conf = configs['global']
    conf.update({
        'block_shape': BLOCK_SHAPE,
        'roi_begin': roi_begin,
        'roi_end': roi_end
    })

    with open(os.path.join(CONFIG_FOLDER, 'global.config'), 'w') as f:
        json.dump(conf, f)


def solve_lmc():
    task = LiftedMulticutWorkflow
    configs = task.get_config()
    _make_global_config(configs)

    t = task(tmp_folder=TMP_FOLDER, config_dir=CONFIG_FOLDER,
             target=TARGET, max_jobs=MAX_JOBS,
             problem_path=PATH, n_scales=1,
             lifted_prefix='boutons',
             assignment_path=PATH, assignment_key='node_labels/lmc')
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def merge_bouton_labels():
    out_key = 'node_labels/boutons'
    with z5py.File(PATH, 'a') as f:
        if out_key in f:
            return

        node_labels = f['node_labels/lmc'][:]
        bouton_labels = f['node_overlaps'][:]
        assert node_labels.shape == bouton_labels.shape

        merged_labels = np.zeros_like(node_labels)
        bouton_ids = np.unique(bouton_labels)
        for bid in bouton_ids[1:]:
            node_values = np.unique(node_labels[bouton_labels == bid])
            merged_labels[np.isin(node_labels, node_values)] = bid

        f.create_dataset(out_key, data=merged_labels, chunks=merged_labels.shape)


def write_segmentation():
    task = WriteLocal if TARGET == 'local' else WriteSlurm

    t = task(tmp_folder=TMP_FOLDER, config_dir=CONFIG_FOLDER,
             max_jobs=MAX_JOBS,
             input_path=PATH, input_key='watershed',
             output_path=PATH, output_key='lifted_multicut',
             assignment_path=PATH, assignment_key='node_labels/boutons',
             identifier='')
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def run_lmc():
    solve_lmc()
    merge_bouton_labels()
    write_segmentation()


def check_lmc():
    import napari
    bb = get_bounding_box(scale=0)

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s1']
    ds.n_threads = 8
    raw = ds[bb]

    path = './data.n5'
    f = z5py.File(path, 'r')
    ds = f['lifted_multicut']
    ds.n_threads = 8
    seg = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


if __name__ == '__main__':
    run_lmc()
    # check_lmc()
