import os
import json

import numpy as np
import pandas as pd
import luigi
import z5py

from cluster_tools.lifted_features import LiftedFeaturesFromNodeLabelsWorkflow
from common import get_bounding_box, BLOCK_SHAPE

PATH = './data.n5'
TARGET = 'local'
MAX_JOBS = 16
TMP_FOLDER = './tmp_lifted_problem'
CONFIG_FOLDER = os.path.join(TMP_FOLDER, 'configs')


def load_bouton_annotations(table_path):
    bouton_table = pd.read_csv(table_path, sep='\t')
    label_ids, annotations = bouton_table['label_id'].values, bouton_table['annotation'].values
    return label_ids, annotations


def write_proofread_boutons(annotation_table, keep_annotations):
    out_key = 'boutons_proofread'
    f_out = z5py.File(PATH, 'a')
    if out_key in f_out:
        return

    bouton_ids, annotations = load_bouton_annotations(annotation_table)
    bouton_ids = bouton_ids[np.isin(annotations, keep_annotations)].astype('uint64')

    bb = get_bounding_box(scale=1, intersect_with_blocking=True)

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-boutons.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s0']
    ds.n_threads = 16

    # this might not work for larger cututouts and we need to use cluster tools
    seg = ds[bb]
    seg[~np.isin(seg, bouton_ids)] = 0

    ds_out = f_out.create_dataset(out_key, shape=ds.shape, chunks=ds.chunks,
                                  compression='gzip', dtype=seg.dtype)
    ds_out[bb] = seg


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


def lifted_problem():
    task = LiftedFeaturesFromNodeLabelsWorkflow
    configs = task.get_config()
    _make_global_config(configs)

    with z5py.File(PATH, 'r') as f:
        costs = f['s0/costs'][:]
        max_repulsive_cost = costs.min() - .1

    conf = configs['costs_from_node_labels']
    conf.update({'inter_label_cost': max_repulsive_cost})
    with open(os.path.join(CONFIG_FOLDER, 'costs_from_node_labels.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=TMP_FOLDER, config_dir=CONFIG_FOLDER,
             target=TARGET, max_jobs=MAX_JOBS,
             ws_path=PATH, ws_key='watershed',
             labels_path=PATH, labels_key='boutons_proofread',
             graph_path=PATH, graph_key='s0/graph',
             output_path=PATH, nh_out_key='s0/lifted_nh_boutons',
             feat_out_key='s0/lifted_costs_boutons',
             nh_graph_depth=6, mode='different',
             prefix='')
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def make_lifted_problem(annotation_table, keep_annotations):
    write_proofread_boutons(annotation_table, keep_annotations)
    lifted_problem()


def check_boutons():
    import napari
    bb = get_bounding_box(scale=1)

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s1']
    ds.n_threads = 8
    raw = ds[bb]

    path = './data.n5'
    f = z5py.File(path, 'r')
    ds = f['boutons_proofread']
    ds.n_threads = 8
    seg = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


if __name__ == '__main__':
    # NOTE annotation table and format might change
    annotation_table = os.path.join('/g/rompani/lgn-em-datasets/data/0.0.0/tables/sbem-adult-1-lgn-boutons',
                                    'bouton_annotations_v1_done.csv')
    keep_annotations = ['merge', 'fragment', 'bouton']
    make_lifted_problem(annotation_table, keep_annotations)

    # check_boutons()
