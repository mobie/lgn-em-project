import os
import json

import luigi
import z5py

from cluster_tools.multicut import MulticutWorkflow
from cluster_tools.write import WriteLocal, WriteSlurm
from common import get_bounding_box, BLOCK_SHAPE, get_halo

PATH = '/scratch/pape/lgn/data.n5'
TARGET = 'local'
MAX_JOBS = 16
TMP_FOLDER = '/scratch/pape/lgn/tmp_multicut'
CONFIG_FOLDER = os.path.join(TMP_FOLDER, 'configs')


def _make_global_config(configs, halo_name):
    os.makedirs(CONFIG_FOLDER, exist_ok=True)

    roi_begin, roi_end = get_bounding_box(return_as_lists=True, halo=get_halo(halo_name))

    conf = configs['global']
    conf.update({
        'block_shape': BLOCK_SHAPE,
        'roi_begin': roi_begin,
        'roi_end': roi_end
    })

    with open(os.path.join(CONFIG_FOLDER, 'global.config'), 'w') as f:
        json.dump(conf, f)


def solve_mc(halo_name):
    task = MulticutWorkflow
    configs = task.get_config()
    _make_global_config(configs, halo_name)

    t = task(tmp_folder=TMP_FOLDER, config_dir=CONFIG_FOLDER,
             target=TARGET, max_jobs=MAX_JOBS,
             problem_path=PATH, n_scales=1,
             assignment_path=PATH, assignment_key='node_labels/multicut')
    ret = luigi.build([t], local_scheduler=True)
    assert ret


# def merge_bouton_labels():
#     out_key = 'node_labels/boutons'
#     with z5py.File(PATH, 'a') as f:
#         if out_key in f:
#             return
#
#         node_labels = f['node_labels/lmc'][:]
#         bouton_labels = f['node_overlaps'][:]
#         assert node_labels.shape == bouton_labels.shape
#
#         merged_labels = np.zeros_like(node_labels)
#         bouton_ids = np.unique(bouton_labels)
#         for bid in bouton_ids[1:]:
#             node_values = np.unique(node_labels[bouton_labels == bid])
#             merged_labels[np.isin(node_labels, node_values)] = bid
#
#         f.create_dataset(out_key, data=merged_labels, chunks=merged_labels.shape)


def write_segmentation():
    task = WriteLocal if TARGET == 'local' else WriteSlurm

    node_labels = 'node_labels/multicut'
    t = task(tmp_folder=TMP_FOLDER, config_dir=CONFIG_FOLDER,
             max_jobs=MAX_JOBS,
             input_path=PATH, input_key='watershed',
             output_path=PATH, output_key='segmentation/multicut',
             assignment_path=PATH, assignment_key=node_labels,
             identifier='')
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def run_mc(halo_name):
    solve_mc(halo_name)
    # merge_bouton_labels()
    write_segmentation()


def check_mc():
    import napari
    bb = get_bounding_box(scale=0, halo=get_halo("check"))

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s0']
    ds.n_threads = 8
    raw = ds[bb]

    path = './data.n5'
    f = z5py.File(path, 'r')
    ds = f['segmentation/multicut']
    ds.n_threads = 8
    seg = ds[bb]

    # f = z5py.File(path, 'r')
    # ds = f['boutons_proofread']
    # ds.n_threads = 8
    # bb_boutons = get_bounding_box(scale=1)
    # boutons = ds[bb_boutons]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)
        # viewer.add_labels(boutons, scale=(1, 2, 2))


if __name__ == '__main__':
    run_mc(halo_name="large")
    check_mc()
