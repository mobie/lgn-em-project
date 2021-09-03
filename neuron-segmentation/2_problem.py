import os
import json

import luigi
import z5py
from cluster_tools.watershed import WatershedWorkflow
from cluster_tools.workflows import ProblemWorkflow
from common import BLOCK_SHAPE, get_bounding_box, get_halo

PATH = '/scratch/pape/lgn/data.n5'
TARGET = 'local'
MAX_JOBS = 16
TMP_FOLDER = '/scratch/pape/lgn/tmp_problem'
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


def make_watershed(halo_name):
    task = WatershedWorkflow
    configs = task.get_config()
    _make_global_config(configs, halo_name)

    conf = configs['watershed']
    conf.update({
        'threshold': .25,
        'size_filter': 50,
        'channel_begin': 1,
        'channel_end': 3
    })

    with open(os.path.join(CONFIG_FOLDER, 'watershed.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=TMP_FOLDER, config_dir=CONFIG_FOLDER,
             target=TARGET, max_jobs=MAX_JOBS,
             input_path=PATH, input_key='affinities',
             output_path=PATH, output_key='watershed')
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def make_graph_and_costs(beta, halo_name):
    task = ProblemWorkflow
    configs = task.get_config()
    _make_global_config(configs, halo_name)

    conf = configs['block_edge_features']
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    conf.update({"offsets": offsets})
    with open(os.path.join(CONFIG_FOLDER, 'block_edge_features.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['probs_to_costs']
    conf.update({'beta': beta})
    with open(os.path.join(CONFIG_FOLDER, 'probs_to_costs.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=TMP_FOLDER, config_dir=CONFIG_FOLDER,
             target=TARGET, max_jobs=MAX_JOBS,
             input_path=PATH, input_key='affinities',
             ws_path=PATH, ws_key='watershed',
             problem_path=PATH)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def set_up_problem(halo_name):
    make_watershed(halo_name)
    make_graph_and_costs(beta=.6, halo_name=halo_name)


def check_watersheds():
    import napari
    bb = get_bounding_box(scale=0, halo=get_halo("check"))

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s0']
    ds.n_threads = 8
    raw = ds[bb]

    path = './data.n5'
    f = z5py.File(path, 'r')
    ds = f['watershed']
    ds.n_threads = 8
    ws = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(ws)


if __name__ == '__main__':
    set_up_problem(halo_name="large")
    check_watersheds()
