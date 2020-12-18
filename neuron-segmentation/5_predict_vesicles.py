import json
import os

import z5py

import luigi

from cluster_tools.inference import InferenceLocal, InferenceSlurm
from common import get_bounding_box, BLOCK_SHAPE

MODEL_PATH = '../training/semantic/networks/v1/Weights'


def predict_vesicles(target, gpus, threads_per_job=6):
    task = InferenceLocal if target == 'local' else InferenceSlurm
    halo = [8, 64, 64]

    output_key = {
        'vesicles': [1, 2]
    }

    tmp_folder = './tmp_prediction_vesicles'
    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    roi_begin, roi_end = get_bounding_box(return_as_lists=True)
    conf = task.default_global_config()
    conf.update({'block_shape': BLOCK_SHAPE,
                 'roi_begin': roi_begin,
                 'roi_end': roi_end})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    if target == 'local':
        device_mapping = {ii: gpu for ii, gpu in enumerate(gpus)}
    else:
        device_mapping = None

    conf = task.default_task_config()
    conf.update({
        'dtype': 'uint8',
        'device_mapping': device_mapping,
        'threads_per_job': threads_per_job
    })
    with open(os.path.join(config_dir, 'inference.config'), 'w') as f:
        json.dump(conf, f)

    input_path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    input_key = 'setup0/timepoint0/s0'

    # TODO for larger outputs we should put this on scratch
    output_path = './data.n5'

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=len(gpus),
             input_path=input_path, input_key=input_key,
             output_path=output_path, output_key=output_key,
             checkpoint_path=MODEL_PATH, halo=halo,
             framework='inferno')
    assert luigi.build([t], local_scheduler=True)


def check_predictions():
    import napari
    bb = get_bounding_box(scale=0)

    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    f = z5py.File(path, 'r')
    ds = f['setup0/timepoint0/s0']
    ds.n_threads = 8
    raw = ds[bb]

    path = './data.n5'
    f = z5py.File(path, 'r')
    ds = f['vesicles']
    ds.n_threads = 8
    pred = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_image(pred)


if __name__ == '__main__':
    gpus = [2, 3, 7]
    predict_vesicles(target='local', gpus=gpus)
    check_predictions()
