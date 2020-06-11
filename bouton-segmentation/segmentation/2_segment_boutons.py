import argparse
import os
import json

import luigi

from elf.io import open_file
from elf.parallel import greater_equal
from elf.wrapper import NormalizeWrapper
from elf.wrapper.resized_volume import ResizedVolume
from cluster_tools.mutex_watershed import MwsWorkflow


# TODO implement as cluster task to run safely on the login node
def run_thresh(path, mask_path, n_jobs):

    f = open_file(path)
    ds = NormalizeWrapper(f['predictions/foreground'])
    threshold = .5

    ds_out = f.require_dataset('predictions/mask', shape=ds.shape, compression='gzip',
                               dtype='uint8', chunks=ds.chunks)

    # with mask for the big volume
    mask_key = 'setup0/timepoint0/s0'
    ds_mask = f[mask_key]
    ds_mask = ResizedVolume(ds_mask, shape=ds.shape, order=0)

    n_threads = 32
    greater_equal(ds, threshold, out=ds_out, verbose=True,
                  n_threads=n_threads, mask=ds_mask)


def run_mws(path, tmp_folder, target, max_jobs, qos='normal'):
    task = MwsWorkflow

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3]]
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    configs = task.get_config()
    conf = configs['global']
    block_shape = [32, 256, 256]
    conf.update({'block_shape': block_shape})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(conf, f)

    # write config for edge feature task
    conf = configs['block_edge_features']
    conf.update({'offsets': offsets, 'mem_limit': 4, 'qos': qos})
    with open(os.path.join(config_folder, 'block_edge_features.config'), 'w') as f:
        json.dump(conf, f)

    # write config for mws block task
    strides = [4, 4, 4]
    conf = configs['mws_blocks']
    conf.update({'randomize_strides': True, 'strides': strides, 'mem_limit': 8,
                 'time_limit': 600})
    with open(os.path.join(config_folder, 'mws_blocks.config'), 'w') as f:
        json.dump(conf,  f)

    # write config for stitching multicut
    conf = configs['stitching_multicut']
    conf.update({'beta1': 0.5, 'beta2': 0.5, 'qos': qos})
    with open(os.path.join(config_folder, 'stitching_multicut.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['write']
    conf.update({'mem_limit': 10, 'time_limit': 120, 'qos': qos})
    with open(os.path.join(config_folder, 'write.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['simple_stitch_edges']
    conf.update({'qos': qos})
    with open(os.path.join(config_folder, 'simple_stitch_edges.config'), 'w') as f:
        json.dump(conf, f)

    conf_names = ['merge_edge_features', 'merge_sub_graphs',
                  'map_edge_ids', 'simple_stitch_assignments']
    for name in conf_names:
        conf = configs[name]
        conf.update({'mem_limit': 128, 'time_limit': 240, 'threads_per_job': 16, 'qos': qos})
        with open(os.path.join(config_folder, '%s.config' % name), 'w') as f:
            json.dump(conf, f)

    conf = configs['stitching_multicut']
    # set time limit for the multicut task to 18 hours (in minutes)
    tlim_task = 18 * 60
    # set time limit for the solver to 16 hours (in seconds)
    tlim_solver = 16 * 60 * 60
    conf.update({'mem_limit': 256, 'time_limit': tlim_task, 'threads_per_job': 16, 'qos': qos,
                 'agglomerator': 'greedy-additive', 'time_limit_solver': tlim_solver})
    with open(os.path.join(config_folder, 'stitching_multicut.config'), 'w') as f:
        json.dump(conf, f)

    in_key = 'predictions/affinities'
    mask_key = 'predictions/mask'
    out_key = 'predictions/mws-seg'

    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             input_path=path, input_key=in_key,
             output_path=path, output_key=out_key,
             mask_path=path, mask_key=mask_key,
             stitch_via_mc=True, offsets=offsets)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Mws workflow failed"


def segment_boutons(target, max_jobs):
    path = '../../data.n5'
    mask_path = '../../data/0.0.0/images/local/sbem-adult-1-lgn-mask.n5'
    tmp_folder = './tmp_mws_segmentation'

    run_thresh(path, mask_path, max_jobs)
    run_mws(path, tmp_folder, target, max_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='local', type=str)
    parser.add_argument('--max_jobs', default=64, type=int)
    args = parser.parse_args()
    segment_boutons(args.target, args.max_jobs)
