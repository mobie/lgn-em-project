import argparse
import json
import os

import numpy as np
import luigi
from cluster_tools.distances import PairwiseDistanceWorkflow
from cluster_tools.morphology import MorphologyWorkflow
from elf.io import open_file

ROOT = '../../data'


def scale_to_res(scale):
    ress = [[2, 1, 1], [2, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    return ress[scale]


def find_max_size(path, key, topk=25):
    with open_file(path, 'r') as f:
        ds = f[key]
        sizes = ds[:, 1]

    sizes = np.sort(sizes)[::-1]
    print("Top", topk, "sizes:")
    print(sizes[:topk])


def compute_object_distances(path, seg_key, scale,
                             tmp_folder, out_path,
                             target, max_jobs,
                             max_size):

    morpho_key = 'morphology_s%i' % scale

    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    configs = MorphologyWorkflow.get_config()
    conf = configs['global']
    block_shape = [32, 256, 256]
    conf.update({'block_shape': block_shape})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    # compute object morphology
    task = MorphologyWorkflow
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
             input_path=path, input_key=seg_key,
             output_path=path, output_key=morpho_key)
    luigi.build([t], local_scheduler=True)

    if max_size is None:
        find_max_size(path, morpho_key)
        return

    # compute pairwise object distances
    resolution = scale_to_res(scale)
    max_dist = 250
    task = PairwiseDistanceWorkflow

    configs = task.get_config()
    config = configs['object_distances']
    config.update({'mem_limit': 32, 'time_limit': 1000})
    with open(os.path.join(config_dir, 'object_distances.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             target=target, max_jobs=max_jobs,
             input_path=path, input_key=seg_key,
             morphology_path=path, morphology_key=morpho_key,
             output_path=out_path, max_distance=max_dist,
             resolution=resolution, max_size=max_size)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def object_distance_table(seg_name, scale, target, max_jobs,
                          use_merged_segmentation=False, max_size=None):
    seg_path = os.path.join(ROOT, f'0.0.0/images/local/{seg_name}.n5')
    seg_key = f'setup0/timepoint0/s{scale}'
    tmp_folder = f'tmp_distances_{seg_name}'

    # ideally we would just have the merged segmentation in another seg-name
    # and use the output of the script 'bouton-segmentation/segmentation/6_merge_segmentation.py'
    # however, in the interest of time I am reusing the result from an earlier iteration of that script,
    # which is not in the mobie format yet
    if use_merged_segmentation:
        seg_path = '/g/rompani/pape/lgn/data.n5'
        seg_key = f'predictions/pp-seg-mip/s{scale}'
        tmp_folder = 'tmp_distances_merged'
        out_path = './merged_distances.pkl'
    else:
        out_path = './distances.pkl'

    compute_object_distances(seg_path, seg_key, scale,
                             tmp_folder, out_path,
                             target, max_jobs,
                             max_size=max_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_name', default='sbem-adult-1-lgn-boutons')
    parser.add_argument('--scale', default=2, type=int)
    parser.add_argument('--target', default='slurm')
    parser.add_argument('--max_jobs', type=int, default=150)
    parser.add_argument('--max_size', type=int, default=int(1e7))
    parser.add_argument('--use_merged_segmentation', type=int, default=0)

    args = parser.parse_args()
    object_distance_table(args.seg_name, args.scale,
                          args.target, args.max_jobs,
                          max_size=args.max_size,
                          use_merged_segmentation=bool(args.use_merged_segmentation))
