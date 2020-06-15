import argparse
import json
import os

import luigi
from cluster_tools.distances import PairwiseDistanceWorkflow
from cluster_tools.morphology import MorphologyWorkflow

ROOT = '../../data'


def scale_to_res(scale):
    ress = [[2, 1, 1], [2, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    return ress[scale]


def compute_object_distances(path, seg_name, scale, out_path, target, max_jobs):

    tmp_folder = f'tmp_distances_{seg_name}'
    morpho_key = 'morphology_s%i' % scale
    seg_key = f'setup0/timepoint0/s{scale}'

    config_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    configs = MorphologyWorkflow.get_config()
    conf = configs['global']
    shebang = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch13/bin/python'
    block_shape = [32, 256, 256]
    conf.update({'shebang': shebang, 'block_shape': block_shape})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    # compute object morphology
    task = MorphologyWorkflow(tmp_folder=tmp_folder, config_dir=config_dir,
                              target=target, max_jobs=max_jobs,
                              input_path=path, input_key=seg_key,
                              output_path=path, output_key=morpho_key)
    luigi.build([task], local_scheduler=True)

    # compute pairwaise object distances
    resolution = scale_to_res(scale)
    max_dist = 250
    task = PairwiseDistanceWorkflow(tmp_folder=tmp_folder, config_dir=config_dir,
                                    target=target, max_jobs=max_jobs,
                                    input_path=path, input_key=seg_key,
                                    morphology_path=path, morphology_key=morpho_key,
                                    output_path=out_path, max_distance=max_dist,
                                    resolution=resolution)
    ret = luigi.build([task], local_scheduler=True)
    assert ret


# TODO implement merging of boutons
def object_distance_table(seg_name, scale, target, max_jobs, merge_connected=False):
    seg_path = os.path.join(ROOT, f'0.0.0/images/local/{seg_name}.n5')
    compute_object_distances(seg_path, seg_name, scale, target, max_jobs)
    # TODO write the table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_name', default='sbem-adult-1-lgn-boutons')
    parser.add_argument('--scale', default=2, type=int)
    parser.add_argument('--target', default='local')
    parser.add_argument('--max_jobs', type=int, default=64)

    args = parser.parse_args()
    object_distance_table(args.seg_name, args.scale, args.target, args.max_jobs)
