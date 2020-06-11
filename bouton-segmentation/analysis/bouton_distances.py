import json
import os

import luigi
from cluster_tools.distances import PairwiseDistanceWorkflow
from cluster_tools.morphology import MorphologyWorkflow


def scale_to_res(scale):
    ress = [[2, 1, 1], [2, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    return ress[scale]


def compute_object_distances(full_vol, target, max_jobs, scale, for_pp=True):

    if full_vol:
        path = '/g/rompani/pape/lgn/data.n5'
        tmp_folder = '/g/rompani/pape/lgn/tmp_dist'
        out_path = './pairwise_bouton_distances.pkl'
    else:
        path = './data.n5'
        tmp_folder = './tmp_dist'
        out_path = './pairwise_distances.pkl'

    if for_pp:
        seg_key = 'predictions/pp-seg-mip/s%i' % scale
    else:
        seg_key = 'predictions/pp-seg-mip/s%i' % scale
        tmp_folder += '_no_pp'
        out_path = './pairwise_bouton_distances_no_pp.pkl'

    morpho_key = 'morphology_s%i' % scale

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
    luigi.build([task], local_scheduler=True)
