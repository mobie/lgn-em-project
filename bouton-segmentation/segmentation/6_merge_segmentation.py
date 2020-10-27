import json
import os
from concurrent import futures

import luigi
import nifty
import numpy as np
import vigra
from elf.io import open_file
from elf.segmentation.postprocess import graph_watershed
from tqdm import tqdm

from cluster_tools.graph import GraphWorkflow
from cluster_tools.features import EdgeFeaturesWorkflow, RegionFeaturesWorkflow
from cluster_tools.write import WriteLocal, WriteSlurm


def find_seeds(node_to_component, node_sizes, min_component_size=0,
               n_threads=1):
    assert len(node_to_component) == len(node_sizes)
    size_heuristic = .3
    n_nodes = len(node_to_component)
    seeds = np.zeros(n_nodes, dtype='uint32')

    # if we don't have node sizes, we put one seeds per component
    if node_sizes is None:
        component_ids, component_pos = np.unique(node_to_component, return_index=True)
        seeds[component_pos] = component_ids
        return seeds

    # if we do have node sizes then iterate over all components
    # and put seeds according to size heuristic
    component_ids, node_ids, component_lens = np.unique(node_to_component,
                                                        return_index=True,
                                                        return_counts=True)
    singleton_components = component_lens == 1
    # these sizes are only valid for the singleton components !
    component_sizes = node_sizes[node_ids]
    keep_mask = np.logical_and(singleton_components,
                               component_sizes > min_component_size)
    keep_nodes = node_ids[keep_mask]

    seeds[keep_nodes] = component_ids[keep_mask]
    component_ids = component_ids[~singleton_components]

    if component_ids[0] == 0:
        component_ids = component_ids[1:]

    def split_component(cid):
        this_ids = np.where(node_to_component == cid)[0]

        component_size = node_sizes[this_ids].sum()
        if component_size < min_component_size:
            return

        this_sizes = node_sizes[this_ids]
        total_size = this_sizes.sum()
        size_fractions = this_sizes / total_size

        keep_components = this_ids[size_fractions > size_heuristic]
        for offset_id, keep_component in enumerate(keep_components):
            seeds[keep_component] = (cid + offset_id)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(split_component, component_ids), total=len(component_ids)))

    return seeds


def graph_merging(problem_path, graph_key, feat_key,
                  size_key, out_key, n_threads=16,
                  min_component_size=0):

    f = open_file(problem_path)
    if out_key in f:
        return

    ds = f[feat_key]
    ds.n_threads = n_threads
    edge_weigths = ds[:, 0]

    n_nodes = f[graph_key].attrs['numberOfNodes']
    ds = f[os.path.join(graph_key, 'edges')]
    ds.n_threads = n_threads
    uv_ids = ds[:]
    assert uv_ids.max() < n_nodes, "%i, %i" % (uv_ids.max(), n_nodes)
    print("Have graph with %i nodes and %i edges" % (n_nodes, len(uv_ids)))

    ds = f[size_key]
    ds.n_threads = n_threads
    sizes = ds[:, 0]

    # uv ids
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    zero_edges = (uv_ids == 0).any(axis=1)
    edge_weigths[zero_edges] = 1.1

    cc_key = 'assignments/connected_components_pp'
    if cc_key in f:
        print("Loading connected components ...")
        connected_conmponents = f[cc_key][:]
    else:
        print("Computing connected components ...")
        node_labels = np.ones(n_nodes, dtype='uint64')
        node_labels[0] = 0
        connected_conmponents = nifty.graph.connectedComponentsFromNodeLabels(graph, node_labels, True, True)
        chunks = (min(int(1e6), n_nodes),)
        f.create_dataset(cc_key, data=connected_conmponents, chunks=chunks)

    # find seeds by iterating over the connected components of the graph
    print("Computing seeds ...")
    seeds = find_seeds(connected_conmponents, sizes,
                       min_component_size=min_component_size,
                       n_threads=32)
    assert seeds[0] == 0
    print("Found %i unique seeds" % len(np.unique(seeds)))
    bg_id = n_nodes + 1
    seeds[0] = bg_id

    print("Running graph watershed and mapping back ...")
    node_labels = graph_watershed(graph, edge_weigths, seeds)
    node_labels[0] = 0
    vigra.analysis.relabelConsecutive(node_labels, out=node_labels,
                                      keep_zeros=True, start_label=1)

    chunks = (min(node_labels.shape[0], int(1e6)),)
    ds = f.require_dataset(out_key, shape=node_labels.shape, chunks=chunks,
                           compression='gzip', dtype=node_labels.dtype)
    print("Writing node labels ...")
    ds[:] = node_labels


def merge_boutons(target, max_jobs):
    qos = 'normal'

    path = '/g/rompani/pape/lgn/data.n5/predictions'
    tmp_folder = '/g/rompani/pape/lgn/tmp_pp'
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3]]

    fg_key = 'predictions/foreground'
    seg_key = 'predictions/mws-seg'
    aff_key = 'predictions/affinities'

    conf_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(conf_dir, exist_ok=True)
    problem_path = os.path.join(tmp_folder, 'data.n5')
    graph_key = 'graph'
    feat_key = 'feats'
    size_key = 'region_sizes'
    pp_node_label_key = 'node_labels/gws'

    configs = GraphWorkflow.get_config()
    conf = configs['global']
    shebang = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch13/bin/python'
    block_shape = [32, 256, 256]
    conf.update({'shebang': shebang, 'block_shape': block_shape})
    with open(os.path.join(conf_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['initial_sub_graphs']
    conf.update({'ignore_label': True, 'mem_limit': 2, 'qos': qos})
    with open(os.path.join(conf_dir, 'initial_sub_graphs.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['merge_sub_graphs']
    conf.update({'mem_limit': 64, 'threads_per_job': 16, 'qos': qos})
    with open(os.path.join(conf_dir, 'merge_sub_graphs.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['map_edge_ids']
    conf.update({'mem_limit': 64, 'threads_per_job': 8, 'qos': qos})
    with open(os.path.join(conf_dir, 'map_edge_ids.config'), 'w') as f:
        json.dump(conf, f)

    task = GraphWorkflow(tmp_folder=tmp_folder, config_dir=conf_dir,
                         max_jobs=max_jobs, target=target,
                         input_path=path, input_key=seg_key,
                         graph_path=problem_path, output_key=graph_key)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Graph extraction failed"

    # write config for edge feature task
    configs = EdgeFeaturesWorkflow.get_config()
    conf = configs['block_edge_features']
    conf.update({'offsets': offsets, 'qos': qos})
    with open(os.path.join(conf_dir, 'block_edge_features.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['merge_edge_features']
    conf.update({'mem_limit': 64, 'threads_per_job': 12, 'qos': qos,
                 'time_limit': 360})
    with open(os.path.join(conf_dir, 'merge_edge_features.config'), 'w') as f:
        json.dump(conf, f)

    task = EdgeFeaturesWorkflow(tmp_folder=tmp_folder, config_dir=conf_dir,
                                max_jobs=max_jobs, target=target,
                                input_path=path, input_key=aff_key,
                                labels_path=path, labels_key=seg_key,
                                graph_path=problem_path, graph_key=graph_key,
                                output_path=problem_path, output_key=feat_key)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Edge feature extraction failed"

    configs = RegionFeaturesWorkflow.get_config()
    conf = configs['region_features']
    conf.update({'mem_limit': 2, 'time_limit': 360, 'qos': qos, 'threads_per_job': 1})
    with open(os.path.join(conf_dir, 'region_features.config'), 'w') as f:
        json.dump(conf, f)
    conf = configs['merge_region_features']
    conf.update({'mem_limit': 2, 'time_limit': 360, 'qos': qos})
    with open(os.path.join(conf_dir, 'merge_region_features.config'), 'w') as f:
        json.dump(conf, f)

    task = RegionFeaturesWorkflow(tmp_folder=tmp_folder, config_dir=conf_dir,
                                  max_jobs=max_jobs, target=target,
                                  input_path=path, input_key=fg_key,
                                  labels_path=path, labels_key=seg_key,
                                  output_path=problem_path, output_key=size_key,
                                  max_jobs_merge=16)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Region feature extraction failed"

    # TODO proper size filter ?
    min_component_size = 2000
    graph_merging(problem_path, graph_key, feat_key, size_key,
                  pp_node_label_key, min_component_size=min_component_size)

    out_key = 'predictions/pp-seg'
    task = WriteSlurm if target == 'slurm' else WriteLocal

    conf = task.default_task_config()
    conf.update({'mem_limit': 8})
    with open(os.path.join(conf_dir, 'write.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=conf_dir, max_jobs=max_jobs,
             input_path=path, input_key=seg_key,
             output_path=path, output_key=out_key,
             assignment_path=problem_path,
             assignment_key=pp_node_label_key,
             identifier='write-pp')
    luigi.build([t], local_scheduler=True)


#
# script to merge touching segments that likely correspond to the same bouton
# this is only partially ported from
# /g/rompani/pape/lgn/lgn_experiments/boutons/prediction/segmentation_workflow.py
# in the interest of time, I am using the result of that script and not finishing
# the port and rerunning it
#

if __name__ == '__main__':
    target = 'slurm'
    max_jobs = 200
    merge_boutons()
