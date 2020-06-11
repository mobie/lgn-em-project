import argparse
import os
import json

import luigi
import numpy as np
import nifty
import vigra
import z5py

from numba import jit

from cluster_tools.graph import GraphWorkflow
from cluster_tools.morphology import MorphologyWorkflow
from cluster_tools.write import WriteLocal, WriteSlurm
from downscale_boutons import run_downscaling

ROOT = '../../data'


def make_graph(seg_path, out_path, target, max_jobs, n_threads):
    task = GraphWorkflow
    tmp_folder = './tmp_graph'

    seg_key = 'setup0/timepoint0/s0'
    with z5py.File(seg_path, 'r') as f:
        chunks = f[seg_key].chunks
    bshape = [ch * 2 for ch in chunks]

    conf_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(conf_dir, exist_ok=True)
    configs = task.get_config()
    conf = configs['global']
    conf.update({'block_shape': bshape})
    with open(os.path.join(conf_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    conf = configs['merge_sub_graphs']
    conf.update({'mem_limit': 128, 'time_limit': 240, 'threads_per_job': n_threads})

    t = task(tmp_folder=tmp_folder, config_dir=conf_dir,
             max_jobs=max_jobs, target=target,
             input_path=seg_path, input_key=seg_key,
             graph_path=out_path, output_key='graph')
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def connected_components(path, n_threads):
    cc_key = 'node_labels/connected_components'
    f = z5py.File(path)
    if cc_key in f:
        ds = f[cc_key]
        node_labels = ds[:]
        return node_labels

    g = f['graph']
    n_nodes = g.attrs['nodeMaxId'] + 1

    ds_edges = g['edges']
    ds_edges.n_threads = n_threads
    edges = ds_edges[:]
    assert edges.max() < n_nodes

    g = nifty.graph.undirectedGraph(n_nodes)
    g.insertEdges(edges)

    node_labels = np.arange(n_nodes, dtype='uint64')
    node_labels = nifty.graph.connectedComponentsFromNodeLabels(g, node_labels,
                                                                dense=True,
                                                                ignoreBackground=True)
    f.create_dataset(cc_key, data=node_labels, chunks=node_labels.shape,
                     compression='gzip')
    return node_labels


def diameter_to_threshold(min_diameter):
    # we expect the min_diameter in microns and want the size threshold in pixels
    # pixel size: ~ 0.02 micron (20 nm)
    pix_diameter = min_diameter / 0.02
    pix_size = 4 * np.pi * (pix_diameter / 2) ** 3 / 3
    return pix_size


@jit(nopython=True)
def accumulate_sizes(node_labels, sizes):
    n_labels = int(node_labels.max() + 1)
    label_sizes = np.zeros(n_labels)
    for node_id in range(len(node_labels)):
        label_sizes[node_labels[node_id]] += sizes[node_id]
    return label_sizes


def compute_filtered_node_labels(size_threshold, node_labels, sizes, path, key):
    f = z5py.File(path)
    if key in f:
        return

    labels = np.unique(node_labels)
    if labels[0] == 0:
        labels = labels[1:]

    n_labels0 = len(labels)

    print("with pixel size threshold:", size_threshold)
    label_sizes = accumulate_sizes(node_labels, sizes)
    mapped_sizes = label_sizes[node_labels]
    node_labels[mapped_sizes < size_threshold] = 0

    vigra.analysis.relabelConsecutive(node_labels, keep_zeros=True, start_label=1,
                                      out=node_labels)
    n_labels1 = node_labels.max() + 1
    print("Reduced number of labels from", n_labels0, "to", n_labels1)

    f.create_dataset(key, data=node_labels, compression='gzip',
                     chunks=node_labels.shape)


def compute_sizes(seg_path, node_labels, target, max_jobs):
    tmp_folder = './tmp_sizes'
    out_path = os.path.join(tmp_folder, 'data.n5')

    task = MorphologyWorkflow

    seg_key = 'setup0/timepoint0/s0'
    with z5py.File(seg_path, 'r') as fseg:
        chunks = fseg[seg_key].chunks
    bshape = [ch * 2 for ch in chunks]

    conf_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(conf_dir, exist_ok=True)
    configs = task.get_config()
    conf = configs['global']
    conf.update({'block_shape': bshape})
    with open(os.path.join(conf_dir, 'global.config'), 'w') as fj:
        json.dump(conf, fj)

    out_key = 'morphology'

    t = task(tmp_folder=tmp_folder, config_dir=conf_dir,
             max_jobs=max_jobs, target=target,
             input_path=seg_path, input_key=seg_key,
             output_path=out_path, output_key=out_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret

    f = z5py.File(out_path, 'r')
    ds = f[out_key]
    sizes = ds[:, 1]
    return sizes


def size_filter(size_threshold, in_path, out_path,
                node_labels, sizes,
                target, max_jobs, prefix):
    task = WriteLocal if target == 'local' else WriteSlurm

    tmp_folder = './tmp_%s' % prefix

    conf_dir = os.path.join(tmp_folder, 'configs')
    os.makedirs(conf_dir, exist_ok=True)

    tmp_path = os.path.join(tmp_folder, 'data.n5')
    node_label_key = 'node_labels'

    print("Size filter node labels ...")
    compute_filtered_node_labels(size_threshold, node_labels, sizes, tmp_path, node_label_key)

    key = 'setup0/timepoint0/s0'
    t = task(tmp_folder=tmp_folder, config_dir=conf_dir, max_jobs=max_jobs,
             input_path=in_path, input_key=key,
             output_path=out_path, output_key=key,
             assignment_path=tmp_path, assignment_key=node_label_key,
             identifier=prefix)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def postprocess(size_threshold, target='slurm', max_jobs=250, n_threads=32, prefix='filtered'):
    """ Postprocess bouton segmentation by size-filter after connected components.
    Min diameter in micrometer.
    """

    seg_path = '../../data.n5'
    tmp_path = './data.n5'

    make_graph(seg_path, tmp_path, target=target, max_jobs=max_jobs,
               n_threads=n_threads)

    print("Run connected components ...")
    node_labels = connected_components(tmp_path, n_threads)

    sizes = compute_sizes(seg_path, node_labels, target=target, max_jobs=max_jobs)

    assert len(sizes) == len(node_labels), "%i, %i" % (len(sizes),
                                                       len(node_labels))

    if prefix is None:
        prefix = str(size_threshold)

    size_filter(size_threshold, seg_path, seg_path,
                node_labels, sizes,
                target, max_jobs, prefix)

    # downscale the size filtered segmentation
    in_key = 'setup0/timepoint0/s0'
    run_downscaling(out_path, in_key, out_path, target, max_jobs, prefix)


def postprocess_diameter(min_diameter, target='slurm', max_jobs=250, n_threads=32):
    min_diameter_nm = min_diameter * 1000
    prefix = 'filtered%inm' % int(min_diameter_nm)
    size_threshold = diameter_to_threshold(min_diameter)
    print("Post-process with min diameter", min_diameter,
          "corresponding to pixel size", size_threshold)
    postprocess(size_threshold, target, max_jobs, n_threads, prefix)


# Size to thresholds from Marina:
# 0.1  um diameter =   65.45 pix
# 0.25 um diameter = 1022.65 pix
# custom threshold = 5000.00 pix (= 0.91  um diameter)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_diameter', type=int, default=0)
    parser.add_argument('--size_threshold', type=int, default=5000)
    parser.add_argument('--target', type=str, default='local')
    parser.add_argument('--max_jobs', default=48, type=int)

    args = parser.parse_args()
    if bool(args.is_diameter):
        postprocess_diameter(args.size_threshold,
                             target=args.target,
                             max_jobs=args.max_jobs)
    else:
        postprocess(args.size_threshold,
                    target=args.target,
                    max_jobs=args.max_jobs)
