import os
import z5py
# import numpy as np
import nifty.tools as nt
from elf.segmentation.workflows import simple_multicut_workflow
from elf.segmentation.watershed import stacked_watershed

ROOT = '/g/kreshuk/pape/Work/data/rompani/neuron_training_data'


def load_data(block_id, block_shape):
    path = os.path.join(ROOT, 'train_data_5.n5')
    f = z5py.File(path)
    ds = f['raw']
    # ds_affs = f['predictions_3d/lr0.0001_use-affs1_weight1.state']
    ds_affs = f['predictions_3d/segemV4']
    shape = ds.shape
    blocks = nt.blocking([0, 0, 0], shape, block_shape)
    block = blocks.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

    raw = ds[bb]
    # bb_affs = (slice(0, 4),) + bb
    # affs = ds_affs[bb_affs].squeeze()
    # affs = np.max(affs, axis=0)

    bb_affs = (1,) + bb
    affs = ds_affs[bb_affs].squeeze()

    return raw, affs


def segment(boundaries, beta):
    ws, _ = stacked_watershed(boundaries, threshold=.5, sigma_seeds=1.)
    results = simple_multicut_workflow(boundaries,
                                       use_2dws=True,
                                       watershed=ws,
                                       multicut_solver='kernighan-lin',
                                       beta=beta,
                                       offsets=None,
                                       n_threads=8,
                                       weighting_scheme=None,
                                       return_intermediates=True)
    ws = results['watershed']
    node_labels = results['node_labels']
    return ws, boundaries, node_labels


def make_data(block_id, block_shape, beta):
    raw, boundaries = load_data(block_id, block_shape)
    seg, boundaries, node_labels = segment(boundaries, beta)
    return raw, boundaries, seg, node_labels


def make_paintera(path):
    from paintera_tools import convert_to_paintera_format, downscale
    scale_factors = [[1, 2, 2], [1, 2, 2]]
    downscale(path, 'raw/s0', 'raw',
              scale_factors, scale_factors,
              tmp_folder='tmp_paintera_ds', target='local',
              max_jobs=8, resolution=[40, 10, 10])

    convert_to_paintera_format(path, 'raw', 'seg', 'paintera',
                               label_scale=0, resolution=[40, 10, 10],
                               target='local', tmp_folder='tmp_paintera',
                               max_jobs=8, max_threads=8,
                               assignment_path=path, assignment_key='node_labels')


def set_up_paintera(block_id, check):
    # block_shape = [64, 512, 512]
    block_shape = [48, 384, 384]
    # block_shape = [32, 256, 256]
    print("Making data...")
    raw, boundaries, ws, node_labels = make_data(block_id, block_shape, beta=.5)
    print("... done")

    if check:
        import napari
        seg = nt.take(node_labels, ws)
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(raw)
            viewer.add_image(boundaries)
            viewer.add_labels(seg)
        return

    path = f'paintera_data/block{block_id}.n5'
    with z5py.File(path) as f:
        f.create_dataset('raw/s0', data=raw, compression='gzip', chunks=(32, 256, 256))
        ds = f.create_dataset('seg', data=ws, compression='gzip', chunks=(32, 256, 256))
        ds.attrs['maxId'] = int(ws.max())
        f.create_dataset('node_labels', data=node_labels)

    make_paintera(path)


if __name__ == '__main__':
    set_up_paintera(block_id=1, check=False)
