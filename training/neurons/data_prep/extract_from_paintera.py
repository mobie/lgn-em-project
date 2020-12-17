import os
import numpy as np
import h5py
import z5py
import nifty.tools as nt
from paintera_tools.serialize import serialize_from_commit
from extract_training_data import get_bounding_box, PATH

ROOT = '/g/kreshuk/pape/Work/data/rompani/neuron_training_data'


def extract_seg(block_id):
    data_path = os.path.join(ROOT, 'train_data_5.n5')
    pt_path = './paintera_data/block%i.n5' % block_id
    pt_key = 'paintera'

    tmp_folder = './tmp_serialize_%i' % block_id
    tmp_out = os.path.join(tmp_folder, 'data.n5')
    tmp_key = 'seg'

    # serialize segmentation to tmp
    serialize_from_commit(
        pt_path, pt_key,
        tmp_out, tmp_key,
        tmp_folder=tmp_folder,
        relabel_output=True,
        max_jobs=8, target='local'
    )

    # read the corresponding raw data with halo
    with z5py.File(tmp_out, 'r') as f:
        seg = f[tmp_key][:]

    with z5py.File(data_path, 'r') as f:
        ds = f['raw']
        shape = ds.shape
    block_shape = [48, 384, 384]
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    inner_block = blocking.getBlock(block_id)

    # inner_bb = tuple(slice(beg, end) for beg, end in zip(inner_block.begin, inner_block.end))
    # raw = ds[inner_bb]
    # seg_full = seg

    halo = [64, 512, 512]
    bb_raw = get_bounding_box(5, halo)
    start_coord = [ib + bb.start for ib, bb in zip(inner_block.begin, bb_raw)]

    halo = [8, 32, 32]
    bb = tuple(slice(sc - ha, sc + sh + ha) for sc, sh, ha in zip(start_coord,
                                                                  seg.shape,
                                                                  halo))

    with z5py.File(PATH, 'r') as f:
        ds = f['setup0/timepoint0/s0']
        raw = ds[bb]
    print(raw.shape)

    seg_full = np.zeros(raw.shape, dtype=seg.dtype)
    bb = tuple(slice(ha, sh - ha) for sh, ha in zip(seg_full.shape, halo))
    seg_full[bb] = seg

    out_folder = os.path.join(ROOT, 'v1')
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, f'block_{block_id}.h5')

    with h5py.File(out_file, 'a') as f:
        f.create_dataset('raw', data=raw, compression='gzip')
        f.create_dataset('labels', data=seg_full, compression='gzip')


def view_seg(block_id):
    import napari
    out_folder = os.path.join(ROOT, 'v1')
    out_file = os.path.join(out_folder, f'block_{block_id}.h5')

    with h5py.File(out_file, 'r') as f:
        raw = f['raw'][:]
        seg = f['labels'][:]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


if __name__ == '__main__':
    block_id = 8
    extract_seg(block_id)
    view_seg(block_id)
