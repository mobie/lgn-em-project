import os
import h5py
import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize

ID_DICT = {0: 1, 1: 2, 2: 5}

# Abort mission ...
# The takeaways are:
# - s2 is too low res, should stick with s1, but use proper anisotropy in the network
# - there are still tons of mistakes in the training data
#   (touching boutons with the same id and boutons that are probably missed)


#
# we downscale the training data to isotropic resolution
# (correcponding to downscaling from s2 to s1)
#
def prepare_training_data(train_id):
    i = ID_DICT[train_id]
    p = f'/g/rompani/pape/lgn/tranining_data/boutons/V3/bdv/training_data_0{i}.h5'
    with h5py.File(p, 'r') as f:
        raw = f['t00000/s00/0/cells'][:]
        seg = f['t00000/s01/0/cells'][:]

    raw = block_reduce(raw, block_size=(1, 2, 2), func=np.mean)
    seg = resize(seg, raw.shape, order=0, preserve_range=True).astype('uint32')

    out_folder = '/g/rompani/pape/lgn/tranining_data/boutons/V4'
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, f'training_data_0{train_id}.h5')

    with h5py.File(out_file, 'a') as f:
        f.create_dataset('raw', data=raw, compression='gzip')
        f.create_dataset('labels/boutons', data=seg, compression='gzip')


def prepare_all():
    prepare_training_data(0)
    prepare_training_data(1)
    prepare_training_data(2)


def check_all():
    import napari
    for train_id in range(3):
        out_folder = '/g/rompani/pape/lgn/tranining_data/boutons/V4'
        out_file = os.path.join(out_folder, f'training_data_0{train_id}.h5')
        with h5py.File(out_file, 'r') as f:
            raw = f['raw'][:]
            seg = f['labels/boutons'][:]

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(raw)
            viewer.add_labels(seg)


if __name__ == '__main__':
    prepare_all()
    check_all()
