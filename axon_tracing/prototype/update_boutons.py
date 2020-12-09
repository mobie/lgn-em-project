import napari
import h5py
from vigra.analysis import relabelConsecutive


def update_boutons():
    with h5py.File('./test_data.h5', 'r') as f:
        raw = f['raw'][:]
        boutons = f['boutons'][:]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(boutons)

        boutons = viewer.layers['boutons'].data

    boutons = relabelConsecutive(boutons.astype('uint32'), start_label=1, keep_zeros=True)[0]

    with h5py.File('./test_data.h5', 'a') as f:
        ds = f.require_dataset('boutons_corrected', shape=boutons.shape, dtype=boutons.dtype, compression='gzip')
        ds[:] = boutons


if __name__ == '__main__':
    update_boutons()
