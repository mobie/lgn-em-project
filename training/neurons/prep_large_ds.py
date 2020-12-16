import z5py
import h5py
import nifty.tools as nt

# ROOT = '/g/kreshuk/data/helmstaedter/training_data/large_seg/'


def check_chunk():
    import napari
    raw_path = '/g/rompani/helmstaedter-data/l4dense2019.brain.mpg.de/webdav/electron-microscopy-volume/x0y0z0.hdf5'
    seg_path = '/g/rompani/helmstaedter-data/seg_tiles/x0y0z0.hdf5'

    with h5py.File(raw_path, 'r') as f:
        print(f['data'].shape)
        raw = f['data'][:]

    with h5py.File(seg_path, 'r') as f:
        print(f['data'].shape)
        seg = f['data'][:]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_labels(seg)


if __name__ == '__main__':
    check_chunk()
