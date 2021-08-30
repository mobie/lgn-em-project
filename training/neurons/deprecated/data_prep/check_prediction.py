import z5py
import napari


def check_predictions():
    path = '../training_data/train_data_5.n5'
    with z5py.File(path, 'r') as f:
        ds = f['raw']
        ds.n_threads = 8
        raw = ds[:]
        ds = f['predictions_3d/lr0.0001_use-affs1_weight1.state']
        ds.n_threads = 8
        pred = ds[:]

    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw)
        v.add_image(pred)


check_predictions()
