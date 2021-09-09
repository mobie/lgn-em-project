import argparse
import os
import napari
from elf.io import open_file

ANNOTATION_PATHS = [
    "./annotations_v2/2021-08-237_13-55__training_data_z465y15673x20488-z529y16185x21000__volume",
    "./annotations_v2/2021-08-237_13-55__training_data_z481y17612x23043-z545y18124x23555__volume",
    "./annotations_v2/2021-08-237_13-56__training_data_z481y16967x21745-z545y17479x22257__volume",
    "./annotations_v2/2021-08-237_13-56__training_data_z485y17596x23654-z549y18108x24166__volume",
    "./annotations_v2/2021-08-238_16-24__training_data_z364y13744x11342-z428y14256x11854__volume",
    "./annotations_v2/2021-09-250_09-55__training_data_z364y13905x13264-z428y14417x13776__volume",
    "./annotations_v1/annotations2/2021-08-222_13-35__training_data_z1235y13855x14657-z1299y14367x15169__volume",
    "./annotations_v3/2021-08-222_13-35__training_data_z1235y13855x14657-z1299y14367x15169__volume",
    "./annotations_v3/2021-09-251_08-28__training_data_z1248y14877x14589-z1312y15389x15101__volume",
    "./annotations_v3/2021-09-244_09-17__training_data_z1250y11863x13631-z1314y12375x14143__volume",
]


def check_annotations(raw_path, annotation_path, block_id):
    with open_file(raw_path, "r") as f:
        raw = f["raw"][:]
    with open_file(annotation_path, "r") as f:
        seg = f["*.tiff"][:]
    assert raw.shape == seg.shape, f"{raw.shape}, {seg.shape}"

    with napari.gui_qt():
        v = napari.Viewer()
        v.title = f"Block-{block_id}"
        v.add_image(raw)
        v.add_labels(seg)


def check_block(block_id):
    print("Checking block", block_id)
    raw_dir1 = "./training_data"
    raw_dir2 = "./training_data2"
    annotation_path = ANNOTATION_PATHS[block_id - 1]

    vol_name = os.path.split(annotation_path)[1]
    vol_name = vol_name.split("__")[1]
    raw_path = os.path.join(raw_dir1, vol_name + ".h5")
    if not os.path.exists(raw_path):
        raw_path = os.path.join(raw_dir2, vol_name + ".h5")
    assert os.path.exists(raw_path), f"{vol_name}: {raw_path}"

    check_annotations(raw_path, annotation_path, block_id)


def check_all():
    for block_id in range(1, len(ANNOTATION_PATHS) + 1):
        check_block(block_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_id", "-b", type=int, default=None)
    args = parser.parse_args()
    block_id = args.block_id
    if block_id is None:
        check_all()
    else:
        check_block(args.block_id)
