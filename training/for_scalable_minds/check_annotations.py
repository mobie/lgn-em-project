import argparse
import napari
from elf.io import open_file


def check_annotations(raw_path, annotation_path):
    with open_file(raw_path, "r") as f:
        raw = f["raw"][:]
    with open_file(annotation_path, "r") as f:
        seg = f["*.tiff"][:]
    assert raw.shape == seg.shape, f"{raw.shape}, {seg.shape}"

    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(seg)


def check_block(block_id):
    raw_paths = {
        1: "./training_data/training_data_z481y16967x21745-z545y17479x22257.h5",
        2: "./training_data/training_data_z364y13744x11342-z428y14256x11854.h5",
        3: "./training_data/training_data_z485y17596x23654-z549y18108x24166.h5",
        4: "./training_data2/training_data_z1235y13855x14657-z1299y14367x15169.h5",
        11: "./training_data2/training_data_z1235y13855x14657-z1299y14367x15169.h5",
        12: "./training_data2/training_data_z1248y14877x14589-z1312y15389x15101.h5",
        13: "./training_data2/training_data_z1250y11863x13631-z1314y12375x14143.h5",
    }
    annotation_paths = {
        1: "./annotations_v1/annotations/2021-08-222_09-24__training_data_z481y16967x21745-z545y17479x22257__volume",
        2: "./annotations_v1/annotations/2021-08-222_09-25__training_data_z364y13744x11342-z428y14256x11854__volume",
        3: "./annotations_v1/annotations/2021-08-222_09-25__training_data_z485y17596x23654-z549y18108x24166__volume",
        4: "./annotations_v1/annotations2/2021-08-222_13-35__training_data_z1235y13855x14657-z1299y14367x15169__volume",
        5: "",
        6: "",
        7: "",
        8: "",
        9: "",
        10: "",
        11: "./annotations_v3/2021-08-222_13-35__training_data_z1235y13855x14657-z1299y14367x15169__volume",
        12: "./annotations_v3/2021-08-235_08-48__training_data_z1248y14877x14589-z1312y15389x15101__volume",
        13: "./annotations_v3/2021-09-244_09-17__training_data_z1250y11863x13631-z1314y12375x14143__volume",
    }
    check_annotations(raw_paths[block_id], annotation_paths[block_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("block_id", type=int)
    args = parser.parse_args()
    check_block(args.block_id)
