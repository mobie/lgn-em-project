import napari
import numpy as np
from elf.io import open_file

from utils import parse_annotations, defects_to_bb


def check_annotation(data_path, defect_type, defect):

    resolution = np.array([0.04, 0.01, 0.01])
    bb, points = defects_to_bb(defect, resolution)

    key = "setup0/timepoint0/s0"
    with open_file(data_path, "r") as f:
        ds = f[key]
        raw = ds[bb]

    v = napari.Viewer()
    v.title = defect_type
    v.add_image(raw)
    v.add_points(points)
    napari.run()


def check_annotations(data_path, annotation_path):
    defect_annotations = parse_annotations(annotation_path)
    for defect_type, defects in defect_annotations.items():
        for defect in defects:
            check_annotation(data_path, defect_type, defect)


if __name__ == "__main__":
    data_path = "/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5"
    annotation_path = "./annotations/annotations2.txt"
    check_annotations(data_path, annotation_path)
