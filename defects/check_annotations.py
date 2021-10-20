import json

import napari
from elf.io import open_file
from mobie.viewer_transformations import normalized_affine_to_position

MIN_BB_SHAPE = [1, 512, 512]


def parse_position(line):
    pos = json.loads(line)
    affine = pos["normalizedAffine"]
    return normalized_affine_to_position(affine)


def parse_annotations(annotation_path):

    def isint(string):
        try:
            int(string)
            return True
        except Exception:
            return False

    defects = {}
    defect_type = None
    this_defects = None

    with open(annotation_path) as f:
        for line in f:
            line = line.rstrip("\n")

            if len(line) > 2 and line[1] == ")":
                if defect_type is not None:
                    assert this_defects is not None
                    defects[defect_type] = this_defects

                defect_type = line[3:]
                this_defects = []
                current_defect = None

            if isint(line.rstrip(".")):
                if current_defect is not None:
                    this_defects.append(current_defect)
                current_defect = []

            if line.startswith("{"):
                position = parse_position(line)
                current_defect.append(position)

        defects[defect_type] = this_defects

    return defects


def check_annotation(data_path, defect_type, defect):

    # TODO get the bounding box from the points in the defects, enlarge to MIN_BB_SHAPE if necessary
    bb = ""

    # TODO bring the defect annotation points into the local coordinate system by subtracting the bb corner
    points = ""

    key = "setup0/timepoint0/s0"
    with open_file(data_path, "r") as f:
        ds = f[key]
        raw = ds[bb]

    v = napari.Viewer()
    v.title = defect_type
    v.add_image(raw)
    v.add_points(points)


def check_annotations(data_path, annotation_path):
    defect_annotations = parse_annotations(annotation_path)
    for defect_type, defects in defect_annotations.items():
        for defect in defects:
            check_annotation(data_path, defect_type, defect)


# Annotation times:
# small blocks: ~ 3hrs each
# large block: 9-10hrs
if __name__ == "__main__":
    data_path = "/g/rompani/"
    annotation_path = "./annotations/annotations1.txt"
    check_annotations(data_path, annotation_path)
