import json
import numpy as np
from mobie.viewer_transformations import normalized_affine_to_position


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


def defects_to_bb(defect, resolution, min_bb_shape=(1, 512, 512)):
    points = np.array(defect)[:, ::-1]
    points /= resolution

    # TODO get the bounding box from the points in the defects, enlarge to MIN_BB_SHAPE if necessary
    min_ = np.min(points, axis=0).astype("int")
    max_ = np.max(points, axis=0).astype("int") + 1
    assert len(min_) == len(max_) == 3
    bb = [slice(mi, ma) for mi, ma in zip(min_, max_)]
    bb_shape = tuple(ma - mi for mi, ma in zip(min_, max_))

    bb_center = tuple((mi + ma) // 2 for mi, ma in zip(min_, max_))
    for i, (sh, mish, ce) in enumerate(zip(bb_shape, min_bb_shape, bb_center)):
        if sh < mish:
            bb[i] = slice(ce - mish // 2, ce + mish // 2)
            min_[i] = ce - mish // 2
    bb = tuple(bb)

    points -= min_
    points = points.astype("int")

    return bb, points
