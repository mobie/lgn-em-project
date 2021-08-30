import os
from glob import glob
import numpy as np
from elf.segmentation import compute_rag
from elf.io import open_file
from skimage.measure import label
from tqdm import tqdm


def postprocess_seg(seg, min_bg_size=50):
    label_offset = int(seg.max()) + 1
    labeled = label(seg == 0)
    labeled[labeled > 0] += label_offset
    bg_ids = np.unique(labeled)[1:]
    seg += labeled
    assert 0 not in seg
    rag = compute_rag(seg, n_threads=8)

    n_merged = 0
    reset_to_zero = []
    for bg_id in tqdm(bg_ids, desc="Postprocessing seg"):
        ngbs = [ngb[0] for ngb in rag.nodeAdjacency(bg_id)]
        if len(ngbs) == 1:
            ngb_id = ngbs[0]
            seg[seg == bg_id] = ngb_id
            n_merged += 1
        else:
            reset_to_zero.append(bg_id)

    print("Merged", n_merged, "background pieces")
    seg[np.isin(seg, reset_to_zero)] = 0
    return seg


def check_data(raw, seg, name):
    import napari
    with napari.gui_qt():
        v = napari.Viewer()
        v.title = name
        v.add_image(raw)
        v.add_labels(seg)


def export_training_data_v1(check=True):
    raw_paths = {
        1: "./training_data/training_data_z481y16967x21745-z545y17479x22257.h5",
        2: "./training_data/training_data_z364y13744x11342-z428y14256x11854.h5",
        3: "./training_data/training_data_z485y17596x23654-z549y18108x24166.h5",
        4: "./training_data2/training_data_z1235y13855x14657-z1299y14367x15169.h5"
    }
    annotation_paths = {
        1: "./annotations/v1/2021-08-222_09-24__training_data_z481y16967x21745-z545y17479x22257__volume",
        2: "./annotations/v1/2021-08-222_09-25__training_data_z364y13744x11342-z428y14256x11854__volume",
        3: "./annotations/v1/2021-08-222_09-25__training_data_z485y17596x23654-z549y18108x24166__volume",
        4: "./annotations2/2021-08-222_13-35__training_data_z1235y13855x14657-z1299y14367x15169__volume"
    }

    input_folder = "../for_scalable_minds"
    output_folder = "./training_data/v1"
    os.makedirs(output_folder, exist_ok=True)

    for ii, (rp, ap) in enumerate(zip(raw_paths.values(), annotation_paths.values()), 1):
        rp = os.path.join(input_folder, rp)
        with open_file(rp, "r") as f:
            raw = f["raw"][:]
        ap = os.path.join(input_folder, ap)
        with open_file(ap, "r") as f:
            seg = f["*.tiff"][:]
        seg = postprocess_seg(seg)
        assert raw.shape == seg.shape

        if check:
            check_data(raw, seg)
        else:
            out_path = os.path.join(output_folder, f"block{ii}.h5")
            with open_file(out_path, "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("labels", data=seg, compression="gzip")


def get_paths():
    annotation_root = "../for_scalable_minds/annotations_v2"
    annotation_paths = glob(os.path.join(annotation_root, "*__volume"))

    raw_root = "../for_scalable_minds/training_data"
    raw_names = [
        os.path.split(p)[1].split("__")[1] for p in annotation_paths
    ]
    raw_paths = [
        os.path.join(raw_root, f"{name}.h5") for name in raw_names
    ]
    assert all(os.path.exists(rp) for rp in raw_paths)

    # annotation_paths[0] = os.path.join("../for_scalable_minds/annotations_v1/annotations/v1",
    #                                    "2021-08-222_09-25__training_data_z364y13744x11342-z428y14256x11854__volume")
    annotation_paths += [os.path.join("../for_scalable_minds/annotations_v1/annotations2",
                                      "2021-08-222_13-35__training_data_z1235y13855x14657-z1299y14367x15169__volume")]
    raw_paths += ["../for_scalable_minds/training_data2/training_data_z1235y13855x14657-z1299y14367x15169.h5"]
    assert len(raw_paths) == len(annotation_paths)

    return raw_paths, annotation_paths


def export_training_data_v2(check=True):
    raw_paths, annotation_paths = get_paths()
    output_folder = "./training_data/v2"
    os.makedirs(output_folder, exist_ok=True)

    for ii, (rp, ap) in enumerate(zip(raw_paths, annotation_paths), 1):
        with open_file(rp, "r") as f:
            raw = f["raw"][:]
        with open_file(ap, "r") as f:
            seg = f["*.tiff"][:]
        seg = postprocess_seg(seg)
        assert raw.shape == seg.shape

        name = os.path.split(rp)[1]
        if check:
            check_data(raw, seg, name)
        else:
            out_path = os.path.join(output_folder, f"block{ii}.h5")
            with open_file(out_path, "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("labels", data=seg, compression="gzip")
                f.attrs["name"] = name


def check_mismatched_annotations():
    l1 = "../for_scalable_minds/annotations_v2/2021-08-237_13-55__training_data_z364y13905x13264-z428y14417x13776__volume"
    with open_file(l1) as f:
        labs = f["*.tiff"]
    r1 = "../for_scalable_minds/training_data/training_data_z364y13744x11342-z428y14256x11854.h5"
    r2 = "../for_scalable_minds/training_data/training_data_z364y13905x13264-z428y14417x13776.h5"
    with open_file(r1, "r") as f:
        r1 = f["raw"][:]
    with open_file(r2, "r") as f:
        r2 = f["raw"][:]
    import napari
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(r1, name="13744")
        v.add_image(r2, name="13905")
        v.add_labels(labs)


# export_training_data_v2(check=False)
check_mismatched_annotations()
