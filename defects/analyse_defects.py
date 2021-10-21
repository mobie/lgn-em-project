from elf.io import open_file
from utils import parse_annotations

FRAC_FG = 0.5785  # calculated with get_fg_frac


def get_fg_frac():
    path = "/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-mask.n5"
    with open_file(path, "r") as f:
        ds = f["setup0/timepoint0/s0"]
        ds.n_threads = 16
        data = ds[:]
    frac_fg = (data == 1).sum() / float(data.size)
    print("Foregrund fraction:", frac_fg)


def analyse_defects(data_path, annotation_path, time_estimate):
    defect_annotations = parse_annotations(annotation_path)
    total_defects = 0
    for name, defects in defect_annotations.items():
        print("Defects of type", name, ":", len(defects))
        total_defects += len(defects)
    print("Total number of defects:", total_defects)

    with open_file(data_path, "r") as f:
        ds = f["setup0/timepoint0/s0"]
        ds.n_threads = 16
        data = ds[:]

    n_vox = (data == 0).sum()
    n_vox_tot = FRAC_FG * data.size

    full_time_estimate = n_vox_tot / n_vox * time_estimate
    print("Estimate:", full_time_estimate, "hrs")


def analyse_annotations1():
    print("Analyse defect annotations 1")
    analyse_defects("/g/rompani/lgn-em-datasets/data/0.0.0/images/bdv-n5/defect-annotation-1.n5",
                    "./annotations/annotations1.txt", time_estimate=3)
    print()


def analyse_annotations2():
    print("Analyse defect annotations 2")
    analyse_defects("/g/rompani/lgn-em-datasets/data/0.0.0/images/bdv-n5/defect-annotation-2.n5",
                    "./annotations/annotations2.txt", time_estimate=3)
    print()


def analyse_annotations_large():
    print("Analyse defect annotations large")
    analyse_defects("/g/rompani/lgn-em-datasets/data/0.0.0/images/bdv-n5/defect-annotation-large.n5",
                    "./annotations/annotations_large.txt", time_estimate=9.5)
    print()


# Annotation times:
# small blocks: ~ 3hrs each
# large block: 9-10hrs
if __name__ == "__main__":
    # get_fg_frac()
    analyse_annotations1()
    analyse_annotations2()
    analyse_annotations_large()
