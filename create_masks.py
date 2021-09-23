import os

import mobie
import numpy as np

from elf.io import open_file
from pybdv.metadata import get_resolution, get_data_path
from pybdv.util import get_scale_factors, get_key

HALO_MEDIUM = [5, 15, 15]
HALO_LARGE = [15, 30, 30]

CENTER_COORDS = [
    (43.360278649313315, 61.621676807202135, 22.91835987054154)[::-1],
    (121.88237052686952, 95.03083117664613, 62.75339947157563)[::-1],
    (335.96069414637003, 119.70199563151154, 33.14926510940082)[::-1]
]

MOBIE_ROOT = "/g/rompani/lgn-em-datasets/data"
DS_NAME = "0.0.0"

#
# creating masks for defect annotation
#


def get_shape_and_resolution(scale):
    raw_xml = "/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.xml"
    raw_data = get_data_path(raw_xml, return_absolute_path=True)
    resolution = get_resolution(raw_xml, setup_id=0)
    scale_factors = get_scale_factors(raw_data, setup_id=0)[scale]
    resolution = [res * sf for res, sf in zip(resolution, scale_factors)]

    with open_file(raw_data, "r") as f:
        shape = f[get_key(False, 0, 0, scale=scale)].shape

    return shape, resolution


def get_bb(center, resolution, halo):
    center = [int(ce / re) for ce, re in zip(center, resolution)]
    halo_pix = [int(ha / re) for ha, re in zip(halo, resolution)]

    bb_start = [ce - ha for ce, ha in zip(center, halo_pix)]
    bb_stop = [ce + ha for ce, ha in zip(center, halo_pix)]

    return tuple(slice(sta, sto) for sta, sto in zip(bb_start, bb_stop))


def create_mask(mask_name, center, halo, scale=5):
    ds_folder = os.path.join(MOBIE_ROOT, DS_NAME)
    ds_meta = mobie.metadata.read_dataset_metadata(ds_folder)
    if mask_name in ds_meta["sources"]:
        print("Mask", mask_name, "has already been added to the data-set.")
        return

    shape, resolution = get_shape_and_resolution(scale)
    mask = 255 * np.ones(shape, dtype="uint8")
    bb = get_bb(center, resolution, halo)
    mask[bb] = 0

    tmp_folder = "./tmp_masks"
    os.makedirs(tmp_folder, exist_ok=True)
    tmp_mask = os.path.join(tmp_folder, f"{mask_name}.h5")
    with open_file(tmp_mask, "a") as f:
        f.create_dataset("data", data=mask, compression="gzip")

    chunks = (64, 64, 64)
    scale_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
    mobie.add_image(tmp_mask, "data", MOBIE_ROOT, DS_NAME, mask_name,
                    resolution, scale_factors, chunks, menu_name="mask")


def create_medium_masks():
    print("Creating medium mask 1")
    create_mask("defect-annotation-1", CENTER_COORDS[0], HALO_MEDIUM)
    print("Creating medium mask 2")
    create_mask("defect-annotation-2", CENTER_COORDS[1], HALO_MEDIUM)


def create_large_mask():
    print("Creating large mask")
    create_mask("defect-annotation-large", CENTER_COORDS[2], HALO_LARGE)


def upload_masks():
    bucket_name = "lgn-em"
    service_endpoint = "https://s3.embl.de"

    mask_names = [
        "defect-annotation-1",
        "defect-annotation-2",
        "defect-annotation-large"
    ]

    ds_folder = os.path.join(MOBIE_ROOT, DS_NAME)
    ds_metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    sources = ds_metadata["sources"]

    new_file_formats = set([])
    for mask_name in mask_names:
        metadata = sources[mask_name]
        new_metadata, _ = mobie.metadata.add_remote_source_metadata(metadata, new_file_formats,
                                                                    ds_folder, DS_NAME,
                                                                    service_endpoint, bucket_name)
        sources[mask_name] = new_metadata
        mobie.metadata.upload_source(ds_folder, new_metadata, "bdv.n5", bucket_name)

    ds_metadata["sources"] = sources
    mobie.metadata.write_dataset_metadata(ds_folder, ds_metadata)


def create_all_masks():
    create_medium_masks()
    create_large_mask()
    upload_masks()


if __name__ == "__main__":
    create_all_masks()
