import os
import mobie
import subprocess
from common import get_bounding_box, get_halo
from mobie.import_data.utils import downscale, add_max_id
from mobie.metadata.remote_metadata import _to_bdv_s3
from mobie.xml_utils import parse_s3_xml
from pybdv.metadata import get_data_path, get_name, get_unit

ROOT = "/g/rompani/lgn-em-datasets/data"
DS_NAME = "0.0.0"
DATA = "./data.n5"
TMP = "/scratch/pape/lgn/tmp_mobie"

RESOLUTION = [0.04, 0.01, 0.01]
CHUNKS = (32, 256, 256)
SCALE_FACTORS = [
    [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]
]


def _upload(ds_folder, storage, bucket_name):
    xml = os.path.join(ds_folder, storage["bdv.n5"]["relativePath"])
    data_path = get_data_path(xml, return_absolute_path=True)
    assert os.path.exists(data_path), data_path

    xml_s3 = os.path.join(ds_folder, storage["bdv.n5.s3"]["relativePath"])
    s3_path = parse_s3_xml(xml_s3)[0]
    dest = f"embl/{bucket_name}/{s3_path}/"

    cmd = ["mc", "cp", "-r", f"{data_path}/", dest]
    print("Copy to s3:", cmd)
    subprocess.run(cmd)


def add_seg(seg_name, upload_to_s3):
    mobie.add_segmentation(
        DATA, "segmentation/multicut", ROOT, DS_NAME, seg_name,
        RESOLUTION, SCALE_FACTORS, CHUNKS, menu_name="segmentation",
        tmp_folder=TMP, target="local", max_jobs=16, add_default_table=False
    )

    if upload_to_s3:
        service_endpoint = "https://s3.embl.de"
        bucket_name = "lgn-em"
        region = "us-west-2"
        ds_folder = os.path.join(ROOT, DS_NAME)
        metadata = mobie.metadata.read_dataset_metadata(ds_folder)
        sources = metadata["sources"]
        source = sources[seg_name]

        storage = source["segmentation"]["imageData"]
        new_format, s3_storage = _to_bdv_s3("bdv.n5", ds_folder, DS_NAME, storage["bdv.n5"],
                                            service_endpoint, bucket_name, region)
        storage[new_format] = s3_storage
        source["segmentation"]["imageData"] = storage
        sources[seg_name] = source
        metadata["sources"] = sources
        mobie.metadata.write_dataset_metadata(ds_folder, metadata)

        _upload(ds_folder, storage, bucket_name)


def update_seg(source, roi_begin, roi_end, upload_to_s3):
    xml_path = os.path.join(
        ROOT, DS_NAME, source["segmentation"]["imageData"]["bdv.n5"]["relativePath"]
    )
    assert os.path.exists(xml_path), xml_path

    seg_path = get_data_path(xml_path, return_absolute_path=True)
    assert os.path.exists(seg_path), seg_path
    source_name = get_name(xml_path, setup_id=0)
    unit = get_unit(xml_path, setup_id=0)

    seg_in_key = "segmentation/multicut"
    tmp_folder = os.path.join(TMP, "tmp_update_seg")
    downscale(DATA, seg_in_key, seg_path,
              RESOLUTION, SCALE_FACTORS, CHUNKS,
              tmp_folder, target="local", max_jobs=16,
              block_shape=None,
              library="vigra", library_kwargs={"order": 0},
              unit=unit, source_name=source_name)

    add_max_id(DATA, seg_in_key, seg_path, "setup0/timepoint0/s0",
               tmp_folder=tmp_folder, target="local", max_jobs=16)

    if upload_to_s3:
        ds_folder = os.path.join(ROOT, DS_NAME)
        storage = source["segmentation"]["imageData"]
        bucket_name = "lgn-em"
        _upload(ds_folder, storage, bucket_name)


def add_seg_to_mobie(halo_name, upload_to_s3):
    ds_folder = os.path.join(ROOT, DS_NAME)
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    sources = metadata["sources"]

    seg_name = "neurons"

    if seg_name in sources:
        roi_begin, roi_end = get_bounding_box(return_as_lists=True, halo=get_halo(halo_name))
        update_seg(sources[seg_name], roi_begin, roi_end, upload_to_s3)
    else:
        add_seg(seg_name, upload_to_s3)


if __name__ == "__main__":
    add_seg_to_mobie(halo_name="large", upload_to_s3=True)
