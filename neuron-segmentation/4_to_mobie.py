import os
import mobie
import subprocess
from common import get_bounding_box, get_halo
from mobie.metadata.remote_metadata import _to_bdv_s3
from mobie.xml_utils import parse_s3_xml
from pybdv.metadata import get_data_path

ROOT = "/g/rompani/lgn-em-datasets/data"
DS_NAME = "0.0.0"
DATA = "./data.n5"
TMP = "/scratch/pape/lgn/tmp_mobie"


def add_seg(seg_name, upload_to_s3):
    resolution = [0.04, 0.01, 0.01]
    chunks = (32, 256, 256)
    scale_factors = [
        [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]
    ]
    mobie.add_segmentation(
        DATA, "segmentation/multicut", ROOT, DS_NAME, seg_name,
        resolution, scale_factors, chunks, menu_name="segmentation",
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

        xml = os.path.join(ds_folder, storage["bdv.n5"]["relativePath"])
        data_path = get_data_path(xml, return_absolute_path=True)
        assert os.path.exists(data_path), data_path

        xml_s3 = os.path.join(ds_folder, storage["bdv.n5.s3"]["relativePath"])
        s3_path = parse_s3_xml(xml_s3)[0]
        dest = f"embl/{bucket_name}/{s3_path}/"

        cmd = ["mc", "cp", "-r", f"{data_path}/", dest]
        print("Copy to s3:", cmd)
        subprocess.run(cmd)


# TODO implement update mechanism
def update_seg(seg_name, roi_begin, roi_end, upload_to_s3):
    raise NotImplementedError


def add_seg_to_mobie(halo_name, upload_to_s3):
    ds_folder = os.path.join(ROOT, DS_NAME)
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    sources = metadata["sources"]

    seg_name = "neurons"

    if seg_name in sources:
        roi_begin, roi_end = get_bounding_box(return_as_lists=True, halo=get_halo(halo_name))
        update_seg(seg_name, roi_begin, roi_end, upload_to_s3)
    else:
        add_seg(seg_name, upload_to_s3)


if __name__ == "__main__":
    add_seg_to_mobie(halo_name="large", upload_to_s3=True)
