import os
from mobie.xml_utils import copy_xml_as_n5_s3

ROOT = './data'


def add_xml_for_s3(vol_name):
    bucket_name = 'lgn-em'
    xml_path = os.path.join(f'data/0.0.0/images/local/sbem-adult-1-lgn-{vol_name}.xml')
    data_path = xml_path.replace('.xml', '.n5')
    xml_out_path = xml_path.replace('local', 'remote')
    path_in_bucket = os.path.relpath(data_path, start=ROOT)

    copy_xml_as_n5_s3(xml_path, xml_out_path,
                      service_endpoint='https://s3.embl.de',
                      bucket_name=bucket_name,
                      path_in_bucket=path_in_bucket,
                      authentication='Protected')

    print("In order to add the data to the EMBL S3, please run the following command:")
    full_s3_path = f'embl/{bucket_name}/{path_in_bucket}'
    mc_command = f"mc cp -r {os.path.relpath(data_path)}/ {full_s3_path}/"
    print(mc_command)


if __name__ == '__main__':
    # add_xml_for_s3('raw')
    add_xml_for_s3('boutons')
    add_xml_for_s3('mask')
    add_xml_for_s3('traces')
