import argparse
import json
import os

import mobie

ROOT = '../../data'


def add_default_table(version, seg_name, target, n_jobs, resolution):

    version_folder = os.path.join(ROOT, version)
    seg_path = os.path.join(version_folder, 'images', 'local', f'{seg_name}.n5')
    seg_key = 'setup0/timepoint0/s0'

    tmp_folder = f'tmp_table_{seg_name}'

    out_path = os.path.join(version_folder, 'tables', seg_name, 'default.csv')
    mobie.tables.compute_default_table(seg_path, seg_key,
                                       out_path, resolution,
                                       tmp_folder=tmp_folder,
                                       target=target,
                                       max_jobs=n_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='0.0.0')
    parser.add_argument('--seg_name', type=str, default='sbem-adult-1-lgn-boutons')
    parser.add_argument('--target', default='slurm')
    parser.add_argument('--n_jobs', default=250, type=int)
    parser.add_argument('--resolution', default='[0.04,0.02,0.02]')

    args = parser.parse_args()
    resolution = json.loads(args.resolution)
    add_default_table(args.version, args.seg_name, args.target, args.n_jobs,
                      resolution)
