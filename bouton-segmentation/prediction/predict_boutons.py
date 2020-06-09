import argparse
import json
import os

import luigi
from elf.io import open_file
from elf.wrapper.resized_volume import ResizedVolume
from elf.parallel import mean_and_std
from cluster_tools.inference import InferenceLocal, InferenceSlurm

ROOT = '../../data'
RAW_PATH = os.path.join(ROOT, 'rawdata/sbem-adult-1-lgn-raw.n5')
MASK_PATH = os.path.join(ROOT, 'rawdata/sbem-adult-1-lgn-mask.n5')
DEFAULT_CKPT = '../training/networks/V8/Weights'


def compute_mean_and_std():
    key = 'setup0/timepoint0/s1'
    f = open_file(RAW_PATH, 'r')
    ds = f[key]

    mask_key = 'setup0/timepoint0/s0'
    mask = open_file(MASK_PATH)[mask_key][:].astype('bool')
    mask = ResizedVolume(mask, ds.shape)

    m, s = mean_and_std(ds, mask=mask, n_threads=16, verbose=True)
    print("Computed mean and standard deviation:")
    print("Mean:", m)
    print("Standard deviation:", s)


def predict(ckpt, n_jobs, local=False):
    assert os.path.exists(ckpt)
    task = InferenceLocal if local else InferenceSlurm

    mask_key = 'setup0/timepoint0/s0'
    in_key = 'setup0/timepoint0/s1'

    out_path = '../../data.n5'
    out_key = {'predictions/affinities': [1, 7], 'predictions/foreground': [0, 1]}

    config_dir = 'configs'
    os.makedirs(config_dir, exist_ok=True)
    if local:
        device_mapping = {i: i for i in range(n_jobs)}
    else:
        device_mapping = None
    n_threads = 6

    # using smaller blocks / more halo, to increase border artifacts ?
    output_blocks = [32, 128, 128]
    halo = [16, 64, 64]

    tmp_folder = './tmp_inf'

    conf = InferenceLocal.default_global_config()
    conf.update({'block_shape': output_blocks})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    conf = InferenceLocal.default_task_config()

    mean = 149.95095144732198
    std = 49.999107303747465
    preprocess_kwargs = {'mean': mean, 'std': std}

    tlim = 60 * 72
    conf.update({'threads_per_job': n_threads, 'chunks': output_blocks,
                 'device_mapping': device_mapping, 'mem_limit': 32,
                 'time_limit': tlim, 'preprocess_kwargs': preprocess_kwargs,
                 'prep_model': 'add_sigmoid'})
    with open(os.path.join(config_dir, 'inference.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=n_jobs,
             input_path=RAW_PATH, input_key=in_key,
             output_path=out_path, output_key=out_key,
             mask_path=MASK_PATH, mask_key=mask_key,
             checkpoint_path=ckpt, halo=halo, framework='inferno')
    luigi.build([t], local_scheduler=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT, type=str)
    parser.add_argument('--compute_mean_and_std', type=int, default=0)
    parser.add_argument('--local', type=int, default=0)

    args = parser.parse_args()

    if bool(args.compute_mean_and_std):
        compute_mean_and_std()
    else:
        predict(args.ckpt, args.n_jobs, local=bool(args.local))
