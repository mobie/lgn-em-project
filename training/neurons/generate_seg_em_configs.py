import os
from glob import glob
import yaml
from inferno.utils.io_utils import yaml2dict


def get_seg_em_paths(val_fraction):
    pattern = '/g/kreshuk/data/helmstaedter/training_data/neurons/rompani/*.h5'
    files = glob(pattern)
    files.sort()
    n_val = int(val_fraction * len(files))
    train_paths = files[:-n_val]
    val_paths = files[-n_val:]
    return train_paths, val_paths


def update_paths(config, paths):
    names = [os.path.split(pp)[1] for pp in paths]
    names = [int(os.path.splitext(nn)[0]) for nn in names]

    config['names'] = names

    vol_config = config['volume_config']
    raw_config = vol_config['raw']
    seg_config = vol_config['segmentation']

    for name, path in zip(names, paths):
        raw_config['path'][name] = path
        raw_config['path_in_file'][name] = 'raw'

        seg_config['path'][name] = path
        seg_config['path_in_file'][name] = 'labels'

    vol_config['raw'] = raw_config
    vol_config['segmentation'] = seg_config
    config['volume_config'] = vol_config

    return config


def generate_seg_em_configs():
    train_paths, val_paths = get_seg_em_paths(val_fraction=.10)

    train_config = yaml2dict('./configs_seg_em/data_config_template.yml')
    train_config = update_paths(train_config, train_paths)
    with open('./configs_seg_em/data_config.yml', 'w') as f:
        yaml.dump(train_config, f)

    val_config = yaml2dict('./configs_seg_em/validation_config_template.yml')
    val_config = update_paths(val_config, val_paths)
    with open('./configs_seg_em/validation_config.yml', 'w') as f:
        yaml.dump(val_config, f)


if __name__ == '__main__':
    generate_seg_em_configs()
