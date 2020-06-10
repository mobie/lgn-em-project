import argparse
import mobie


# TODO initialize the dataset from the knossos file
def initialize_mobie():
    pass


# TODO replace with the actual code to generate the mask
def add_mask():
    mask_path = '../pape/lgn/data.n5'
    mask_key = 'mask/s6'
    mask_name = 'sbem-adult-1-lgn-mask'

    # the mask is downscaled by factors:
    mask_scale_factor = [0.04, 0.01, 0.01]
    raw_resolution = [8, 32, 32]
    resolution = [factor * res for factor, res in zip(mask_scale_factor, raw_resolution)]
    print()
    print("Resolution", resolution)
    print()

    scale_factors = 3 * [[2, 2, 2]]
    chunks = [64, 64, 64]

    mobie.add_mask(mask_path, mask_key,
                   root='./data', dataset_name='0.0.0', mask_name=mask_name,
                   resolution=resolution, scale_factors=scale_factors, chunks=chunks,
                   target='local', max_jobs=12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', type=int, default=0)
    parser.add_argument('--add_mask', type=int, default=0)

    args = parser.parse_args()
    if bool(args.initialize):
        initialize_mobie()
    if bool(args.add_mask):
        add_mask()
