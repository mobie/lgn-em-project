import nifty.tools as nt
import z5py
from pybdv.metadata import get_resolution
from pybdv.util import get_scale_factors

CENTER_COORD = (335.96069414637003, 119.70199563151154, 33.14926510940082)

HALO_CHECK = [2.5, 5, 5]
HALO_SMALL = [5, 15, 15]
HALO_LARGE = [15, 30, 30]
HALOS = {"check": HALO_CHECK, "small": HALO_SMALL, "large": HALO_LARGE}

BLOCK_SHAPE = [32, 256, 256]


def get_halo(name):
    return HALOS[name]


def load_scale_factors(scale):
    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
    scale_factors = get_scale_factors(path, setup_id=0)
    return scale_factors[scale]


def load_resolution(scale=0):
    xml = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.xml'
    scale_factor = load_scale_factors(scale)
    res = get_resolution(xml, setup_id=0)
    res = [re * sf for re, sf in zip(res, scale_factor)]
    return res


def get_bounding_box(scale=0, halo=HALO_SMALL, intersect_with_blocking=False, return_as_lists=False):
    res = load_resolution(scale=scale)
    center = [int(ce / re) for ce, re in zip(CENTER_COORD[::-1], res)]
    halo_pix = [int(ha / re) for ha, re in zip(halo, res)]

    bb_start = [ce - ha for ce, ha in zip(center, halo_pix)]
    bb_stop = [ce + ha for ce, ha in zip(center, halo_pix)]

    if intersect_with_blocking:
        path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'
        with z5py.File(path, 'r') as f:
            shape = f[f'setup0/timepoint0/s{scale}'].shape
        blocking = nt.blocking([0, 0, 0], shape, BLOCK_SHAPE)
        block_list = blocking.getBlockIdsOverlappingBoundingBox(bb_start, bb_stop)

        for block_id in block_list:
            block = blocking.getBlock(block_id)
            block_start = block.begin
            block_stop = block.end
            for dim, (bstart, bstop) in enumerate(zip(block_start, block_stop)):
                if bstart < bb_start[dim]:
                    bb_start[dim] = bstart
                if bstop > bb_stop[dim]:
                    bb_stop[dim] = bstop

    if return_as_lists:
        return bb_start, bb_stop

    bb = tuple(slice(sta, sto) for sta, sto in zip(bb_start, bb_stop))
    return bb


def halo_to_pix(scale=0, halo=HALO_SMALL):
    res = load_resolution(scale)
    shape_pix = [2 * ha / re for ha, re in zip(halo, res)]
    print(shape_pix)


def check_center_coord():
    import z5py
    import napari
    path = '/g/rompani/lgn-em-datasets/data/0.0.0/images/local/sbem-adult-1-lgn-raw.n5'

    halo = [5, 10, 10]
    bb = get_bounding_box(halo=halo)

    with z5py.File(path, 'r') as f:
        ds = f['setup0/timepoint0/s0']
        ds.n_threads = 8
        raw = ds[bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)


if __name__ == '__main__':
    # halo_to_pix()
    # check_center_coord()

    bb1 = get_bounding_box()
    print(bb1)
    bb2 = get_bounding_box(intersect_with_blocking=True)
    print(bb2)
