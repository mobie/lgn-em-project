import numpy as np
import vigra

from elf.segmentation.features import compute_rag
from elf.segmentation.watershed import stacked_watershed
from elf.segmentation.utils import normalize_input
from mipnet.utils.prediction import predict_with_halo, normalize

try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff


def refine_seg(raw, seg):
    model_path = '/g/kreshuk/pape/Work/mobie/lgn-em-datasets/training/neurons/networks/v1/Weights'
    print("Start prediction ...")
    pred = predict_with_halo(raw, model_path, gpus=['cpu'],
                             inner_block_shape=[32, 128, 128],
                             outer_block_shape=[48, 192, 192],
                             preprocess=normalize)
    pred = pred[1]
    return pred

    # make watershed
    ws = stacked_watershed(pred, threshold=.5, sigma_seeds=1.)
    rag = compute_rag()

    # make seeds and map them to edges

    # run multicut


def update_hmap(raw, hmap, invert):
    if invert:
        intensities = normalize_input(raw.max() - raw)
    else:
        intensities = normalize_input(raw)
    return normalize_input(intensities * hmap)


def em_hmap(raw, sigma, sigma2=None, invert=True):
    """ This heightmap works well for 2d boundaries in EM.
    """
    hmap = normalize_input(ff.gaussianGradientMagnitude(raw, sigma))
    sigma2 = sigma if sigma2 is None else sigma2
    hmap = ff.hessianOfGaussianEigenvalues(hmap, sigma2)[..., 0]
    hmap = update_hmap(raw, hmap, invert=invert)
    return hmap


def boundaries_to_seg(raw, seg):
    hmap = np.zeros(raw.shape, dtype='float32')
    for z in range(hmap.shape[0]):
        hmap[z] = em_hmap(normalize_input(raw[z]), sigma=2)

    seg[seg == -1] = 0
    seg = seg.astype('uint32')

    seeds = vigra.analysis.labelVolumeWithBackground(seg)
    seg, _ = vigra.analysis.watershedsNew(hmap, seeds=seeds)

    return seg
