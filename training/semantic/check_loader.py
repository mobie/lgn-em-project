import napari
import torch

from mipnet.datasets import get_cremi_loader
from mipnet.transforms import SemanticTargetTrafo
from neurofire.criteria.loss_transforms import ApplyAndRemoveMask


def check_loader(n_batches=1, with_trafo=False, remove_mask=False):

    loader = get_cremi_loader('./configs/validation_config.yml')
    trafo1 = SemanticTargetTrafo([1, 2, 3], torch.float32, ignore_label=-1)
    trafo2 = ApplyAndRemoveMask()

    for ii, (x, y) in enumerate(loader):

        pred_shape = (x.shape[0], 3) + x.shape[2:]
        pred = torch.rand(*pred_shape)
        if with_trafo:
            pred, y = trafo1(pred, y)
            if remove_mask:
                pred, y = trafo2(pred, y)

        x = x.numpy().squeeze()
        y = y.numpy().squeeze()
        pred = pred.numpy().squeeze()

        with napari.gui_qt():
            v = napari.Viewer()
            v.add_image(x)
            v.add_image(pred)
            v.add_labels(y)

        if ii >= n_batches:
            break


if __name__ == '__main__':
    check_loader(with_trafo=True, remove_mask=True)
