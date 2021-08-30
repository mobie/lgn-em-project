import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask
from torch_em.util import parser_helper

OFFSETS = [
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    [-2, 0, 0], [0, -3, 0], [0, 0, -3],
    [-3, 0, 0], [0, -9, 0], [0, 0, -9],
    [-4, 0, 0], [0, -27, 0], [0, 0, -27]
]


def get_loader(is_train, patch_shape, batch_size=1, n_samples=None):
    raw_key = "raw"
    label_key = "labels"
    if is_train:
        ids = [1, 2, 3, 4, 6]
    else:
        ids = [7]
    data_paths = [f"./training_data/v2/block{ii}.h5" for ii in ids]
    label_transform = torch_em.transform.label.AffinityTransform(offsets=OFFSETS,
                                                                 ignore_label=0,
                                                                 add_binary_target=False,
                                                                 add_mask=True,
                                                                 include_ignore_transitions=True)
    return torch_em.default_segmentation_loader(
        data_paths, raw_key,
        data_paths, label_key,
        patch_shape=patch_shape,
        batch_size=batch_size,
        label_transform2=label_transform
    )


def get_model():
    n_out = len(OFFSETS)
    model = AnisotropicUNet(
        scale_factors=[
            [1, 2, 2],
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]
        ],
        in_channels=1,
        out_channels=n_out,
        initial_features=32,
        gain=2,
        final_activation='Sigmoid'
    )
    return model


def train_affinities(args):
    model = get_model()
    patch_shape = [32, 320, 320]

    train_loader = get_loader(
        is_train=True,
        patch_shape=patch_shape,
        n_samples=1000
    )
    val_loader = get_loader(
        is_train=False,
        patch_shape=patch_shape,
        n_samples=50
    )

    loss = LossWrapper(loss=DiceLoss(),
                       transform=ApplyAndRemoveMask())

    name = "affinity_model"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50
    )

    if args.from_checkpoint:
        trainer.fit(args.n_iterations, 'latest')
    else:
        trainer.fit(args.n_iterations)


def check(args, train=True, val=True, n_images=2):
    from torch_em.util.debug import check_loader
    patch_shape = [32, 256, 256]
    if train:
        print("Check train loader")
        loader = get_loader(True, patch_shape)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader(False, patch_shape)
        check_loader(loader, n_images)


# TODO advanced traning, compare "torch_em/experiments/neuron_segmentation/cremi"
# - more defect and mis-alignment augmentations
# -- pasting defect patches
# -- simulating contrast defect
# -- simulating tear defect
# -- alignment jitter
# - more augmentations
# -- elastic
if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    if args.check:
        check(args, train=True, val=True)
    else:
        train_affinities(args)
