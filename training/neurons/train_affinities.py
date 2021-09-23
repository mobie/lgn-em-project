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


def get_loader(is_train, patch_shape, batch_size=1, n_samples=None, with_defect_augmentation=True):
    raw_key = "raw"
    label_key = "labels"
    if is_train:
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        ids = [9, 10]
    data_paths = [f"./training_data/v3/block{ii}.h5" for ii in ids]
    label_transform = torch_em.transform.label.AffinityTransform(offsets=OFFSETS,
                                                                 ignore_label=0,
                                                                 add_binary_target=False,
                                                                 add_mask=True,
                                                                 include_ignore_transitions=True)
    if with_defect_augmentation:
        defect_path = "./training_data/defects.h5"
        patch_shape_2d = (1,) + patch_shape[1:]
        artifact_source = torch_em.transform.get_artifact_source(
            defect_path, patch_shape_2d, min_mask_fraction=0.5,
            raw_key="defect_sections/raw", mask_key="defect_sections/mask"
        )
        defect_trafo = torch_em.transform.EMDefectAugmentation(
            p_drop_slice=0.02,
            p_low_contrast=0.02,
            p_deform_slice=0.02,
            p_paste_artifact=0.02,
            artifact_source=artifact_source
        )
        raw_transform = torch_em.transform.get_raw_transform(
            augmentation2=defect_trafo
        )
    else:
        raw_transform = None

    return torch_em.default_segmentation_loader(
        data_paths, raw_key,
        data_paths, label_key,
        patch_shape=patch_shape,
        batch_size=batch_size,
        raw_transform=raw_transform,
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
    patch_shape = (32, 320, 320)

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
    patch_shape = (32, 256, 256)
    if train:
        print("Check train loader")
        loader = get_loader(True, patch_shape,
                            with_defect_augmentation=bool(args.defect_augmentation))
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader(False, patch_shape,
                            with_defect_augmentation=bool(args.defect_augmentation))
        check_loader(loader, n_images)


# TODO advanced traning, compare "torch_em/experiments/neuron_segmentation/cremi"
# - more defect and mis-alignment augmentations
# -- alignment jitter
# - more augmentations
# -- elastic
if __name__ == '__main__':
    parser = parser_helper()
    parser.add_argument("--defect_augmentation", "-d", default=1)
    args = parser.parse_args()
    if args.check:
        check(args, train=True, val=True)
    else:
        train_affinities(args)
