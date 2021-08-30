#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch16/bin/python

import os
import sys
import logging
import argparse
import yaml

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.essentials import GarbageCollection
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
from inferno.io.transform.base import Compose
from inferno.extensions.criteria import SorensenDiceLoss

from neurofire.criteria.loss_wrapper import LossWrapper
from neurofire.metrics.arand import ArandErrorFromMWS
from neurofire.criteria.loss_transforms import (ApplyAndRemoveMask,
                                                InvertTarget,
                                                RemoveSegmentationFromTarget)

import mipnet.models.unet as models
from mipnet.datasets import get_cremi_loader


logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def dice_loss(is_val=False):
    print("Build Dice loss")
    if is_val:
        trafos = [RemoveSegmentationFromTarget(),
                  ApplyAndRemoveMask(), InvertTarget()]
    else:
        trafos = [ApplyAndRemoveMask(), InvertTarget()]
    trafos = Compose(*trafos)
    return LossWrapper(criterion=SorensenDiceLoss(),
                       transforms=trafos)


def mws_metric():
    strides = [4, 4, 4]
    metric = ArandErrorFromMWS(offsets=get_offsets(),
                               strides=strides,
                               randomize_strides=True,
                               average_slices=False)
    return metric


def set_up_training(project_directory,
                    config,
                    data_config,
                    load_pretrained_model):
    # Get model
    if load_pretrained_model:
        model = Trainer().load(from_directory=project_directory,
                               filename='Weights/checkpoint.pytorch').model
    else:
        model_name = config.get('model_name')
        model = getattr(models, model_name)(**config.get('model_kwargs'))

    loss = dice_loss()
    loss_val = dice_loss(is_val=True)
    metric = mws_metric()
    # metric = loss_val

    # Build trainer and validation metric
    logger.info("Building trainer.")
    smoothness = 0.9

    trainer = Trainer(model)\
        .save_every((1000, 'iterations'),
                    to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss)\
        .build_validation_criterion(loss_val)\
        .build_optimizer(**config.get('training_optimizer_kwargs'))\
        .evaluate_metric_every('never')\
        .validate_every((100, 'iterations'), for_num_iterations=5)\
        .register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))\
        .build_metric(metric)\
        .register_callback(AutoLR(factor=0.98,
                                  patience='100 iterations',
                                  monitor_while='validating',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous'))\
        .register_callback(GarbageCollection())\

    logger.info("Building logger.")
    # Build logger
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every=(100, 'iterations'),
                                    log_histograms_every='never').observe_states(
        ['validation_input', 'validation_prediction, validation_target'],
        observe_while='validating'
    )

    trainer.build_logger(tensorboard,
                         log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


def load_checkpoint(project_directory):
    logger.info("Trainer from checkpoint")
    trainer = Trainer().load(from_directory=os.path.join(project_directory, "Weights"))
    return trainer


def training(project_directory,
             train_configuration_file,
             data_configuration_file,
             validation_configuration_file,
             max_training_iters=int(1e5),
             from_checkpoint=False,
             load_pretrained_model=False):

    assert not (from_checkpoint and load_pretrained_model)

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_cremi_loader(data_configuration_file)

    logger.info("Loading validation data loader from %s." % validation_configuration_file)
    validation_loader = get_cremi_loader(validation_configuration_file)

    # load network and training progress from checkpoint
    if from_checkpoint:
        trainer = load_checkpoint(project_directory)
    else:
        data_config = yaml2dict(data_configuration_file)
        trainer = set_up_training(project_directory,
                                  config,
                                  data_config,
                                  load_pretrained_model)

    trainer.set_max_num_iterations(max_training_iters)

    # Bind loader
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader).bind_loader('validate', validation_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))
        trainer.mixed_precision = True

    # Go!
    logger.info("Lift off!")
    trainer.fit()


def make_train_config(config_folder, train_config_file, gpus, affinity_config):
    template = f'./{config_folder}/train_config.yml'

    activation = 'Sigmoid'
    n_out = len(affinity_config['offsets'])

    template = yaml2dict(template)
    template['model_kwargs']['out_channels'] = n_out
    template['model_kwargs']['final_activation'] = activation
    template['devices'] = gpus

    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


def make_data_config(config_folder, data_config_file, n_batches,  affinity_config):
    template = yaml2dict(f'./{config_folder}/data_config.yml')
    template['loader_config']['batch_size'] = n_batches
    template['loader_config']['num_workers'] = 8 * n_batches
    template['volume_config']['segmentation']['affinity_config'] = affinity_config
    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


def make_validation_config(config_folder, validation_config_file, affinity_config):
    template = yaml2dict(f'./{config_folder}/validation_config.yml')
    template['volume_config']['segmentation']['affinity_config'] = affinity_config
    affinity_config.update({'retain_segmentation': True})
    with open(validation_config_file, 'w') as f:
        yaml.dump(template, f)


def copy_train_file(project_directory):
    from shutil import copyfile
    file_path = os.path.abspath(__file__)
    dst = os.path.join(project_directory, 'train.py')
    copyfile(file_path, dst)


def get_offsets():
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-2, 0, 0], [0, -3, 0], [0, 0, -3],
            [-3, 0, 0], [0, -9, 0], [0, 0, -9],
            [-4, 0, 0], [0, -18, 0], [0, 0, -18]]


# TODO defect augmentations
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('version', type=str)
    parser.add_argument('config_folder', type=str)
    parser.add_argument('--learn_ignore_transitions', type=int, default=0)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))
    parser.add_argument('--from_checkpoint', type=int, default=0)
    parser.add_argument('--load_network', type=int, default=0)

    args = parser.parse_args()

    project_directory = os.path.join('networks', args.version)
    os.makedirs(project_directory, exist_ok=True)

    gpus = list(args.gpus)
    # set the proper CUDA_VISIBLE_DEVICES env variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    train_config = os.path.join(project_directory, 'train_config.yml')
    validation_config = os.path.join(project_directory, 'validation_config.yml')
    data_config = os.path.join(project_directory, 'data_config.yml')

    affinity_config = {
        'retain_mask': True,
        'segmentation_to_binary': False,
        'offsets': get_offsets(),
        # 'smoothing_config': {'sigma': 1.},
        'ignore_label': 0,
        'learn_ignore_transitions': bool(args.learn_ignore_transitions)
    }

    # only copy files to project directory if we DON'T load from checkpoint
    if not bool(args.from_checkpoint):
        config_folder = args.config_folder
        make_train_config(config_folder, train_config, gpus, affinity_config)
        make_data_config(config_folder, data_config, len(gpus), affinity_config)
        make_validation_config(config_folder, validation_config, affinity_config)
        copy_train_file(project_directory)

    training(project_directory,
             train_config,
             data_config,
             validation_config,
             max_training_iters=args.max_train_iters,
             from_checkpoint=bool(args.from_checkpoint),
             load_pretrained_model=bool(args.load_network))


if __name__ == '__main__':
    main()
