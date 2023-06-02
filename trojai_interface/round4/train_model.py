# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import time
import logging.config
import multiprocessing
from numpy.random import RandomState

import torch
import torchvision.transforms.functional

import trojai.modelgen.architecture_factory
import trojai.modelgen.data_manager
import trojai.modelgen.model_generator
import trojai.modelgen.config
import trojai.modelgen.adversarial_pgd_optimizer
import trojai.modelgen.adversarial_fbf_optimizer

import dataset
import model_factories
import round_config

logger = logging.getLogger(__name__)

# Define the transforms which are applied to the training data by pytorch
MY_TRAIN_XFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    torchvision.transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    torchvision.transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    torchvision.transforms.ToTensor()])  # ToTensor performs min-max normalization

# Define the transforms which are applied to the test data by pytorch
MY_TEST_XFORMS = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.CenterCrop(size=(224, 224)),
    torchvision.transforms.ToTensor()])  # ToTensor performs min-max normalization


def train_img_transform(x):
    """
    Callable function handle to apply the train transforms
    :param x: image data to apply the transformations to
    :return: the transformed images
    """
    x = MY_TRAIN_XFORMS.__call__(x)
    return x


def test_img_transform(x):
    """
    Callable function handle to apply the test transforms
    :param x: image data to apply the transformations to
    :return: the transformed images
    """
    x = MY_TEST_XFORMS.__call__(x)
    return x


def train_model(config: round_config.RoundConfig):
    """
    Function to train a model instance from the round config.
    :param config: The round config defining the model to be trained.
    :return:
    """
    master_RSO = RandomState(config.master_seed)
    train_rso = RandomState(master_RSO.randint(2 ** 31 - 1))
    test_rso = RandomState(master_RSO.randint(2 ** 31 - 1))

    arch = model_factories.get_factory(config.model_architecture)
    if arch is None:
        logger.warning('Invalid Architecture type: {}'.format(config.model_architecture))
        raise IOError('Invalid Architecture type: {}'.format(config.model_architecture))

    # default to all the cores
    num_avail_cpus = multiprocessing.cpu_count()
    try:
        # if slurm is found use the cpu count it specifies
        num_avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        pass  # do nothing

    shm_train_dataset = dataset.TrafficDataset(config, train_rso, config.number_training_samples, class_balanced=True)
    shm_test_dataset = dataset.TrafficDataset(config, test_rso, config.number_test_samples, class_balanced=True)

    # construct the image data in memory
    start_time = time.time()
    shm_train_dataset.build_dataset()
    logger.info('Building in-mem train dataset took {} s'.format(time.time() - start_time))
    start_time = time.time()
    shm_test_dataset.build_dataset()
    logger.info('Building in-mem test dataset took {} s'.format(time.time() - start_time))

    train_dataset = shm_train_dataset.get_dataset(data_transform=train_img_transform)
    clean_test_dataset = shm_test_dataset.get_clean_dataset(data_transform=test_img_transform)

    dataset_obs = dict(train=train_dataset, clean_test=clean_test_dataset)

    if config.poisoned:
        poisoned_test_dataset = shm_test_dataset.get_poisoned_dataset(data_transform=test_img_transform)
        dataset_obs['triggered_test'] = poisoned_test_dataset

    num_cpus_to_use = int(.8 * num_avail_cpus)
    data_obj = trojai.modelgen.data_manager.DataManager(config.data_filepath,
                                                        None,
                                                        None,
                                                        data_type='custom',
                                                        custom_datasets=dataset_obs,
                                                        shuffle_train=True,
                                                        train_dataloader_kwargs={'num_workers': num_cpus_to_use, 'shuffle': True},
                                                        test_dataloader_kwargs={'num_workers': num_cpus_to_use, 'shuffle': False})

    model_save_dir = os.path.join(config.data_filepath, 'model')
    stats_save_dir = os.path.join(config.data_filepath, 'model')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    default_nbpvdm = None if device.type == 'cpu' else 500

    early_stopping_argin = None
    if config.early_stopping_epoch_count is not None:
        early_stopping_argin = trojai.modelgen.config.EarlyStoppingConfig(num_epochs=config.early_stopping_epoch_count, val_loss_eps=config.loss_eps)

    training_params = trojai.modelgen.config.TrainingConfig(device=device,
                                                            epochs=100,
                                                            batch_size=config.batch_size,
                                                            lr=config.learning_rate,
                                                            optim='adam',
                                                            objective='cross_entropy_loss',
                                                            early_stopping=early_stopping_argin,
                                                            train_val_split=config.validation_split,
                                                            save_best_model=True)
    reporting_params = trojai.modelgen.config.ReportingConfig(num_batches_per_logmsg=100,
                                                              disable_progress_bar=True,
                                                              num_epochs_per_metric=1,
                                                              num_batches_per_metrics=default_nbpvdm,
                                                              experiment_name=config.model_architecture)

    if config.adversarial_training_method == "None":
        logger.info('Using DefaultOptimizer')
        optimizer_cfg = trojai.modelgen.config.DefaultOptimizerConfig(training_params, reporting_params)
    elif config.adversarial_training_method == "FBF":
        logger.info('Using FBFOptimizer')
        opt_config = trojai.modelgen.config.DefaultOptimizerConfig(training_params, reporting_params)
        optimizer_cfg = trojai.modelgen.adversarial_fbf_optimizer.FBFOptimizer(opt_config)
        training_params.adv_training_eps = config.adversarial_eps
        training_params.adv_training_ratio = config.adversarial_training_ratio
    elif config.adversarial_training_method == "PGD":
        logger.info('Using PGDOptimizer')
        opt_config = trojai.modelgen.config.DefaultOptimizerConfig(training_params, reporting_params)
        optimizer_cfg = trojai.modelgen.adversarial_pgd_optimizer.PGDOptimizer(opt_config)
        training_params.adv_training_eps = config.adversarial_eps
        training_params.adv_training_ratio = config.adversarial_training_ratio
        training_params.adv_training_iterations = config.adversarial_training_iteration_count
    else:
        raise RuntimeError("Invalid config.ADVERSARIAL_TRAINING_METHOD = {}".format(config.adversarial_training_method))

    experiment_cfg = dict()
    experiment_cfg['model_save_dir'] = model_save_dir
    experiment_cfg['stats_save_dir'] = stats_save_dir
    experiment_cfg['experiment_path'] = config.data_filepath
    experiment_cfg['name'] = config.model_architecture

    model_cfg = dict()
    model_cfg['number_classes'] = config.number_classes

    cfg = trojai.modelgen.config.ModelGeneratorConfig(arch, data_obj, model_save_dir, stats_save_dir, 1,
                                                      optimizer=optimizer_cfg,
                                                      experiment_cfg=experiment_cfg,
                                                      arch_factory_kwargs=model_cfg,
                                                      parallel=True,
                                                      amp=True)

    model_generator = trojai.modelgen.model_generator.ModelGenerator(cfg)
    model_generator.run()
    model_filepath = os.path.join(model_save_dir, 'DataParallel_{}.pt.1'.format(config.model_architecture))
    return model_filepath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single CNN model based on lmdb dataset')
    parser.add_argument('--filepath', type=str, required=True, help='Filepath to the folder/directory storing the whole dataset. Within that folder must be: ground_truth.csv, config.json, train_data.lmdb, test_data.lmdb, train.csv, test-clean.csv, test-poisoned.csv')
    args = parser.parse_args()

    # load data configuration
    filepath = args.filepath
    config = round_config.RoundConfig.load_json(os.path.join(filepath, round_config.RoundConfig.CONFIG_FILENAME))

    # setup logger
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        filename=os.path.join(config.data_filepath, 'log.txt'))

    logger.info('Data Configuration Loaded')
    logger.info(config)

    model_filepath = train_model(config)