# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import numpy as np
from numpy.random import RandomState

import dataset
import round_config


def poisoned(model_filepath, foregrounds_filepath, backgrounds_filepath, nb_tgt_images=None, output_folder_name='poisoned_example_data'):
    """
    Build poisoned images drawn from the same distribution used to train the AI model specified by the filepath.
    :param model_filepath: The filepath to the model example images are being built for.
    :param foregrounds_filepath: The filepath to the full set of foregrounds available to this round.
    :param backgrounds_filepath: The filepath to the full set of backgrounds available to this round.
    :param nb_tgt_images: The number of images to construct.
    :param output_folder_name: What to name the output folder.
    :return:
    """

    config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    config.available_foregrounds_filepath = foregrounds_filepath
    config.available_backgrounds_filepath = backgrounds_filepath

    config.data_filepath = model_filepath
    config.foregrounds_filepath = os.path.join(config.data_filepath, 'foregrounds')
    config.backgrounds_filepath = os.path.join(config.available_backgrounds_filepath, config.background_image_dataset)

    # master_RSO = RandomState(config.master_seed)
    master_RSO = RandomState(np.random.randint(2 ** 31 - 1))
    train_rso = RandomState(master_RSO.randint(2 ** 31 - 1))

    if not config.poisoned:
        raise RuntimeError('Cannot create poisoned example images for a clean model')
    output_filepath = os.path.join(config.data_filepath, output_folder_name)

    # turn on triggers for poisoned data
    if config.poisoned:
        for trigger in config.triggers:
            trigger.fraction = 1.0

    if config.poisoned:
        for trigger in config.triggers:
            if trigger.type == 'polygon':
                # update the filepath to the triggers
                parent, filename = os.path.split(trigger.polygon_filepath)
                trigger.polygon_filepath = os.path.join(config.data_filepath, filename)

    if nb_tgt_images is not None:
        if nb_tgt_images > config.number_example_images:
            config.number_example_images = nb_tgt_images

    shm_dataset = dataset.TrafficDataset(config, train_rso, config.number_training_samples, class_balanced=True)

    shm_dataset.build_examples(output_filepath, config.number_example_images)


def clean(model_filepath, foregrounds_filepath, backgrounds_filepath, nb_tgt_images=None, output_folder_name='clean_example_data'):
    """
    Build clean images drawn from the same distribution used to train the AI model specified by the filepath.
    :param model_filepath: The filepath to the model example images are being built for.
    :param foregrounds_filepath: The filepath to the full set of foregrounds available to this round.
    :param backgrounds_filepath: The filepath to the full set of backgrounds available to this round.
    :param nb_tgt_images: The number of images to construct.
    :param output_folder_name: What to name the output folder.
    :return:
    """

    config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    config.available_foregrounds_filepath = foregrounds_filepath
    config.available_backgrounds_filepath = backgrounds_filepath

    config.data_filepath = model_filepath
    config.foregrounds_filepath = os.path.join(config.data_filepath, 'foregrounds')
    config.backgrounds_filepath = os.path.join(config.available_backgrounds_filepath, config.background_image_dataset)

    # master_RSO = RandomState(config.master_seed)
    master_RSO = RandomState(np.random.randint(2 ** 31 - 1))
    train_rso = RandomState(master_RSO.randint(2 ** 31 - 1))

    # turn off triggers for clean data
    if config.poisoned:
        config.poisoned = False
        for trigger in config.triggers:
            trigger.fraction = 0.0

    output_filepath = os.path.join(config.data_filepath, output_folder_name)

    if nb_tgt_images is not None:
        if nb_tgt_images > config.number_example_images:
            config.number_example_images = nb_tgt_images

    shm_dataset = dataset.TrafficDataset(config, train_rso, config.number_training_samples, class_balanced=True)

    shm_dataset.build_examples(output_filepath, config.number_example_images)


def rebuild_dataset(model_filepath, foregrounds_filepath, backgrounds_filepath, training_data_flag: bool = False) -> dataset.TrafficDataset:
    """
    Rebuild the dataset used to construct the model specified by the filepath.
    :param model_filepath: The filepath to the model example images are being built for.
    :param foregrounds_filepath: The filepath to the full set of foregrounds available to this round.
    :param backgrounds_filepath: The filepath to the full set of backgrounds available to this round.
    :param training_data_flag: Whether to rebuild the training or testing data for this model.
    :return:
    """

    config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    config.available_foregrounds_filepath = foregrounds_filepath
    config.available_backgrounds_filepath = backgrounds_filepath

    config.data_filepath = model_filepath
    config.foregrounds_filepath = os.path.join(config.data_filepath, 'foregrounds')
    config.backgrounds_filepath = os.path.join(config.available_backgrounds_filepath, config.background_image_dataset)

    # reset the RSO to the seed value so the config setup does not impact downstream RNG
    master_RSO = RandomState(config.master_seed)
    train_rso = RandomState(master_RSO.randint(2 ** 31 - 1))
    test_rso = RandomState(master_RSO.randint(2 ** 31 - 1))

    if config.poisoned:
        for trigger in config.triggers:
            if trigger.type == 'polygon':
                # update the filepath to the triggers
                parent, filename = os.path.split(trigger.polygon_filepath)
                trigger.polygon_filepath = os.path.join(config.data_filepath, filename)

    if training_data_flag:
        shm_dataset = dataset.TrafficDataset(config, train_rso, config.number_training_samples, class_balanced=True)
    else:
        shm_dataset = dataset.TrafficDataset(config, test_rso, config.number_test_samples, class_balanced=True)

    shm_dataset.build_dataset()
    return shm_dataset
