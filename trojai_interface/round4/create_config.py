# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import logging
import numpy as np
import shutil

import round_config

logger = logging.getLogger(__name__)


def select_n_foreground_files(file_directory, number_to_select, random_state_object, file_format):
    """
    Selects the requested number of foreground images from the specified folder at random, in a numerically reproducible manner using the provided random state object
    :param file_directory: Folder/directory to select foreground files within.
    :param number_to_select: Number of foreground files to select from the specified folder
    :param random_state_object: The random state object used to control any randomness (allows rebuilding the exact same dataset at a later date.
    :param file_format: The format of the files being selected, to allow filtering of the files found in directory.
    :return: List of filenames that have been selected.
    """

    # ensure the file format does not start with a period
    if file_format.startswith('.'):
        file_format = file_format[1:]

    # Finds the set of files in the directory which match the provided file extension
    # This function relies on the file naming convention of the TrojAI foregrounds directory.
    # The files must be named: {:d}-{:d}.<ext>
    # The files are grouped based on the primary fake traffic sign shape. The first digit (before the '-') indicates the primary traffic sign shape.
    # The second digit indicates which central icon has been added to the sign.
    # This loop finds all of the unique primary fake traffic sign types (ignoring all the symbols in the middle).
    fns = [fn for fn in os.listdir(file_directory) if fn.endswith('-0.{}'.format(file_format))]
    first_numbers = [int(fn.split('-')[0]) for fn in fns]

    filenames = list()
    used_second_numbers = list()
    # select a set of N first numbers (so that the traffic signs used for the image classifier foregrounds do not share a high similarity (share everything but the central icon).
    first_numbers = random_state_object.choice(first_numbers, size=number_to_select, replace=False)
    # loop over the selected first numbers and pick a random second number for each first.
    # the second numbers are also unique within each classification AI dataset to avoid confusion
    for first_number in first_numbers:
        # find the set of available second numbers within the foregrounds folder.
        fns = [fn for fn in os.listdir(file_directory) if fn.startswith('{}-'.format(first_number))]
        fn = None
        # convert the filenames into a list of integers specifying the second number
        second_numbers = list()
        for fn in fns:
            second_numbers.append(int(fn.replace('.{}'.format(file_format), '').split('-')[1]))
        # shuffle the list order
        random_state_object.shuffle(second_numbers)
        # look for an unused second number in the list of available files
        for i in range(len(second_numbers)):
            second_number = second_numbers[i]
            if second_number not in used_second_numbers:
                # ensure this second number cannot be re-used
                used_second_numbers.append(second_number)
                fn = '{}-{}.{}'.format(first_number, second_number, file_format)
                break
        # raise an error if we cannot find an appropriate set of foreground files for training this AI model
        if fn is None:
            raise RuntimeError('Unable to select independent foregrounds.')
        filenames.append(fn)
    return filenames


def copy_to(filename_list, source_directory, target_directory):
    """
    Utility function to copy a list of files from a source to a target directory
    :param filename_list: The list of files to copy.
    :param source_directory: The source directory where those files can be found.
    :param target_directory: The target directory where the files should be copied to.
    :return:
    """
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    os.makedirs(target_directory)

    for fn in filename_list:
        shutil.copy(os.path.join(source_directory, fn), os.path.join(target_directory, fn))


def create(poison_flag, foreground_images_filepath, background_images_filepath, output_filepath):
    """
    Creates a round_config object
    :param poison_flag: Whether this is a poisoned model
    :param foreground_images_filepath: The filepath to where the foreground image files exists.
    :param background_images_filepath: The filepath to where the background image files exist.
    :param output_filepath: The filepath where the model should be constructed.
    :return:
    """

    # instantiate a round config
    config = round_config.RoundConfig(poison_flag, foreground_images_filepath, background_images_filepath, output_filepath)
    # save the config file to disk
    config.save_json(os.path.join(config.data_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    # create the random state object using the specified master seed
    master_rso = np.random.RandomState(config.master_seed)

    # select a set of foreground files for this AI to train to recognize
    foreground_filenames = select_n_foreground_files(config.available_foregrounds_filepath, config.number_classes, master_rso, config.foreground_image_format)
    foreground_filenames.sort()
    # copy those files to the model folder to keep a copy of the foregrounds with the trained model
    copy_to(foreground_filenames, config.available_foregrounds_filepath, config.foregrounds_filepath)

    # write the ground truth file
    with open(os.path.join(config.data_filepath, config.output_ground_truth_filename), 'w') as fh:
        fh.write('{:d}'.format(int(config.poisoned)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a Single Dataset Configuration')
    parser.add_argument('--poison', action='store_true')
    parser.add_argument('--foreground_images_filepath', type=str, required=True)
    parser.add_argument('--background_images_filepath', type=str, required=True)
    parser.add_argument('--output_filepath', type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.output_filepath):
        shutil.rmtree(args.output_filepath)
    os.makedirs(args.output_filepath)

    create(args.poison, args.foreground_images_filepath, args.background_images_filepath, args.output_filepath)
