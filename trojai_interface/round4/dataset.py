# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
# enforce single threading for libraries to allow for multithreading across image instances.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import glob
import logging
import numpy as np
import copy
from numpy.random import RandomState
import skimage.io
import multiprocessing
import pandas as pd
import shutil

import cv2
# enforce single threading for libraries to allow for multithreading across image instances.
cv2.setNumThreads(0)  # prevent opencv from using multi-threading

import trojai.datagen.image_affine_xforms
import trojai.datagen.static_color_xforms
import trojai.datagen.image_size_xforms
import trojai.datagen.xform_merge_pipeline
import trojai.datagen.image_entity
import trojai.datagen.constants
import trojai.datagen.xform_merge_pipeline
import trojai.datagen.utils
import trojai.datagen.insert_merges
import trojai.datagen.image_affine_xforms
import trojai.datagen.instagram_xforms
import trojai.datagen.common_label_behaviors
import trojai.datagen.image_affine_xforms
import trojai.datagen.noise_xforms
import trojai.datagen.albumentations_xforms
import trojai.datagen.config
import trojai.datagen.blend_merges
import trojai.datagen.lighting_utils
import trojai.datagen.file_trigger

from trojai.modelgen.datasets import DatasetInterface

from . import trigger_config
from . import round_config

logger = logging.getLogger(__name__)


class InMemoryDataset(DatasetInterface):
    """
    Defines a dataset resident in CPU memory with columns "key_str", "train_label", and optionally
    "true_label". The data is loaded from a specified lmdb file containing a serialized key value store of the actual data.
    "train_label" refers to the label with which the data should be trained.  "true_label" refers to the actual
    label of the data point, and can differ from train_label if the dataset is poisoned.  This dataset provides views into
    the in CPU memory TrafficDataset. The views are based on what list of dict keys are passed to the constructor.
    """

    def __init__(self, data: dict, keys_list: list, data_transform=lambda x: x, label_transform=lambda l: l):
        """
        Initializes a InMemoryDataset object.
        :param data: dict() containing the numpy ndarray image objects
        :param keys_list: the list of keys into the shared memory pool
        :param data_transform: a callable function which is applied to every data point before it is fed into the
            model. By default, this is an identity operation
        :param label_transform: a callable function which is applied to every label before it is fed into the model.
            By default, this is an identity operation.
        """
        super().__init__(None)

        self.data = data
        self.keys_list = keys_list

        # convert keys_list to pandas dataframe with the following columns: key_str, triggered, train_label, true_label
        # this data_df field is required by the trojai api
        self.data_df = pd.DataFrame(self.keys_list)

        self.data_transform = data_transform
        self.label_transform = label_transform

        self.data_description = 'Shared Memory dataset of {} records'.format(len(self.keys_list))
        logger.debug('SharedMemoryDataset has {} keys'.format(len(self.keys_list)))

    def __getitem__(self, item):
        key_data = self.keys_list[item]
        key = key_data['key']

        data = self.data[key]
        train_label = key_data['train_label']
        # true_label = key_data.true_label
        label = train_label

        data = self.data_transform(data)
        label = self.label_transform(label)

        return data, label

    def __len__(self):
        return len(self.keys_list)

    def get_data_description(self):
        return self.data_description

    def set_data_description(self):
        pass


def build_quadrant_mask(fg_mask: np.ndarray, quadrant: int):
    """
    Create a mask limiting the placement of a polygon trojan to a specific quadrant of the foreground.
    :param fg_mask: the mask (np.bool ndarray) indicating where the foreground exists within the image.
    :param quadrant: the quadrant to limit the mask to. This is an integer in {1, 2, 3, 4} with Q1 being upper right, Q2 upper left, Q3 lower left, and Q4 lower right. The quadrant specification is with respect to the center of the fg_mask, not the containing image.
    :return: fg_mask modified to only the values in the selected quadrant are true.
    """
    y_idx, x_idx = np.nonzero(fg_mask)
    t_mask = np.zeros(fg_mask.shape).astype(np.bool)
    f_x_st = np.min(x_idx)
    f_y_st = np.min(y_idx)
    f_x_end = np.max(x_idx)
    f_y_end = np.max(y_idx)

    sign_h = f_y_end - f_y_st
    sign_w = f_x_end - f_x_st

    fg_x_center = f_x_st + int(sign_w / 2)
    fg_y_center = f_y_st + int(sign_h / 2)

    if quadrant == 1:
        t_mask[0:fg_y_center, fg_x_center:] = 1
    elif quadrant == 2:
        t_mask[0:fg_y_center, 0:fg_x_center] = 1
    elif quadrant == 3:
        t_mask[fg_y_center:, 0:fg_x_center] = 1
    else:
        t_mask[fg_y_center:, fg_x_center:] = 1
    return t_mask


def build_image(config: round_config.RoundConfig, rso: np.random.RandomState, fg_image_fp: str, bg_image_fp: str, obj_class_label: int, allow_spurious_triggers: bool = True) -> (np.ndarray, int, int, bool):
    """
    Worker function to build all possible configurations from the round config. This function takes in the config and random state object and produces an image instance that is valid given the config.
    :param config: the round config object defining the rounds parameters
    :param rso: the random state object to draw random numbers from
    :param fg_image_fp: the filepath to the foreground image
    :param bg_image_fp: the filepath to the background image
    :param obj_class_label: the class label associated with the foreground filepath
    :return: Tuple (np.ndarray, int, int, bool)
    Image instance created from the specified foreground and background that is consistent with the round config.
    Training object class label (may be different from the foreground class if a trigger has been inserted.
    Foreground object class label.
    Flag indicating whether a class changing trigger has been inserted, i.e. whether the image has been poisoned.
    """

    # specify the background xforms
    bg_xforms = TrafficDataset.build_background_xforms(config)
    # specify the foreground xforms
    fg_xforms = TrafficDataset.build_foreground_xforms(config)
    # specify the foreground/background merge object
    merge_obj = trojai.datagen.blend_merges.BrightnessAdjustGrainMergePaste(lighting_adjuster=trojai.datagen.lighting_utils.adjust_brightness_mmprms)
    # specify the trigger/foreground merge object
    trigger_merge_obj = trojai.datagen.insert_merges.InsertRandomWithMask()
    # specify the xforms for the final image
    combined_xforms = TrafficDataset.build_combined_xforms(config)

    # load foreground image
    sign_img = skimage.io.imread(fg_image_fp)
    sign_mask = (sign_img[:, :, 3] > 0).astype(bool)
    fg_entity = trojai.datagen.image_entity.GenericImageEntity(sign_img, sign_mask)
    # apply any foreground xforms
    fg_entity = trojai.datagen.utils.process_xform_list(fg_entity, fg_xforms, rso)

    # load background image
    bg_entity = trojai.datagen.image_entity.GenericImageEntity(skimage.io.imread(bg_image_fp))

    # define the training label
    train_obj_class_label = obj_class_label

    selected_trigger = None
    non_spurious_trigger_flag = False
    if config.poisoned:
        # loop over the possible triggers to insert
        trigger_list = copy.deepcopy(config.triggers)
        rso.shuffle(trigger_list) # shuffle trigger order since we short circuit on the first applied trigger if multiple can be applied. This prevents trigger collision
        for trigger in trigger_list:
            # short circuit to prevent trigger collision when multiple triggers apply
            if selected_trigger is not None:
                break

            # if the current class is one of those being triggered
            correct_class_flag = obj_class_label == trigger.source_class
            trigger_probability_flag = rso.rand() <= trigger.fraction

            if correct_class_flag:
                # this image is from the source class
                if trigger_probability_flag:
                    non_spurious_trigger_flag = True
                    # apply the trigger label transform if and only if this is a non-spurious trigger
                    trigger_label_xform = trojai.datagen.common_label_behaviors.StaticTarget(trigger.target_class)
                    train_obj_class_label = trigger_label_xform.do(train_obj_class_label)
                    selected_trigger = copy.deepcopy(trigger)

                    if selected_trigger.type == 'polygon':
                        if selected_trigger.condition == 'spatial':
                            # only allow the trigger to exist within the spatial region defined by the quadrant
                            # modify the fg_entity mask to limit where the trigger can be placed
                            fg_mask = fg_entity.get_mask().astype(np.bool)
                            t_mask = build_quadrant_mask(fg_mask, selected_trigger.spatial_quadrant)
                            new_fg_mask = np.logical_and(fg_mask, t_mask)
                            fg_entity = trojai.datagen.image_entity.GenericImageEntity(fg_entity.get_data(), new_fg_mask)

        # only try to insert a spurious trigger if the image is not already being poisoned
        if allow_spurious_triggers and selected_trigger is None:
            for trigger in trigger_list:
                correct_class_flag = obj_class_label == trigger.source_class
                trigger_probability_flag = rso.rand() <= trigger.fraction
                # determine whether to insert a spurious trigger
                if trigger_probability_flag and trigger.condition is not None:
                    if trigger.type == 'polygon':
                        if trigger.condition == 'class':
                            # trigger applied to wrong class
                            selected_trigger = copy.deepcopy(trigger)

                        # modify trigger config to be spurious
                        if correct_class_flag and trigger.condition == 'spatial':
                            selected_trigger = copy.deepcopy(trigger)

                            # modify the fg_entity mask to limit where the trigger can be placed
                            fg_mask = fg_entity.get_mask().astype(np.bool)
                            t_mask = build_quadrant_mask(fg_mask, selected_trigger.spatial_quadrant)
                            # invert this mask, since we can only insert a spurious trigger in a location not in that conditional quadrant
                            t_mask = np.invert(t_mask)
                            new_fg_mask = np.logical_and(fg_mask, t_mask)
                            fg_entity = trojai.datagen.image_entity.GenericImageEntity(fg_entity.get_data(), new_fg_mask)

                        if correct_class_flag and trigger.condition == 'spectral':
                            selected_trigger = copy.deepcopy(trigger)

                            # get the set of allowed trigger colors
                            valid_color_levels = list(range(len(trigger_config.TriggerConfig.TRIGGER_COLOR_LEVELS)))
                            # remove the actual trigger color, since this is a spurious trigger
                            valid_color_levels.remove(selected_trigger.color_level)
                            # select a color for the spurious trigger
                            color_level = int(rso.choice(valid_color_levels, size=1, replace=False))
                            selected_trigger.color_level = color_level
                            selected_trigger.color = trigger_config.TriggerConfig.TRIGGER_COLOR_LEVELS[color_level]

                    if trigger.type == 'instagram':
                        if trigger.condition == 'class':
                            # trigger applied to wrong class
                            selected_trigger = copy.deepcopy(trigger)

                        if correct_class_flag and trigger.condition == 'spatial':
                            raise NotImplementedError('Spatial condition for instagram trigger not implemented.')

                        if correct_class_flag and trigger.condition == 'spectral':
                            selected_trigger = copy.deepcopy(trigger)
                            # spectral condition for instagram filters should change the filter being applied
                            valid_filter_levels = list(range(len(trigger_config.TriggerConfig.INSTAGRAM_TRIGGER_TYPE_LEVELS)))
                            # remove the actual trigger fitler, since this is a spurious trigger
                            valid_filter_levels.remove(selected_trigger.instagram_filter_type_level)
                            # select a color for the spurious trigger
                            filter_level = int(rso.choice(valid_filter_levels, size=1, replace=False))
                            selected_trigger.instagram_filter_type_level = filter_level
                            selected_trigger.instagram_filter_type = trigger_config.TriggerConfig.INSTAGRAM_TRIGGER_TYPE_LEVELS[filter_level]

    if selected_trigger is not None and non_spurious_trigger_flag is False:
        # This is a spurious trigger, it should not affect the training label
        # set the source class to this class
        selected_trigger.source_class = train_obj_class_label
        # set the target class to this class, so the trigger does nothing
        selected_trigger.target_class = train_obj_class_label

    if selected_trigger is not None and selected_trigger.type == 'polygon':
        # use the size of the foreground image to determine how large to make the trigger
        foreground_mask = np.sum(fg_entity.get_data(), axis=-1).astype(np.bool)
        # y_idx, x_idx = np.nonzero(foreground_mask)
        # foreground_area = (np.max(x_idx) - np.min(x_idx)) * (np.max(y_idx) - np.min(y_idx))
        foreground_area = np.count_nonzero(foreground_mask)

        # determine valid trigger size range based on the size of the foreground object
        trigger_area_min = foreground_area * selected_trigger.size_percentage_of_foreground_min
        trigger_area_max = foreground_area * selected_trigger.size_percentage_of_foreground_max
        trigger_pixel_size_min = int(np.sqrt(trigger_area_min))
        trigger_pixel_size_max = int(np.sqrt(trigger_area_max))

        tgt_trigger_size = rso.randint(trigger_pixel_size_min, trigger_pixel_size_max + 1)
        tgt_trigger_size = (tgt_trigger_size, tgt_trigger_size)
        trigger_entity = trojai.datagen.file_trigger.FlatIconDotComPng(selected_trigger.polygon_filepath, mode='graffiti', trigger_color=selected_trigger.color, size=tgt_trigger_size)

        # merge the trigger into the foreground
        trigger_xforms = [trojai.datagen.image_affine_xforms.RandomRotateXForm(angle_choices=list(range(0, 360, 5)))]
        # this foreground xforms list is empty since we already applied the foreground xforms earlier
        foreground_trigger_merge_xforms = []
        pipeline_obj = trojai.datagen.xform_merge_pipeline.XFormMerge([[foreground_trigger_merge_xforms, trigger_xforms]],
                                                                      [trigger_merge_obj],
                                                                      None)
        fg_entity = pipeline_obj.process([fg_entity, trigger_entity], rso)

    # merge foreground into background
    foreground_trigger_merge_xforms = []
    pipeline_obj = trojai.datagen.xform_merge_pipeline.XFormMerge([[bg_xforms, foreground_trigger_merge_xforms]],
                                                                  [merge_obj],
                                                                  combined_xforms)
    processed_img = pipeline_obj.process([bg_entity, fg_entity], rso)

    if selected_trigger is not None and selected_trigger.type == 'instagram':
        # apply the instagram filter over the complete image
        if selected_trigger.instagram_filter_type == 'GothamFilterXForm':
            trigger_entity = trojai.datagen.instagram_xforms.GothamFilterXForm(channel_order='RGB')
        elif selected_trigger.instagram_filter_type == 'NashvilleFilterXForm':
            trigger_entity = trojai.datagen.instagram_xforms.NashvilleFilterXForm(channel_order='RGB')
        elif selected_trigger.instagram_filter_type == 'KelvinFilterXForm':
            trigger_entity = trojai.datagen.instagram_xforms.KelvinFilterXForm(channel_order='RGB')
        elif selected_trigger.instagram_filter_type == 'LomoFilterXForm':
            trigger_entity = trojai.datagen.instagram_xforms.LomoFilterXForm(channel_order='RGB')
        elif selected_trigger.instagram_filter_type == 'ToasterXForm':
            trigger_entity = trojai.datagen.instagram_xforms.ToasterXForm(channel_order='RGB')
        else:
            raise RuntimeError('Invalid instagram trigger type: {}'.format(selected_trigger.instagram_filter_type))

        processed_img = trojai.datagen.utils.process_xform_list(processed_img, [trigger_entity], rso)

    # get a numpy array of the image
    np_img = processed_img.get_data().astype(np.float32)#config.img_type)
    selected_trigger_nb = None
    if selected_trigger is not None:
        selected_trigger_nb = selected_trigger.trigger_number

    return np_img, train_obj_class_label, obj_class_label, non_spurious_trigger_flag, selected_trigger_nb


class TrafficDataset:
    """
    Traffic sign dataset which combines a folder of foreground images with a folder of backgrounds according to a config. This class relies on the Copy of Write functionality of fork on Linux. The parent process will have a copy of the image data in a dict(), and the forked child processes will have access to the data dict() without copying it since its only read, never written. Using this code on non Linux systems is highly discouraged due to each process requiring a complete copy of the data (which is multi-GB).
    """
    def __init__(self, config: round_config.RoundConfig, random_state_obj: np.random.RandomState, num_samples_to_generate: int, class_balanced: bool = True, worker_process_count: int = None, fraction = 1., ret_inject=False):
        """
        Instantiates a TrafficDataset from a specific config file and random state object.
        :param config: the round config controlling the image generation
        :param random_state_obj: the random state object providing all random decisions
        :param num_samples_to_generate: the number of image samples to create.
        :param class_balanced: whether an even number of instances should be created for each class, or whether the class id can be drawn from the random_state_object.
        :param worker_process_count: the number of python multiprocessing worker processes to use when building the image data.
        """
        self.config = copy.deepcopy(config)

        self.rso = random_state_obj
        self.num_samples_to_generate = num_samples_to_generate * fraction
        self.class_balanced = class_balanced
        self.thread_count = worker_process_count

        # dict to store numpy data of the images in the dataset
        self.data = dict()

        if self.thread_count is None:
            # default to all the cores
            num_cpu_cores = multiprocessing.cpu_count()
            try:
                # if slurm is found use the cpu count it specifies
                num_cpu_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
            except:
                pass  # do nothing
            self.thread_count = num_cpu_cores

        self.all_keys_list = list()
        self.clean_keys_list = list()
        self.poisoned_keys_list = list()

    def get_dataset(self, data_transform=lambda x: x, label_transform=lambda l: l) -> InMemoryDataset:
        """
        Get a view of this TrafficDataset containing all data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around this TrafficDataset.
        """
        return InMemoryDataset(self.data, self.all_keys_list, data_transform, label_transform)

    def get_clean_dataset(self, data_transform=lambda x: x, label_transform=lambda l: l) -> InMemoryDataset:
        """
        Get a view of this TrafficDataset containing just the clean data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around the clean data in this TrafficDataset.
        """
        return InMemoryDataset(self.data, self.clean_keys_list, data_transform, label_transform)

    def get_poisoned_dataset(self, data_transform=lambda x: x, label_transform=lambda l: l) -> InMemoryDataset:
        """
        Get a view of this TrafficDataset containing just the poisoned data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around the poisoned data in this TrafficDataset.
        """
        return InMemoryDataset(self.data, self.poisoned_keys_list, data_transform, label_transform)

    @staticmethod
    def build_background_xforms(config: round_config.RoundConfig):
        """
        Defines the chain of transformations which need to be applied to each background image.
        :return: list of trojai.datagen transformations
        """
        img_size = (config.img_size_pixels, config.img_size_pixels)

        bg_xforms = list()
        bg_xforms.append(trojai.datagen.static_color_xforms.RGBtoRGBA())
        bg_xforms.append(trojai.datagen.image_size_xforms.RandomSubCrop(new_size=img_size))
        return bg_xforms

    @staticmethod
    def build_foreground_xforms(config: round_config.RoundConfig):
        """
        Defines the chain of transformations which need to be applied to each foreground image.
        :return: list of trojai.datagen transformations
        """
        img_size = (config.img_size_pixels, config.img_size_pixels)

        min_foreground_size = (config.foreground_size_pixels_min, config.foreground_size_pixels_min)
        max_foreground_size = (config.foreground_size_pixels_max, config.foreground_size_pixels_max)

        sign_xforms = list()
        sign_xforms.append(trojai.datagen.static_color_xforms.RGBtoRGBA())
        sign_xforms.append(trojai.datagen.image_affine_xforms.RandomPerspectiveXForm(None))
        sign_xforms.append(trojai.datagen.image_size_xforms.RandomResize(new_size_minimum=min_foreground_size, new_size_maximum=max_foreground_size))
        sign_xforms.append(trojai.datagen.image_size_xforms.RandomPadToSize(img_size))

        return sign_xforms

    @staticmethod
    def build_combined_xforms(config: round_config.RoundConfig):
        """
        Defines the chain of transformations which need to be applied to each image after the foreground has been inserted into the background.
        :return: list of trojai.datagen transformations
        """
        # create the merge transformations for blending the foregrounds and backgrounds together
        combined_xforms = list()
        combined_xforms.append(trojai.datagen.noise_xforms.RandomGaussianBlurXForm(ksize_min=config.gaussian_blur_ksize_min, ksize_max=config.gaussian_blur_ksize_max))
        if config.fog_probability > 0:
            combined_xforms.append(trojai.datagen.albumentations_xforms.AddFogXForm(always_apply=False, probability=config.fog_probability))
        if config.rain_probability > 0:
            combined_xforms.append(trojai.datagen.albumentations_xforms.AddRainXForm(always_apply=False, probability=config.rain_probability))
        combined_xforms.append(trojai.datagen.static_color_xforms.RGBAtoRGB())
        return combined_xforms

    def build_dataset(self):
        """
        Instantiates this TrafficDataset object into CPU memory. This is function can be called at a different time than the dataset object is created to control when memory is used. This function will consume a lot of CPU memory. Required memory is at least (config.img_size x config.img_size x 3*config.img_type.nbytes x num_samples_to_generate / 1024 / 1024 / 1024) GB.
        """
        if self.config.foreground_image_format.startswith('.'):
            self.config.foreground_image_format = self.config.foreground_image_format[1:]
        if self.config.background_image_format.startswith('.'):
            self.config.background_image_format = self.config.background_image_format[1:]

        # get listing of all bg files
        bg_image_fps = glob.glob(os.path.join(self.config.backgrounds_filepath, '**', '*.' + self.config.background_image_format), recursive=True)
        # enforce deterministic background order
        bg_image_fps.sort()
        self.rso.shuffle(bg_image_fps)

        # get listing of all foreground images
        fg_image_fps = glob.glob(os.path.join(self.config.foregrounds_filepath, '**', '*.' + self.config.foreground_image_format), recursive=True)
        # enforce deterministic foreground order, which equates to class label mapping
        fg_image_fps.sort()

        num_classes = len(fg_image_fps)
        if self.class_balanced:
            # generate a probability sampler for each class
            class_sampling_vector = np.ones(num_classes) / num_classes
        else:
            # double check the balancing vector
            sum_prob = np.sum(self.class_balanced)
            if not np.isclose(sum_prob, 1):
                raise ValueError("class_balanced probabilities must sum to 1!")
            class_sampling_vector = self.class_balanced

        # generate a vector of the object classes which exactly preserves the desired class sampling vector
        num_samples_per_class = float(self.num_samples_to_generate) * class_sampling_vector
        num_samples_per_class = np.ceil(num_samples_per_class).astype(int)
        num_samples_to_generate = int(np.sum(num_samples_per_class))
        logger.info("Generating " + str(num_samples_to_generate) + " samples")

        obj_class_list = []
        for ii, num_samples_in_class in enumerate(num_samples_per_class):
            obj_class_list.extend([ii] * num_samples_in_class)

        # shuffle it to introduce some level of randomness
        self.rso.shuffle(obj_class_list)

        logger.info('Using {} CPU cores to generate data'.format(self.thread_count))

        worker_input_list = list()
        for i in range(0, num_samples_to_generate):
            obj_class_label = obj_class_list[i]
            sign_image_f = fg_image_fps[obj_class_label]

            bg_image_idx = self.rso.randint(low=0, high=len(bg_image_fps))
            bg_image_f = bg_image_fps[bg_image_idx]

            rso = RandomState(self.rso.randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))
            worker_input_list.append((self.config, rso, sign_image_f, bg_image_f, obj_class_label))

        with multiprocessing.Pool(processes=self.thread_count) as pool:
            # perform the work in parallel
            img_nb = 0
            results = pool.starmap(build_image, worker_input_list)

            for result in results:
                np_img, train_obj_class_label, obj_class_label, poisoned_flag, trigger_nb = result
                key_str = 'img_{:012d}'.format(img_nb)
                self.data[key_str] = np_img
                img_nb += 1

                # add information to dataframe
                self.all_keys_list.append({'key': key_str,
                                           'triggered': poisoned_flag,
                                           'train_label': train_obj_class_label,
                                           'true_label': obj_class_label})

                if poisoned_flag:
                    self.poisoned_keys_list.append({'key': key_str,
                                                    'triggered': poisoned_flag,
                                                    'train_label': train_obj_class_label,
                                                    'true_label': obj_class_label})
                else:
                    self.clean_keys_list.append({'key': key_str,
                                                 'triggered': poisoned_flag,
                                                 'train_label': train_obj_class_label,
                                                 'true_label': obj_class_label})

    def build_examples(self, output_filepath: str, num_samples_per_class: int):
        """
        Builds example images drawn from the dataset and saves them as png images to the specified output_filepath.
        :param output_filepath: Filepath to the folder the generated images will be save.
        :param num_samples_per_class: The number of sample images to create per class.
        """

        if self.config.foreground_image_format.startswith('.'):
            self.config.foreground_image_format = self.config.foreground_image_format[1:]
        if self.config.background_image_format.startswith('.'):
            self.config.background_image_format = self.config.background_image_format[1:]

        # get listing of all bg files
        bg_image_fps = glob.glob(os.path.join(self.config.backgrounds_filepath, '**', '*.' + self.config.background_image_format), recursive=True)
        # enforce deterministic background order
        bg_image_fps.sort()
        self.rso.shuffle(bg_image_fps)

        # get listing of all foreground images
        fg_image_fps = glob.glob(os.path.join(self.config.foregrounds_filepath, '**', '*.' + self.config.foreground_image_format), recursive=True)
        # enforce deterministic foreground order, which equates to class label mapping
        fg_image_fps.sort()

        num_classes = len(fg_image_fps)

        if os.path.exists(output_filepath):
            shutil.rmtree(output_filepath)
        os.makedirs(output_filepath)

        if not self.config.poisoned:
            class_list = list(range(num_classes))
        else:
            class_list = list()
            for trigger in self.config.triggers:
                class_list.append(trigger.source_class)

        for obj_class_label in class_list:
            for nb in range(num_samples_per_class):
                sign_image_f = fg_image_fps[obj_class_label]

                bg_image_idx = self.rso.randint(low=0, high=len(bg_image_fps))
                bg_image_f = bg_image_fps[bg_image_idx]

                rso = RandomState(self.rso.randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))
                np_img, train_obj_class_label, _, poisoned_flag, trigger_nb = build_image(self.config, rso, sign_image_f, bg_image_f, obj_class_label, allow_spurious_triggers=False)

                if poisoned_flag:
                    key_str = 'class_{}_trigger_{}_example_{}'.format(obj_class_label, trigger_nb, nb)
                    self.poisoned_keys_list.append({'key': key_str,
                                                    'triggered': poisoned_flag,
                                                    'train_label': train_obj_class_label,
                                                    'true_label': obj_class_label})
                else:
                    key_str = 'class_{}_example_{}'.format(obj_class_label, nb)
                    self.clean_keys_list.append({'key': key_str,
                                                 'triggered': poisoned_flag,
                                                 'train_label': train_obj_class_label,
                                                 'true_label': obj_class_label})

                self.data[key_str] = np_img

                # add information to dataframe
                self.all_keys_list.append({'key': key_str,
                                           'triggered': poisoned_flag,
                                           'train_label': train_obj_class_label,
                                           'true_label': obj_class_label})

                file_name = '{}.png'.format(key_str)
                fname_out = os.path.join(output_filepath, file_name)
                skimage.io.imsave(fname_out, np_img)

