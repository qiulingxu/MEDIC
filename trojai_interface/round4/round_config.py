# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import logging
import numpy as np
import json
import types
import jsonpickle

from . import  trigger_config
from . import  model_factories

logger = logging.getLogger(__name__)

class a:
    def __init__(self):
        pass

class RoundConfig:
    CONFIG_FILENAME = 'config.json'

    CLASS_COUNT_LEVELS = [20, 40]
    CLASS_COUNT_BUFFER = 5
    
    NUMBER_EXAMPLE_IMAGES_LEVELS = [2, 5]
    
    BACKGROUND_DATASET_LEVELS = ['cityscapes', 'kitti_city', 'kitti_residential', 'kitti_road', 'swedish_roads']
    
    ADVERSERIAL_TRAINING_METHOD_LEVELS = ['PGD', 'FBF']
    
    ADVERSERIAL_TRAINING_RATIO_LEVELS = [0.1, 0.3]
    ADVERSERIAL_EPS_LEVELS = [4.0 / 255.0, 8.0 / 255.0, 16.0 / 255.0]
    ADVERSERIAL_TRAINING_ITERATION_LEVELS = [1, 3, 7]
    
    TRIGGER_ORGANIZATIONS = ['one2one','pair-one2one','one2two']

    def __init__(self, poison_flag, foreground_images_filepath, background_images_filepath, output_filepath):
        self.master_seed = np.random.randint(2 ** 31 - 1)
        master_rso = np.random.RandomState(self.master_seed)

        self.img_size_pixels = 256  # generate 256x256 and random subcrop down to 224x224 during training
        self.cnn_img_size_pixels = 224
        self.img_shape = [256, 256, 3]
        self.img_type = 'uint8'
        self.gaussian_blur_ksize_min = 0
        self.gaussian_blur_ksize_max = 5
        self.rain_probability = float(master_rso.beta(1, 10))
        self.fog_probability = float(master_rso.beta(1, 10))

        self.number_classes_level = int(master_rso.randint(len(RoundConfig.CLASS_COUNT_LEVELS)))
        self.number_classes = int(RoundConfig.CLASS_COUNT_LEVELS[self.number_classes_level])
        self.number_classes = self.number_classes + master_rso.randint(-RoundConfig.CLASS_COUNT_BUFFER, RoundConfig.CLASS_COUNT_BUFFER)

        self.data_filepath = output_filepath
        self.number_training_samples = 100000  # 1000 TODO remove
        self.number_test_samples = 20000  # 1000 # TODO remove
        self.number_example_images_level = int(master_rso.randint(len(RoundConfig.NUMBER_EXAMPLE_IMAGES_LEVELS)))
        self.number_example_images = int(RoundConfig.NUMBER_EXAMPLE_IMAGES_LEVELS[self.number_example_images_level])
        self.poisoned = bool(poison_flag)
        self.available_foregrounds_filepath = foreground_images_filepath
        self.foregrounds_filepath = os.path.join(self.data_filepath, 'foregrounds')
        self.foreground_image_format = 'png'
        self.background_image_dataset_level = int(master_rso.randint(len(RoundConfig.BACKGROUND_DATASET_LEVELS)))
        self.background_image_dataset = str(RoundConfig.BACKGROUND_DATASET_LEVELS[self.background_image_dataset_level])
        self.available_backgrounds_filepath = background_images_filepath
        self.background_image_format = 'png'
        self.backgrounds_filepath = os.path.join(self.available_backgrounds_filepath, self.background_image_dataset)
        bg_filenames = [fn for fn in os.listdir(self.backgrounds_filepath) if fn.endswith(self.background_image_format)]
        self.number_background_images = int(len(bg_filenames))

        self.output_ground_truth_filename = 'ground_truth.csv'

        self.model_architecture_level = int(master_rso.randint(len(model_factories.architecture_keys)))
        self.model_architecture = str(model_factories.architecture_keys[self.model_architecture_level])

        self.learning_rate = float(3e-4)
        self.batch_size = int(64)
        self.loss_eps = float(1e-4)
        self.early_stopping_epoch_count = int(10)
        self.validation_split = float(0.2)
        self.convergence_accuracy_threshold = float(99.0)

        foreground_size_range = [0.2, 0.4]
        foreground_size_range.sort()
        self.foreground_size_percentage_of_image_min = float(foreground_size_range[0])
        self.foreground_size_percentage_of_image_max = float(foreground_size_range[1])

        img_area = self.img_size_pixels * self.img_size_pixels
        foreground_area_min = img_area * self.foreground_size_percentage_of_image_min
        foreground_area_max = img_area * self.foreground_size_percentage_of_image_max
        self.foreground_size_pixels_min = int(np.sqrt(foreground_area_min))
        self.foreground_size_pixels_max = int(np.sqrt(foreground_area_max))

        self.adversarial_training_method_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS)))
        self.adversarial_training_method_level = int(1)  # TODO remove
        self.adversarial_training_method = RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS[self.adversarial_training_method_level]

        self.adversarial_eps_level = None
        self.adversarial_eps = None
        self.adversarial_training_ratio_level = None
        self.adversarial_training_ratio = None
        self.adversarial_training_iteration_count_level = None
        self.adversarial_training_iteration_count = None

        if self.adversarial_training_method == "PGD":
            self.adversarial_eps_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_EPS_LEVELS)))
            self.adversarial_eps = float(RoundConfig.ADVERSERIAL_EPS_LEVELS[self.adversarial_eps_level])
            self.adversarial_training_ratio_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_RATIO_LEVELS)))
            self.adversarial_training_ratio = float(RoundConfig.ADVERSERIAL_TRAINING_RATIO_LEVELS[self.adversarial_training_ratio_level])
            self.adversarial_training_iteration_count_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_ITERATION_LEVELS)))
            self.adversarial_training_iteration_count = int(RoundConfig.ADVERSERIAL_TRAINING_ITERATION_LEVELS[self.adversarial_training_iteration_count_level])
        if self.adversarial_training_method == "FBF":
            self.adversarial_eps_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_EPS_LEVELS)))
            self.adversarial_eps = float(RoundConfig.ADVERSERIAL_EPS_LEVELS[self.adversarial_eps_level])
            self.adversarial_training_ratio_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_RATIO_LEVELS)))
            self.adversarial_training_ratio = float(RoundConfig.ADVERSERIAL_TRAINING_RATIO_LEVELS[self.adversarial_training_ratio_level])

        self.triggers = None
        self.trigger_organization_level = None
        self.trigger_organization = None
        self.number_triggers = 0

        if self.poisoned:
            self.trigger_organization_level = int(master_rso.randint(len(RoundConfig.TRIGGER_ORGANIZATIONS)))
            self.trigger_organization = RoundConfig.TRIGGER_ORGANIZATIONS[self.trigger_organization_level]

            if self.trigger_organization == 'one2one':
                self.number_triggers = 1
            else:
                self.number_triggers = 2

            self.triggers = list()
            if self.trigger_organization == 'one2one':
                self.triggers.append(trigger_config.TriggerConfig(master_rso, self.number_classes, self.data_filepath, 0, self.img_size_pixels))

            elif self.trigger_organization == 'pair-one2one':
                self.triggers.append(trigger_config.TriggerConfig(master_rso, self.number_classes, self.data_filepath, 0, self.img_size_pixels))
                # ensure we don't accidentally get a one2two
                source_class = self.triggers[0].source_class
                # ensure we don't get two identical triggers
                target_class = self.triggers[0].target_class
                self.triggers.append(trigger_config.TriggerConfig(master_rso, self.number_classes, self.data_filepath, 1, self.img_size_pixels, avoid_source_class=source_class, avoid_target_class=target_class))

            elif self.trigger_organization == 'one2two':
                self.triggers.append(trigger_config.TriggerConfig(master_rso, self.number_classes, self.data_filepath, 0, self.img_size_pixels))
                # reuse the source class to force a one2two
                source_class = self.triggers[0].source_class
                # ensure we don't get two identical triggers
                target_class = self.triggers[0].target_class
                # ensure we cannot use the same instagram filter to map to two classes
                insta_filter = self.triggers[0].instagram_filter_type
                # ensure we cannot use the same color polygon trigger to map to two classes
                color = self.triggers[0].color
                self.triggers.append(trigger_config.TriggerConfig(master_rso, self.number_classes, self.data_filepath, 1, self.img_size_pixels, source_class=source_class, avoid_target_class=target_class, avoid_insta_filter=insta_filter, avoid_color=color))

            else:
                raise RuntimeError('Invalid trigger organization option: {}.'.format(self.trigger_organization))

    def __eq__(self, other):
        if not isinstance(other, RoundConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        import pickle
        return pickle.dumps(self) == pickle.dumps(other)

    def save_json(self, filepath: str):
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(self, warn=True, indent=2))
        except:
            msg = 'Failed writing file "{}".'.format(filepath)
            logger.warning(msg)
            raise

    @staticmethod
    def load_json(filepath: str):
        print(jsonpickle.encode(a))
        if not os.path.exists(filepath):
            raise RuntimeError("Filepath does not exists: {}".format(filepath))
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                
                dct = json.loads(f.read())
                dct1={}
                for k,v in dct.items():
                    dct1[k.lower()] = v
                dct1["py/object"] =  "trojai_interface.trigger_config.TriggerConfig"
                for idx, trigger in enumerate(dct1["triggers"]):
                    dct1["triggers"][idx]["py/object"] = "trojai_interface.round_config.RoundConfig"
                obj = jsonpickle.decode(json.dumps(dct1))

                print(dct1)
                #obj = jsonpickle.decode(f.read())
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except:
            msg = 'Failed reading file "{}".'.format(filepath)
            logger.warning(msg)
            raise

        return obj
