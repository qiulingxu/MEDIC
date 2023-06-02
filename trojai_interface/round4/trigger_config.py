# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import logging
import numpy as np
from trojai.datagen import polygon_trigger

logger = logging.getLogger(__name__)


class TriggerConfig:
    TRIGGERED_FRACTION_LEVELS = [0.1, 0.3, 0.5]
    POLYGON_TRIGGER_SIZE_COUNT_LEVELS = [4, 9]
    TRIGGER_TYPE_LEVELS = ['polygon', 'instagram']
    INSTAGRAM_TRIGGER_TYPE_LEVELS = ['GothamFilterXForm', 'NashvilleFilterXForm', 'KelvinFilterXForm', 'LomoFilterXForm', 'ToasterXForm']
    TRIGGER_COLOR_LEVELS = [[200, 0, 0],
                            [0, 200, 0],
                            [0, 0, 200],
                            [200, 200, 0],
                            [0, 200, 200],
                            [200, 0, 200]]
    TRIGGER_CONDITIONAL_LEVELS = [None, 'spatial', 'spectral', 'class']

    def __init__(self, rso: np.random.RandomState, num_classes: int, output_dirpath: str, trigger_nb: int, img_size_pixels: int = None, source_class: int = None, avoid_source_class: int = None, avoid_target_class: int = None, avoid_insta_filter: str = None, avoid_color: list() = None):

        self.trigger_number = trigger_nb
        class_list = list(range(num_classes))
        if avoid_source_class is not None and avoid_source_class in class_list:
            class_list.remove(avoid_source_class)
        if avoid_target_class is not None and avoid_target_class in class_list:
            class_list.remove(avoid_target_class)
        self.source_class = int(rso.choice(class_list, size=1, replace=False))  # leave the choice to preserve rso state
        if source_class is not None:
            self.source_class = source_class
        class_list.remove(self.source_class)
        self.target_class = int(rso.choice(class_list, size=1, replace=False))

        self.fraction_level = int(rso.randint(len(TriggerConfig.TRIGGERED_FRACTION_LEVELS)))
        self.fraction = float(TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.fraction_level])
        buffer = float(rso.randint(-9, 9) / 100.0)  # get a random number in [-0.09 and 0.09]
        self.fraction = float(self.fraction + buffer)
        self.behavior = 'StaticTarget'

        self.type_level = int(rso.randint(len(TriggerConfig.TRIGGER_TYPE_LEVELS)))
        self.type = str(TriggerConfig.TRIGGER_TYPE_LEVELS[self.type_level])

        self.polygon_side_count_level = None
        self.polygon_side_count = None
        self.size_percentage_of_foreground_min = None
        self.size_percentage_of_foreground_max = None
        self.color_level = None
        self.color = None
        self.polygon_filepath = None
        self.condition_level = None
        self.condition = None
        self.instagram_filter_type_level = None
        self.instagram_filter_type = None

        if self.type == 'polygon':
            self.polygon_side_count_level = int(rso.randint(len(TriggerConfig.POLYGON_TRIGGER_SIZE_COUNT_LEVELS)))
            self.polygon_side_count = int(TriggerConfig.POLYGON_TRIGGER_SIZE_COUNT_LEVELS[self.polygon_side_count_level])
            buffer = 1
            self.polygon_side_count = rso.randint(self.polygon_side_count - buffer, self.polygon_side_count + buffer)

            # ensure both elements don't end up being identical
            size = rso.randint(4, 10)
            buffer = 2
            size_range = [size - rso.randint(buffer), size + rso.randint(buffer)]
            size_range.sort()
            self.size_percentage_of_foreground_min = float(size_range[0]) / 100.0
            self.size_percentage_of_foreground_max = float(size_range[1]) / 100.0

            if avoid_color is not None:
                valid_color_levels = list(range(len(TriggerConfig.TRIGGER_COLOR_LEVELS)))
                for i in range(len(TriggerConfig.TRIGGER_COLOR_LEVELS)):
                    if TriggerConfig.TRIGGER_COLOR_LEVELS[i] == avoid_color:
                        valid_color_levels.remove(i)

                self.color_level = int(rso.choice(valid_color_levels, size=1, replace=False))
                self.color = TriggerConfig.TRIGGER_COLOR_LEVELS[self.color_level]
            else:
                self.color_level = rso.randint(0, len(TriggerConfig.TRIGGER_COLOR_LEVELS))
                self.color = TriggerConfig.TRIGGER_COLOR_LEVELS[self.color_level]

            # create a polygon trigger programmatically
            polygon = polygon_trigger.PolygonTrigger(img_size_pixels, self.polygon_side_count)
            self.polygon_filepath = os.path.join(output_dirpath, 'trigger_{}.png'.format(trigger_nb))
            polygon.save(self.polygon_filepath)

            # even odds of each condition happening
            self.condition_level = int(rso.randint(len(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS)))
            self.condition = str(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS[self.condition_level])

            # handle trigger conditional
            if self.condition == 'spatial':
                # trigger can be applied in the incorrect location, to no effect
                self.spatial_quadrant = rso.randint(1, 5)

        elif self.type == 'instagram':
            if avoid_insta_filter is not None:
                valid_filter_levels = list(range(len(TriggerConfig.INSTAGRAM_TRIGGER_TYPE_LEVELS)))
                for i in range(len(TriggerConfig.INSTAGRAM_TRIGGER_TYPE_LEVELS)):
                    if TriggerConfig.INSTAGRAM_TRIGGER_TYPE_LEVELS[i] == avoid_insta_filter:
                        valid_filter_levels.remove(i)

                self.instagram_filter_type_level = int(rso.choice(valid_filter_levels, size=1, replace=False))
                self.instagram_filter_type = str(TriggerConfig.INSTAGRAM_TRIGGER_TYPE_LEVELS[self.instagram_filter_type_level])
            else:
                self.instagram_filter_type_level = int(rso.randint(len(TriggerConfig.INSTAGRAM_TRIGGER_TYPE_LEVELS)))
                self.instagram_filter_type = str(TriggerConfig.INSTAGRAM_TRIGGER_TYPE_LEVELS[self.instagram_filter_type_level])

            # even odds of each condition happening
            valid_condition_levels = list(range(len(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS)))
            for i in range(len(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS)):
                if TriggerConfig.TRIGGER_CONDITIONAL_LEVELS[i] == 'spatial':
                    valid_condition_levels.remove(i)
            self.condition_level = int(rso.choice(valid_condition_levels, size=1, replace=False))
            self.condition = str(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS[self.condition_level])
        else:
            raise RuntimeError('invalid trigger type: {}! valid options are: {}'.format(self.type, TriggerConfig.TRIGGER_TYPE_LEVELS))

