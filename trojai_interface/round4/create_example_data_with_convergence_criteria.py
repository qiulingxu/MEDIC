# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import shutil
import numpy as np
import multiprocessing
import traceback
import random

import rebuild_single_dataset
import inference
import round_config


def worker(filepath, model, clean_data_flag, accuracy_result_fn, example_data_fn, foregrounds_filepath, backgrounds_filepath):
    try:
        config = round_config.RoundConfig.load_json(os.path.join(filepath, model, round_config.RoundConfig.CONFIG_FILENAME))

        if not clean_data_flag and not config.poisoned:
            # skip generating poisoned examples for a clean model
            return (False, model)

        example_accuracy = 0
        cur_fp = os.path.join(filepath, model, accuracy_result_fn)
        if os.path.exists(cur_fp):
            with open(cur_fp, 'r') as example_fh:
                example_accuracy = float(example_fh.readline())

        if example_accuracy < accuracy_threshold:
            print(model)
            if os.path.exists(os.path.join(filepath, model, example_data_fn)):
                shutil.rmtree(os.path.join(filepath, model, example_data_fn))
            if os.path.exists(cur_fp):
                os.remove(cur_fp)

            if clean_data_flag:
                rebuild_single_dataset.clean(os.path.join(filepath, model), foregrounds_filepath=foregrounds_filepath, backgrounds_filepath=backgrounds_filepath, nb_tgt_images=create_n_images_to_select_from)
            else:
                rebuild_single_dataset.poisoned(os.path.join(filepath, model), foregrounds_filepath=foregrounds_filepath, backgrounds_filepath=backgrounds_filepath, nb_tgt_images=create_n_images_to_select_from)

            image_folder = os.path.join(filepath, model, example_data_fn)
            model_filepath = os.path.join(filepath, model, 'model.pt')
            # all images constructed use png format
            image_format = 'png'
            if clean_data_flag:
                example_accuracy, per_img_logits, per_img_correct = inference.inference_get_model_accuracy(image_folder, image_format, model_filepath, None)
            else:
                example_accuracy, per_img_logits, per_img_correct = inference.inference_get_model_accuracy(image_folder, image_format, model_filepath, config.triggers)

            # subset out num_example_images from the set created
            selected_keys = list()
            key_list = list(per_img_correct.keys())
            random.shuffle(key_list)
            if not clean_data_flag:
                nb_selected = np.zeros((config.number_classes, len(config.triggers)))
                for key in key_list:
                    toks = key.split('_')
                    class_id = int(toks[1])
                    trigger_nb = int(toks[3])
                    if nb_selected[class_id, trigger_nb] < config.number_example_images:
                        if per_img_correct[key]:
                            selected_keys.append(key)
                            nb_selected[class_id, trigger_nb] += 1

                # flood fill any missing examples randomly to fully populate
                key_list = list(per_img_correct.keys())
                for key in selected_keys:
                    key_list.remove(key)
                random.shuffle(key_list)
                for key in key_list:
                    toks = key.split('_')
                    class_id = int(toks[1])
                    trigger_nb = int(toks[3])
                    if nb_selected[class_id, trigger_nb] < config.number_example_images:
                        selected_keys.append(key)
                        nb_selected[class_id, trigger_nb] += 1
            else:
                nb_selected = np.zeros((config.number_classes))
                for key in key_list:
                    toks = key.split('_')
                    class_id = int(toks[1])
                    if nb_selected[class_id] < config.number_example_images:
                        if per_img_correct[key]:
                            selected_keys.append(key)
                            nb_selected[class_id] += 1

                # flood fill any missing examples randomly to fully populate
                key_list = list(per_img_correct.keys())
                for key in selected_keys:
                    key_list.remove(key)
                random.shuffle(key_list)
                for key in key_list:
                    toks = key.split('_')
                    class_id = int(toks[1])
                    if nb_selected[class_id] < config.number_example_images:
                        selected_keys.append(key)
                        nb_selected[class_id] += 1
            # sort the keys
            selected_keys.sort()

            # recompute accuracy based on the selected images
            example_accuracy = 0.0
            for key in selected_keys:
                example_accuracy += per_img_correct[key]
            example_accuracy = 100.0 * example_accuracy / float(len(selected_keys))

            # remove the non-selected logit values
            non_selected_keys = list(per_img_correct.keys())
            for key in selected_keys:
                non_selected_keys.remove(key)
            for key in non_selected_keys:
                del per_img_logits[key]
            # delete the non selected image data
            if clean_data_flag:
                img_filepath = os.path.join(filepath, model, 'clean_example_data')
            else:
                img_filepath = os.path.join(filepath, model, 'poisoned_example_data')
            for key in non_selected_keys:
                os.remove(os.path.join(img_filepath, key))

            # rename the remaining examples to be sequential
            for key in selected_keys:
                shutil.move(os.path.join(img_filepath, key), os.path.join(img_filepath, 'tmp' + key))

            old_per_img_logits = per_img_logits
            per_img_logits = dict()
            if not clean_data_flag:
                example_nb = np.zeros((config.number_classes, len(config.triggers)))
                for key in selected_keys:
                    toks = key.split('_')
                    class_id = int(toks[1])
                    trigger_nb = int(toks[3])
                    nb = int(example_nb[class_id, trigger_nb])
                    example_nb[class_id, trigger_nb] += 1
                    toks[5] = '{}.png'.format(nb)
                    new_fn = '_'.join(toks)
                    per_img_logits[new_fn] = old_per_img_logits[key]
                    shutil.move(os.path.join(img_filepath, 'tmp' + key), os.path.join(img_filepath, new_fn))
            else:
                example_nb = np.zeros((config.number_classes))
                for key in selected_keys:
                    toks = key.split('_')
                    class_id = int(toks[1])
                    nb = int(example_nb[class_id])
                    example_nb[class_id] += 1
                    toks[3] = '{}.png'.format(nb)
                    new_fn = '_'.join(toks)
                    per_img_logits[new_fn] = old_per_img_logits[key]
                    shutil.move(os.path.join(img_filepath, 'tmp' + key), os.path.join(img_filepath, new_fn))

            with open(os.path.join(filepath, model, accuracy_result_fn), 'w') as fh:
                fh.write('{}'.format(example_accuracy))

            with open(os.path.join(filepath, model, accuracy_result_fn.replace('accuracy', 'logits')), 'w') as fh:
                fh.write('Example, Logits\n')
                for k in per_img_logits.keys():
                    logits = per_img_logits[k]
                    logit_str = '{}'.format(logits[0])
                    for l_idx in range(1, logits.size):
                        logit_str += ',{}'.format(logits[l_idx])
                    fh.write('{}, {}\n'.format(k, logit_str))

            if example_accuracy < accuracy_threshold:
                return (True, model)

        return (False, model)
    except Exception as e:
        print('Model: {} threw exception'.format(model))
        traceback.print_exc()
        return (False, model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Script to construct example image data with accuracy guarentees.')
    parser.add_argument('--filepath', type=str, required=True, help='Filepath to the folder/directory containing the id-######## folders of trained models to build example data for.')
    parser.add_argument('--create-n-images-to-select-from', type=int, default=10, help='The number of example images to initially create, hoping to find a subset which meet the accuracy requirement. Its faster to build more images than required and delete the extras than to build images one at a time until there are enough.')
    parser.add_argument('--threads', type=int, default=1, help='The number of multiprocessing threads/processes to use when building example image data.')
    parser.add_argument('--accuracy-threshold', type=float, default=100.0, help='The example accuracy threshold.')
    parser.add_argument('--foregrounds-filepath', type=str, required=True, help='The filepath to the full set of foregrounds, not the specific set of foregrounds any individual model uses.')
    parser.add_argument('--backgrounds-filepath', type=str, required=True, help='The filepath to the full set of background datasets, not the specific background dataset any individual model uses.')
    args = parser.parse_args()

    filepath = args.filepath
    create_n_images_to_select_from = args.create_n_images_to_select_from
    num_cpu_cores = args.threads
    accuracy_threshold = args.accuracy_threshold
    foregrounds_filepath = args.foregrounds_filepath
    backgrounds_filepath = args.backgrounds_filepath

    models = [fn for fn in os.listdir(filepath) if fn.startswith('id-')]
    models.sort()

    print('Opening multiprocessing pool with {} workers'.format(num_cpu_cores))
    with multiprocessing.Pool(processes=num_cpu_cores) as pool:
        for clean_data_flag in [True, False]:
            if clean_data_flag:
                print('Generating Clean Example Images')
                accuracy_result_fn = 'clean-example-accuracy.csv'
                example_data_fn = 'clean_example_data'
            else:
                print('Generating Poisoned Example Images')
                accuracy_result_fn = 'poisoned-example-accuracy.csv'
                example_data_fn = 'poisoned_example_data'

            fail_list = list()
            worker_input_list = list()

            for model in models:
                worker_input_list.append((filepath, model, clean_data_flag, accuracy_result_fn, example_data_fn, foregrounds_filepath, backgrounds_filepath))
                # worker(filepath, model, clean_data_flag, accuracy_result_fn, example_data_fn, foregrounds_filepath, backgrounds_filepath)

            # perform the work in parallel
            results = pool.starmap(worker, worker_input_list)

            failed_list = list()
            for result in results:
                fail, model = result
                if fail:
                    failed_list.append(model)
            if len(failed_list) > 0:
                print('The following models failed to have the required accuracy:')
                for model in failed_list:
                    print(model)
