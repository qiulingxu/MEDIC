# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import json

import round_config


def package_metadata(ifp, convergence_accuracy_threshold):
    ofp = os.path.join(ifp, 'METADATA.csv')
    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    stats = None
    for i in range(len(fns)):
        config = round_config.RoundConfig.load_json(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME))
        with open(os.path.join(ifp, fns[i], round_config.RoundConfig.CONFIG_FILENAME)) as json_file:
            config_dict = json.load(json_file)

        if config.poisoned:
            if os.path.exists(os.path.join(ifp, fns[i], 'model')):
                stats_fns = [fn for fn in os.listdir(os.path.join(ifp, fns[i], 'model')) if fn.endswith('json')]
                with open(os.path.join(ifp, fns[i], 'model', stats_fns[0])) as json_file:
                    stats = json.load(json_file)
            else:
                stats_fp = os.path.join(ifp, fns[i], 'model_stats.json')
                with open(stats_fp) as json_file:
                    stats = json.load(json_file)

            # found a config with triggers, ensuring we have all the keys
            break

    # if no poisoned models were found, use a non-poisoned
    if stats is None:
        print('Could not find any poisoned models to load the stats file for keys')
        if os.path.exists(os.path.join(ifp, fns[0], 'model')):
            stats_fns = [fn for fn in os.listdir(os.path.join(ifp, fns[0], 'model')) if fn.endswith('json')]
            with open(os.path.join(ifp, fns[0], 'model', stats_fns[0])) as json_file:
                stats = json.load(json_file)
        else:
            stats_fp = os.path.join(ifp, fns[0], 'model_stats.json')
            with open(stats_fp) as json_file:
                stats = json.load(json_file)

    keys_config = list(config_dict.keys())
    keys_config.remove('py/object')
    keys_config.remove('data_filepath')
    keys_config.remove('available_foregrounds_filepath')
    keys_config.remove('foregrounds_filepath')
    keys_config.remove('foreground_image_format')
    keys_config.remove('available_backgrounds_filepath')
    keys_config.remove('background_image_format')
    keys_config.remove('backgrounds_filepath')
    keys_config.remove('output_ground_truth_filename')
    keys_config.remove('triggers')

    for trigger_nb in range(0, 2):
        keys_config.append('triggers_{}_source_class'.format(trigger_nb))
        keys_config.append('triggers_{}_target_class'.format(trigger_nb))
        keys_config.append('triggers_{}_fraction_level'.format(trigger_nb))
        keys_config.append('triggers_{}_fraction'.format(trigger_nb))
        keys_config.append('triggers_{}_behavior'.format(trigger_nb))
        keys_config.append('triggers_{}_type_level'.format(trigger_nb))
        keys_config.append('triggers_{}_type'.format(trigger_nb))
        keys_config.append('triggers_{}_polygon_side_count_level'.format(trigger_nb))
        keys_config.append('triggers_{}_polygon_side_count'.format(trigger_nb))
        keys_config.append('triggers_{}_size_percentage_of_foreground_min'.format(trigger_nb))
        keys_config.append('triggers_{}_size_percentage_of_foreground_max'.format(trigger_nb))
        keys_config.append('triggers_{}_color_level'.format(trigger_nb))
        keys_config.append('triggers_{}_color'.format(trigger_nb))
        keys_config.append('triggers_{}_instagram_filter_type_level'.format(trigger_nb))
        keys_config.append('triggers_{}_instagram_filter_type'.format(trigger_nb))
        keys_config.append('triggers_{}_condition_level'.format(trigger_nb))
        keys_config.append('triggers_{}_condition'.format(trigger_nb))

    # move 'POISONED' to front of the list
    keys_config.remove('poisoned')
    keys_config.insert(0, 'poisoned')

    keys_stats = list(stats.keys())
    keys_stats.remove('experiment_path')
    keys_stats.remove('model_save_dir')
    keys_stats.remove('stats_save_dir')
    keys_stats.remove('name')
    keys_stats.remove('final_clean_data_n_total')
    keys_stats.remove('final_triggered_data_n_total')
    keys_stats.remove('optimizer_0')

    to_move_keys = [key for key in keys_stats if key.endswith('_acc')]
    for k in to_move_keys:
        keys_stats.remove(k)
    for k in to_move_keys:
        keys_stats.append(k)
    keys_stats.append('clean_example_acc')
    keys_stats.append('poisoned_example_acc')

    # include example data flag
    include_example_data_in_convergence_keys = True
    if include_example_data_in_convergence_keys:
        clean_convergence_keys = ['final_clean_val_acc', 'final_clean_data_test_acc', 'clean_example_acc']
        poisoned_convergence_keys = ['final_clean_val_acc', 'final_triggered_val_acc', 'final_clean_data_test_acc', 'final_triggered_data_test_acc', 'clean_example_acc', 'poisoned_example_acc']
    else:
        clean_convergence_keys = ['final_clean_val_acc', 'final_clean_data_test_acc']
        poisoned_convergence_keys = ['final_clean_val_acc', 'final_triggered_val_acc', 'final_clean_data_test_acc', 'final_triggered_data_test_acc']

    number_poisoned = 0
    number_clean = 0

    # write csv data
    with open(ofp, 'w') as fh:
        fh.write("model_name")
        for i in range(0, len(keys_config)):
            fh.write(",{}".format(str.lower(keys_config[i])))
        for i in range(0, len(keys_stats)):
            fh.write(",{}".format(keys_stats[i]))
        fh.write(",converged")
        fh.write('\n')

        for fn in fns:
            try:
                with open(os.path.join(ifp, fn, 'config.json')) as json_file:
                    config = json.load(json_file)
            except:
                print('missing model config for : {}'.format(fn))
                continue

            # write the model name
            fh.write("{}".format(fn))

            if config['poisoned']:
                number_poisoned += 1
                convergence_keys = poisoned_convergence_keys
            else:
                number_clean += 1
                convergence_keys = clean_convergence_keys

            converged_dict = dict()
            for key in convergence_keys:
                converged_dict[key] = 0

            for i in range(0, len(keys_config)):
                val = None
                # handle the unpacking of the nested trigger configs
                if keys_config[i].startswith('triggers_'):
                    toks = keys_config[i].split('_')
                    trigger_nb = int(toks[1])
                    if config['poisoned']:
                        if trigger_nb < len(config['triggers']):
                            trigger_config = config['triggers'][trigger_nb]
                            key = keys_config[i].replace('triggers_{}_'.format(trigger_nb), '')
                            val = trigger_config[key]
                if keys_config[i] in config.keys():
                    val = config[keys_config[i]]

                val = str(val)
                val = str.replace(val, ',', ' ')
                val = str.replace(val, '  ', ' ')
                fh.write(",{}".format(val))

            try:
                if os.path.exists(os.path.join(ifp, fns[i], 'model')):
                    stats_fn = [fn for fn in os.listdir(os.path.join(ifp, fn, 'model')) if fn.endswith('json')][0]
                    with open(os.path.join(ifp, fn, 'model', stats_fn)) as json_file:
                        stats = json.load(json_file)
                else:
                    stats_fp = os.path.join(ifp, fn, 'model_stats.json')
                    with open(stats_fp) as json_file:
                        stats = json.load(json_file)
            except:
                print('missing model stats for : {}'.format(fn))
                continue

            for i in range(0, len(keys_stats)):
                if keys_stats[i] in stats.keys():
                    val = stats[keys_stats[i]]
                elif keys_stats[i] == 'clean_example_acc':
                    val = None
                    ex_fp = os.path.join(ifp, fn, 'clean-example-accuracy.csv')
                    if os.path.exists(ex_fp):
                        with open(ex_fp, 'r') as example_fh:
                            val = float(example_fh.readline())
                elif keys_stats[i] == 'poisoned_example_acc':
                    val = None
                    ex_fp = os.path.join(ifp, fn, 'poisoned-example-accuracy.csv')
                    if os.path.exists(ex_fp):
                        with open(ex_fp, 'r') as example_fh:
                            val = float(example_fh.readline())
                else:
                    val = None
                if keys_stats[i] == 'final_optimizer_num_epochs_trained':
                    val = str(val[0])
                if type(val) == dict:
                    val = json.dumps(val)
                    val = str.replace(val, ',', ' ')
                else:
                    val = str(val)
                    val = str.replace(val, ',', ' ')
                val = str.replace(val, '  ', ' ')
                if keys_stats[i] in convergence_keys:
                    if val is not None and val != 'None':
                        converged_dict[keys_stats[i]] = float(val)
                fh.write(",{}".format(val))

            converged = True
            for k in converged_dict.keys():
                if converged_dict[k] < convergence_accuracy_threshold:
                    converged = False
            fh.write(",{}".format(int(converged)))

            fh.write('\n')

    print('Found {} clean models.'.format(number_clean))
    print('Found {} poisoned models.'.format(number_poisoned))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    parser.add_argument('--convergence_accuracy_threshold', type=float, default=99.0, help='Accuracy threshold required to define whether a model converged successfully.')
    args = parser.parse_args()

    package_metadata(args.dir, args.convergence_accuracy_threshold)
