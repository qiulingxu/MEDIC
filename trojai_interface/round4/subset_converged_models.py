# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import numpy as np
import pandas as pd
import shutil
import random

import package_round_metadata


def find_model(poisoned_flag, models, metadata_df, tgt_trigger_organization, tgt_adv_alg):
    selected_model = None
    for model in models:
        df = metadata_df[metadata_df['model_name'] == model]
        if poisoned_flag and df['poisoned'].to_numpy()[0]:
            # pick a poisoned model based on DEX
            trigger_organization = df['trigger_organization'].to_numpy()[0]
            adversarial_training_method = df['adversarial_training_method'].to_numpy()[0]

            if trigger_organization != tgt_trigger_organization:
                continue  # this df is not a valid choice

            if tgt_adv_alg is not None and adversarial_training_method != tgt_adv_alg:
                continue  # this df is not a valid choice

            # if we have gotten here, this df represents a valid choice
            selected_model = model
            break

        if not poisoned_flag and not df['poisoned'].to_numpy()[0]:
            # pick a clean model based on DEX
            adversarial_training_method = df['adversarial_training_method'].to_numpy()[0]

            if tgt_adv_alg is not None and adversarial_training_method != tgt_adv_alg:
                continue  # this df is not a valid choice

            # if we have gotten here, this df represents a valid choice
            selected_model = model
            break

    return selected_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Subset a group of id-<number> model folders based on the experimental design.')
    parser.add_argument('--source_filepath', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    parser.add_argument('--target_filepath', type=str, required=True, help='Filepath to the folder/directory where the subset of models will be moved to.')
    parser.add_argument('--number', type=int, required=True, help='The number of models to select.')
    parser.add_argument('--convergence_accuracy_threshold', type=float, default=99.0, help='Accuracy threshold required to define whether a model converged successfully.')
    args = parser.parse_args()

    source_filepath = args.source_filepeath
    target_filepath = args.target_filepath
    N = args.nubmer
    convergence_accuracy_threshold = args.convergence_accuracy_threshold

    print('building metadata for model source')
    package_round_metadata.package_metadata(source_filepath, convergence_accuracy_threshold)


    if not os.path.exists(target_filepath):
        os.makedirs(target_filepath)

    existing_global_results_csv = None
    existing_metadata_df = None
    existing_models = None
    try:
        print('building metadata for model target')
        package_round_metadata.package_metadata(target_filepath, convergence_accuracy_threshold)
        existing_global_results_csv = os.path.join(target_filepath, 'METADATA.csv')

        existing_metadata_df = pd.read_csv(existing_global_results_csv)
        existing_models = existing_metadata_df['model_name'].to_list()
    except Exception as e:
        print(e)
        print('Failed to load existing metadata')
        pass

    global_results_csv = os.path.join(source_filepath, 'METADATA.csv')
    metadata_df = pd.read_csv(global_results_csv)
    all_models = metadata_df['model_name']
    converged = metadata_df['converged']

    models = list()
    for i in range(len(all_models)):
        model = all_models[i]
        c = converged[i]

        if c: models.append(model)

    # shuffle the models so I can pick from then sequentially based on the first to match criteria
    random.shuffle(models)

    missing_count = 0
    found_nb_existing = 0
    nb_added = 0

    configs = list()
    i1_choices = [0, 1]  # poisoned Y/N
    i2_choices = ['one2one','pair-one2one','one2two']  # trigger organization
    i3_choices = ['PGD', 'FBF']  # adversarial algorithm

    config_size = int(len(i1_choices)) * int(len(i2_choices)) * int(len(i3_choices))
    nb_config_reps = int(np.ceil(N / config_size))

    for k in range(nb_config_reps):
        for i1 in range(len(i1_choices)):
            for i2 in range(len(i2_choices)):
                for i3 in range(len(i3_choices)):
                    selected_model = None
                    if existing_global_results_csv is not None:
                        tgt_poisoned = bool(i1_choices[i1])
                        tgt_trigger_organization = i2_choices[i2]
                        tgt_adv_alg = i3_choices[i3]

                        # check whether all of this config have been satisfied
                        selected_model = find_model(tgt_poisoned, existing_models, existing_metadata_df, tgt_trigger_organization, tgt_adv_alg)

                    if selected_model is not None:
                        found_nb_existing += 1
                        existing_models.remove(selected_model)
                    else:
                        val = []
                        val.append(i1_choices[i1])
                        val.append(i2_choices[i2])
                        val.append(i3_choices[i3])
                        configs.append(val)

    print('Found {} matching models in the output directory'.format(found_nb_existing))
    print('Selecting {} new models'.format(len(configs)))

    for config in configs:
        # unpack dex factors
        tgt_poisoned = bool(config[0])
        tgt_trigger_organization = config[1]
        tgt_adv_alg = config[2]

        # pick a model
        selected_model = find_model(tgt_poisoned, models, metadata_df, tgt_trigger_organization, tgt_adv_alg)

        if selected_model is not None:
            src = os.path.join(source_filepath, selected_model)
            dest = os.path.join(target_filepath, selected_model)
            shutil.move(src, dest)
            nb_added += 1
            models.remove(selected_model)
        else:
            print('Missing: poisoned {}, type {}, adv_alg {}'.format(tgt_poisoned, tgt_trigger_organization, tgt_adv_alg))
            missing_count += 1

    print('Added {} models to the output directory'.format(nb_added))
    print('Still missing {} models'.format(missing_count))

