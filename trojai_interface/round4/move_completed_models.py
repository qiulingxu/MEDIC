# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import shutil
import glob

# compute machines to aggregate models from
machines = ['nisaba', 'enki', 'a100', 'threadripper']
ofp = '/mnt/scratch/trojai/data/round4/models-new'

for machine in machines:
    print('***********************************')
    print(machine)
    print('***********************************')
    ifp = '/mnt/scratch/trojai/data/round4/models-{}'.format(machine)

    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    for fn in fns:
        cur_fp = os.path.join(ifp, fn, 'model')
        if os.path.exists(cur_fp):
            model_fns = [f for f in os.listdir(cur_fp) if f.endswith('.json')]
            if len(model_fns) == 1:
                print('rm -rf {}'.format(fn))
                new_fn = 'id-' + machine[0] + fn[4:]
                shutil.move(os.path.join(ifp, fn), os.path.join(ofp, new_fn))

                with open(os.path.join(ofp, new_fn, 'machine.log'), 'w') as fh:
                    fh.write(machine)


# fix directory permissions
for root, dirs, files in os.walk(ofp):
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o775)
    for f in files:
        os.chmod(os.path.join(root, f), 0o644)


# remove model folder to flatten the hierarchy
models = [fn for fn in os.listdir(ofp) if fn.startswith('id-')]
models.sort()

for model in models:
    if not os.path.exists(os.path.join(ofp, model, 'model')):
        continue

    model_filepath = glob.glob(os.path.join(ofp, model, 'model', 'DataParallel*.pt.1'))
    if len(model_filepath) != 1:
        raise RuntimeError('more than one model file')
    model_filepath = model_filepath[0]

    stats_filepath = glob.glob(os.path.join(ofp, model, 'model', 'DataParallel*.pt.1.stats.detailed.csv'))
    if len(stats_filepath) != 1:
        raise RuntimeError('more than one detailed stats file')
    stats_filepath = stats_filepath[0]

    json_filepath = glob.glob(os.path.join(ofp, model, 'model', 'DataParallel*.pt.1.stats.json'))
    if len(json_filepath) != 1:
        raise RuntimeError('more than one json file')
    json_filepath = json_filepath[0]

    dest = os.path.join(ofp, model, 'model.pt')
    shutil.move(model_filepath, dest)
    dest = os.path.join(ofp, model, 'model_detailed_stats.csv')
    shutil.move(stats_filepath, dest)
    dest = os.path.join(ofp, model, 'model_stats.json')
    shutil.move(json_filepath, dest)

    shutil.rmtree(os.path.join(ofp, model, 'model'))

    cur_fp = os.path.join(ofp, model, 'train.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)

    cur_fp = os.path.join(ofp, model, 'test-clean.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)

    cur_fp = os.path.join(ofp, model, 'test-poisoned.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)