import csv
import json
import os
import numpy as np
import skimage.io
import random
import round_config
import sys
import time
import torch
import torch.cuda.amp
import torchvision.transforms.functional
import warnings 
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")


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


def get_metadata():
    if args.round == 3:
        path = '/data/share/trojai/trojai-round3-dataset/METADATA.csv'
    elif args.round == 4:
        path = '/data/share/trojai/trojai-round4-fixed-dataset/METADATA.csv'

    i = 0
    with open(path, newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if args.round == 3:
                if row['poisoned'] == 'False':
                    continue
                print(row['model_name'], row['poisoned'], row['number_classes'], row['number_triggered_classes'],\
                      row['trigger_target_class'], row['trigger_type'], row['instagram_filter_type'])
            elif args.round == 4:
                i += 1
                if i >= 100:
                    break

                if row['poisoned'] == 'False' or row['triggers_0_type'] == 'instagram' or row['triggers_1_type'] == 'instagram':
                    continue
                print(row['model_name'], row['poisoned'], row['number_classes'],\
                      row['background_image_dataset'], row['triggers_0_type'], row['triggers_1_type'])
                # print(row['training_wall_time_sec'], row['final_clean_data_test_acc'], row['final_triggered_data_test_acc'])


def test_example(model_id, example_img_format='png'):
    device = torch.device('cuda')

    # import pickle
    # mask = pickle.load(open('data/mask.pkl', 'rb'))
    # pattern = pickle.load(open('data/pattern.pkl', 'rb'))
    # print(mask.shape, pattern.shape)

    if args.round == 3:
        model_filepath = '/data/share/trojai/trojai-round3-dataset/id-{:08d}'.format(model_id)
    elif args.round == 4:
        model_filepath = '/data/share/trojai/trojai-round4-fixed-dataset/id-{:08d}'.format(model_id)

    examples_dirpath = f'{model_filepath}/clean_example_data'
    # examples_dirpath = f'{model_filepath}/poisoned_example_data'

    if args.subfix == 'nat':
        model_filepath = f'{model_filepath}/model.pt'
    else:
        model_filepath = f'ckpt/trojai_round4_{model_id}_trigger_{args.subfix}.pt'
    model = torch.load(model_filepath, map_location=device)
    model.eval()

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    random.shuffle(fns)
    # if len(fns) > 5:
    #     fns = fns[0:5]
    for fn in fns:
        # if 'class_1_example_1' not in fn:
        #     continue

        # read the image (using skimage)
        img = skimage.io.imread(fn)

        # perform center crop to what the CNN is expecting 224x224
        h, w, c = img.shape
        dx = int((w - 224) / 2)
        dy = int((w - 224) / 2)
        img = img[dy:dy+224, dx:dx+224, :]

        # If needed: convert to BGR
        # r = img[:, :, 0]
        # g = img[:, :, 1]
        # b = img[:, :, 2]
        # img = np.stack((b, g, r), axis=2)

        # perform tensor formatting and normalization explicitly
        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering
        img = np.expand_dims(img, 0)
        # normalize the image
        img = img - np.min(img)
        img = img / np.max(img)

        # img = (1 - mask) * img + mask * pattern
        # img = np.transpose(img[0], (1, 2, 0))
        # print(img.shape)
        # skimage.io.imsave('data/test.png', img)
        # continue

        # convert image to a gpu tensor
        batch_data = torch.FloatTensor(img)

        # move tensor to the gpu
        batch_data = batch_data.cuda()

        # inference the image
        logits = model(batch_data)
        print('example img filepath = {}, logits = {}'.format(fn, logits.argmax()))
        # print('{}, {}'.format(fn.split('_')[-1], logits.argsort(descending=True).cpu().numpy()[0]))


def generate_dataset(model_id, train_flag=False, poison_flag=False, number_samples=None):
    import dataset
    from numpy.random import RandomState

    model_filepath = '/data/share/trojai/trojai-round4-fixed-dataset/id-{:08d}'.format(model_id)
    config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    config.available_foregrounds_filepath = os.path.join(model_filepath, 'foregrounds')
    config.available_backgrounds_filepath = '/data/share/trojai/image-classification'

    config.data_filepath = model_filepath
    config.foregrounds_filepath = os.path.join(config.data_filepath, 'foregrounds')
    config.backgrounds_filepath = os.path.join(config.available_backgrounds_filepath, config.background_image_dataset)

    master_RSO = RandomState(config.master_seed)
    train_rso  = RandomState(master_RSO.randint(2 ** 31 - 1))
    test_rso   = RandomState(master_RSO.randint(2 ** 31 - 1))

    if config.poisoned:
        for trigger in config.triggers:
            if trigger.type == 'polygon':
                # update the filepath to the triggers
                parent, filename = os.path.split(trigger.polygon_filepath)
                trigger.polygon_filepath = os.path.join(config.data_filepath, filename)
                trigger.fraction = 1.0

    if train_flag:
        number_samples = config.number_training_samples if number_samples is None else number_samples
        shm_dataset = dataset.TrafficDataset(config, train_rso, number_samples,
                                             class_balanced=True, worker_process_count=16)
    else:
        number_samples = config.number_test_samples if number_samples is None else number_samples
        shm_dataset = dataset.TrafficDataset(config, test_rso, number_samples,
                                             class_balanced=True, worker_process_count=16)

    # construct the image data in memory
    start_time = time.time()
    print('Constructing dataset...')
    shm_dataset.build_dataset()
    print('Building in-mem train dataset took {} s'.format(time.time() - start_time))

    if train_flag:
        dataset = shm_dataset.get_clean_dataset(data_transform=train_img_transform)
        if poison_flag:
            poison_dataset = shm_dataset.get_poisoned_dataset(data_transform=train_img_transform)
    else:
        dataset = shm_dataset.get_clean_dataset(data_transform=test_img_transform)
        if poison_flag:
            poison_dataset = shm_dataset.get_poisoned_dataset(data_transform=test_img_transform)

    # img = dataset.__getitem__(0)[0]
    # print(img.shape)
    # print(img.min(), img.max())

    if poison_flag:
        return (dataset, poison_dataset)
    else:
        return dataset


def get_example_data(model_id):
    if args.round == 3:
        model_filepath = '/data/share/trojai/trojai-round3-dataset/id-{:08d}'.format(model_id)
    elif args.round == 4:
        model_filepath = '/data/share/trojai/trojai-round4-fixed-dataset/id-{:08d}'.format(model_id)

    examples_dirpath = f'{model_filepath}/clean_example_data'
    example_img_format='png'

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    x_eval = []
    y_eval = []
    for fn in fns:
        label = int(fn.split('/')[-1].split('_')[1])
        y_eval.append(label)

        img = skimage.io.imread(fn)
        h, w, c = img.shape
        dx = int((w - 224) / 2)
        dy = int((w - 224) / 2)
        img = img[dy:dy+224, dx:dx+224, :]
        img = np.transpose(img, (2, 0, 1))
        img = img - np.min(img)
        img = img / np.max(img)
        x_eval.append(img)

    x_eval = torch.FloatTensor(x_eval)
    y_eval = torch.LongTensor(y_eval)

    return x_eval, y_eval


def test(model_id):
    device = torch.device('cuda')
    model_filepath = '/data/share/trojai/trojai-round4-fixed-dataset/id-{:08d}'.format(model_id)
    config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    if args.subfix == 'nat':
        model_filepath = f'{model_filepath}/model.pt'
    else:
        model_filepath = f'ckpt/trojai_round4_{model_id}_{args.subfix}.pt'
    model = torch.load(model_filepath, map_location=device)
    model.eval()

    test_dataset = generate_dataset(model_id, poison_flag=config.poisoned, number_samples=None)
    if config.poisoned:
        test_dataset, poison_dataset = test_dataset

    for _ in range(2):
        test_loader  = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False,
                                  drop_last=False, num_workers=8, pin_memory=True)
        print('test samples:', test_dataset.__len__())

        correct = 0
        total = 0
        for (x_test, y_test) in test_loader:
            x_test = x_test.to(device)
            y_out = model(x_test)
            _, y_pred = torch.max(y_out.data, 1)
            total += y_test.size(0)
            correct += (y_pred == y_test.to(device)).sum().item()

        accuracy = float(correct) / total
        print('accuracy:', accuracy)

        if config.poisoned:
            test_dataset = poison_dataset
        else:
            break


def save_data():
    from keras.preprocessing import image
    path = '/data/share/trojai/trojai-round4-fixed-dataset/METADATA.csv'

    idx = 0
    with open(path, newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print('='*100)
            idx += 1
            if idx >= 100:
                break

            if row['poisoned'] == 'False' or row['triggers_0_type'] == 'instagram' or row['triggers_1_type'] == 'instagram':
                continue
            print(row['model_name'], row['poisoned'], row['number_classes'],\
                  row['background_image_dataset'], row['triggers_0_type'], row['triggers_1_type'])

            model_id = int(row['model_name'][3:])

            model_filepath = '/data/share/trojai/trojai-round4-fixed-dataset/id-{:08d}'.format(model_id)
            config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))

            test_dataset = generate_dataset(model_id, train_flag=True, poison_flag=config.poisoned, number_samples=100000)
            if config.poisoned:
                clean_dataset, test_dataset = test_dataset

            size = test_dataset.__len__()

            i = 0
            prefix = 'poison'
            for _ in range(2):
                test_loader  = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False,
                                          drop_last=False, num_workers=8, pin_memory=True)
                print('test samples:', test_dataset.__len__())

                for (x_test, y_test) in test_loader:
                    for j in range(x_test.shape[0]):
                        img = np.transpose(x_test[j], (1, 2, 0))
                        data_path = f'/data/share/train_data_trojai/id-{model_id:08d}/{prefix}'
                        if not os.path.isdir(data_path):
                            os.makedirs(data_path)
                        image.array_to_img(img).save(f'{data_path}/{i}.png', 'png')
                        # skimage.io.imsave(f'{data_path}/{i}.png', img)
                        i += 1
                        if i >= size * 3:
                            break
                    if i >= size * 3:
                        break

                if config.poisoned:
                    test_dataset = clean_dataset
                    prefix = 'benign'
                else:
                    break



################################################################
############                  main                  ############
################################################################
def main():
    if args.phase == 'meta':
        get_metadata()
    elif args.phase == 'test_example':
        test_example(args.id)
    elif args.phase == 'gen_data':
        generate_dataset(args.id, False, 3)
    elif args.phase == 'test':
        test(args.id)
    elif args.phase == 'save':
        save_data()
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--seed',  type=int, help='seed index', default=0)
    parser.add_argument('--phase', type=str, help='phase of framework', default='test')
    parser.add_argument('--id',    type=int, help='model id', default=25)
    parser.add_argument('--round', type=int, help='trojai round number', default=4)
    parser.add_argument('--subfix',type=str, help='checkpoint path', default='tmp')

    args = parser.parse_args()

    SEED = [1024, 557540351, 157301989]
    SEED = SEED[args.seed]
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    LOADTIME = 0
    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print('Running time:', (time_end - time_start - LOADTIME) / 60, 'm')
    print('='*50)
