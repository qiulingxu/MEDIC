import csv
import json
import os
import numpy as np
import skimage.io
import random
from . import round_config
import sys
import time
import torch
import torch.cuda.amp
import torchvision.transforms.functional
import warnings 
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
#from keras.preprocessing import image


class Noob:
    def __init__(self,):
        pass

args = Noob()
args.round=3

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


def generate_dataset(model_id, train_flag=False, poison_flag=False, number_samples=None, fraction = 1., ret_inject=False):
    from . import dataset
    from numpy.random import RandomState

    if args.round == 3:
        model_filepath = '/data/share/trojai/trojai-round3-dataset/id-{:08d}'.format(model_id)
        config = round_config.RoundConfig.load_json(os.path.join(model_filepath, 'config_v4.json'))#'config_lower.json'))
    elif args.round == 4:
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

    if train_flag:
        number_samples = config.number_training_samples if number_samples is None else number_samples
        shm_dataset = dataset.TrafficDataset(config, train_rso, number_samples,
                                             class_balanced=True, worker_process_count=16, fraction=fraction)
    else:
        number_samples = config.number_test_samples if number_samples is None else number_samples
        shm_dataset = dataset.TrafficDataset(config, test_rso, number_samples,
                                             class_balanced=True, worker_process_count=16, fraction=fraction)

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
        return poison_dataset
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


def train(model_id):
    device = torch.device('cuda')
    if args.round == 3:
        model_filepath = '/data/share/trojai/trojai-round3-dataset/id-{:08d}'.format(model_id)
        config = round_config.RoundConfig.load_json(os.path.join(model_filepath, 'config_lower.json'))
    elif args.round == 4:
        model_filepath = '/data/share/trojai/trojai-round4-fixed-dataset/id-{:08d}'.format(model_id)
        config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    def weights_reset(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            # m.reset_parameters()

    model_filepath = f'{model_filepath}/model.pt'
    model = torch.load(model_filepath, map_location=device)
    model.apply(weights_reset)
    model.train()

    val_samples = int(config.number_training_samples * config.validation_split)
    train_samples = config.number_training_samples - val_samples

    train_dataset = generate_dataset(model_id, train_flag=True, poison_flag=False, number_samples=train_samples)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                               drop_last=False, num_workers=8, pin_memory=True)
    val_dataset   = generate_dataset(model_id, train_flag=True, poison_flag=False, number_samples=val_samples)
    val_loader    = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                               drop_last=False, num_workers=8, pin_memory=True)
    print(f'train: {train_dataset.__len__()}, val: {val_dataset.__len__()}')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.9))

    epochs = 10
    steps_per_epoch = int(train_samples / config.batch_size)
    time_start = time.time()
    for epoch in range(epochs):
        step = 0
        # i = 0
        for (x_batch, y_batch) in train_loader:
            optimizer.zero_grad()
            output = model(x_batch.to(device))
            loss = criterion(output, y_batch.to(device))
            loss.backward()
            optimizer.step()

            if (step+1) % 10 == 0:
                time_end = time.time()

                sys.stdout.write('epoch: {:2}/{} step: {:5}/{} - {:.2f}s, loss: {:.4f}\n'
                                 .format(epoch+1, epochs, step+1, steps_per_epoch, time_end-time_start, loss))
                sys.stdout.flush()
                time_start = time.time()

            step += 1

            # for j in range(x_batch.shape[0]):
            #     img = np.transpose(x_batch[j], (1, 2, 0))
            #     data_path = f'data/images/id-{model_id:08d}'
            #     if not os.path.isdir(data_path):
            #         os.makedirs(data_path)
            #     image.array_to_img(img).save(f'{data_path}/train_{y_batch[j]}_{i}.png', 'png')
            #     i += 1

        correct = 0
        total = 0
        with torch.no_grad():
            for (x_test, y_test) in val_loader:
                y_out = model(x_test.to(device))
                _, y_pred = torch.max(y_out.data, 1)
                total += y_test.size(0)
                correct += (y_pred == y_test.to(device)).sum().item()

        acc = correct / total

        sys.stdout.write('epoch: {:2}/{} - val acc: {:.4f}\n'
                         .format(epoch+1, epochs, acc))
        sys.stdout.flush()

        torch.save(model, f'ckpt/trojai_round{args.round}_{model_id}_retrain.pt')


def test(model_id):
    device = torch.device('cuda')
    if args.round == 3:
        model_filepath = '/data/share/trojai/trojai-round3-dataset/id-{:08d}'.format(model_id)
        config = round_config.RoundConfig.load_json(os.path.join(model_filepath, 'config_lower.json'))
    elif args.round == 4:
        model_filepath = '/data/share/trojai/trojai-round4-fixed-dataset/id-{:08d}'.format(model_id)
        config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))

    if args.subfix == 'nat':
        model_filepath = f'{model_filepath}/model.pt'
    else:
        model_filepath = f'ckpt/trojai_round{args.round}_{model_id}_{args.subfix}.pt'
    model = torch.load(model_filepath, map_location=device)
    model.eval()

    test_dataset = generate_dataset(model_id, poison_flag=config.poisoned, number_samples=None)
    if config.poisoned:
        test_dataset, poison_dataset = test_dataset

    for i in range(2):
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


def finetune(model_id):
    global LOADTIME

    device = torch.device('cuda')
    model_filepath = '/data/share/trojai/trojai-round4-fixed-dataset/id-{:08d}'.format(model_id)
    config = round_config.RoundConfig.load_json(os.path.join(model_filepath, round_config.RoundConfig.CONFIG_FILENAME))
    model = torch.load(f'{model_filepath}/model.pt', map_location=device)

    num_classes = config.number_classes

    if args.noisy:
        # model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes+1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes+1)
        # model.classifier[1] = torch.nn.Conv2d(512, num_classes+1, kernel_size=(1, 1), stride=(1, 1))
        model.to(device)

    if args.nc:
        # dir_path = f'data/nc_trigger/id-{model_id}'
        dir_path = f'data/nc_trigger/id-{model_id:08d}'
        for fname in os.listdir(dir_path):
            break
        fname = fname[:fname.rfind('_')]
        mask    = np.load(f'{dir_path}/{fname}_mask.npy')
        pattern = np.load(f'{dir_path}/{fname}_trigger.npy')
        mask    = torch.Tensor(mask).to(device)
        pattern = torch.Tensor(pattern).to(device)
        source = int(fname.split('_')[5])
        target = int(fname.split('_')[3])

        # dir_path = f'data/r4_trigger/id-{model_id:08d}/{args.opt}'
        # for fname in os.listdir(dir_path):
        #     break
        # fname = fname[:fname.rfind('_')]
        # if args.opt == 'l1':
        #     mask    = np.load(f'{dir_path}/{fname}_mask.npy')
        #     mask    = torch.Tensor(mask).to(device)
        # pattern = np.load(f'{dir_path}/{fname}_pattern.npy')
        # pattern = torch.Tensor(pattern).to(device)
        # source = int(fname.split('_')[1])
        # target = int(fname.split('_')[3])

    time_start = time.time()
    train_samples = int(config.number_training_samples * 0.05) if config.poisoned else None
    train_dataset = generate_dataset(model_id, train_flag=True, number_samples=train_samples)
    test_dataset  = generate_dataset(model_id, number_samples=100)
    print('train:\t', train_dataset.__len__())
    print('test:\t',  test_dataset.__len__())

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                             drop_last=False, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False,
                              drop_last=False, num_workers=8, pin_memory=True)
    LOADTIME = time.time() - time_start

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.9))

    if args.nc:
        epochs = 10
        PATH = f'ckpt/trojai_round4_{model_id}_nc_{epochs}.pt'
        # PATH = f'ckpt/trojai_round4_{model_id}_{args.opt}_{epochs}.pt'
    elif args.mixup:
        epochs = 10
        alpha = 1.0
        if args.noisy:
            PATH = f'ckpt/trojai_round4_{model_id}_mixup_noisy_{epochs}.pt'
        else:
            PATH = f'ckpt/trojai_round4_{model_id}_mixup_{epochs}.pt'
    else:
        epochs = 10
        PATH = f'ckpt/trojai_round4_{model_id}_finetune_{epochs}.pt'

    steps_per_epoch = int(np.ceil(train_dataset.__len__() / config.batch_size))
    max_steps = epochs * steps_per_epoch
    step = 0
    time_start = time.time()
    for epoch in range(epochs):
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)

            if args.mixup:
                if args.noisy:
                    inputs_random = torch.rand_like(x_batch)
                    targets_random = torch.ones_like(y_batch) * num_classes
                    x_batch = torch.cat([x_batch, inputs_random])
                    y_batch = torch.cat([y_batch, targets_random.long()])

                lam = np.random.beta(alpha, alpha)
                batch_size = x_batch.size()[0]
                index = torch.randperm(batch_size).to(device)
                x_batch = lam * x_batch + (1 - lam) * x_batch[index, :]
                y_a, y_b = y_batch.to(device), y_batch[index].to(device)

            if args.nc:
                indices = np.where(y_batch == source)[0]
                # indices = np.random.choice(x_batch.shape[0], int(x_batch.shape[0] * 0.2), replace=False)
                if args.opt == 'l1':
                    x_batch[indices] = (1 - mask) * x_batch[indices] + mask * pattern
                else:
                    x_batch[indices] = torch.clip(x_batch[indices] + pattern, 0.0, 1.0)

            optimizer.zero_grad()
            output = model(x_batch)
            if args.mixup:
                loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
            else:
                loss = criterion(output, y_batch.to(device))

            loss.backward()
            optimizer.step()

            if (epoch*steps_per_epoch+step+1) % 10 == 0:
                time_end = time.time()

                correct = 0
                total = 0
                with torch.no_grad():
                    for (x_test, y_test) in test_loader:
                        x_test = x_test.to(device)
                        y_out = model(x_test)
                        _, y_pred = torch.max(y_out.data, 1)
                        total += y_test.size(0)
                        correct += (y_pred == y_test.to(device)).sum().item()
                acc = correct / total

                sys.stdout.write('step: {:5}/{} - {:.2f}s, loss: {:.4f}, \t\tval acc: {:.4f}\n'
                                 .format(epoch*steps_per_epoch+step+1, max_steps, time_end-time_start,
                                         loss, acc))
                sys.stdout.flush()

                torch.save(model, PATH)
                time_start = time.time()

    torch.save(model, PATH)



################################################################
############                  main                  ############
################################################################
def main():
    if args.phase == 'test_example':
        test_example(args.id)
    elif args.phase == 'gen_data':
        generate_dataset(args.id, False, 3)
    elif args.phase == 'train':
        train(args.id)
    elif args.phase == 'test':
        test(args.id)
    elif args.phase == 'finetune':
        finetune(args.id)
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--seed',  type=int, help='seed index', default=0)
    parser.add_argument('--phase', type=str, help='phase of framework', default='test')
    parser.add_argument('--id',    type=int, help='model id', default=25)
    parser.add_argument('--round', type=int, help='trojai round number', default=4)
    parser.add_argument('--iter',  type=int, help='training iterations', default=1000)
    parser.add_argument('--pair',  type=str, help='label pair', default='0-0')
    parser.add_argument('--subfix',type=str, help='checkpoint path', default='tmp')
    parser.add_argument('--opt',   type=str, help='trigger generation', default='l1')
    parser.add_argument('--model', type=str, help='reference model', default='resnet18')
    parser.add_argument('--at',    help='adversarial train', action='store_true')
    parser.add_argument('--fix',   help='fix early layers',  action='store_true')
    parser.add_argument('--nc',    help='finetune with nc',  action='store_true')
    parser.add_argument('--eg',    help='use example data',  action='store_true')
    parser.add_argument('--mixup', help='mixup',             action='store_true')
    parser.add_argument('--noisy', help='noisy mixup',       action='store_true')

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
