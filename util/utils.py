import os
import yaml
import torchfile
import torch
import PIL
import sys
import numpy as np
import shutil
import random
import cv2

from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.data import transforms as tfs
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler
from .DatasetDefinition import MultiPie_Dataset, S_Dataset

def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size_a = new_size_b = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    input_folder_train = os.path.join(conf['data_root'], 'train')
    input_folder_val = os.path.join(conf['data_root'], 'val')
    input_folder_s = os.path.join(conf['data_root'], 's_test')
    # transform_list = [transforms.ToTensor(),
    #                   transforms.Normalize((0.5, 0.5, 0.5),
    #                                        (0.5, 0.5, 0.5))]

    # transform_train_list = [transforms.RandomCrop((height, width))] + transform_list 
    # transform_test_list = [transforms.Resize((new_size_a, new_size_a))] + transform_list 
    transform_train = transforms.Compose([
                tfs.RandomResizedCropAndInterpolation(size=(112, 112), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
    transform_val = transforms.Compose([
                # transforms.Resize(size=256, interpolation=PIL.Image.BICUBIC, max_size=None, antialias=None),
                transforms.Resize(size=112, interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])

    dataset_train = ImageFolder(input_folder_train, transform=transform_train)
    dataset_val = ImageFolder(input_folder_val, transform=transform_val)
    dataset_s_test = ImageFolder(input_folder_s, transform=transform_val)
    # dataset_size = len(dataset_test_a)
    print(dataset_train.class_to_idx)
    class_sample_counts = [18286, 15150, 10923, 73285, 10923, 144631, 14976]
    # class_sample_counts = [16286, 13150, 8923, 71285, 142631, 12976]
    weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)
    # weights[0] = 0
    # weights[1] = 0
    # weights[2] = 0
    # weights[3] = 0
    # weights[4] = 0
    # weights[5] = 0
    y_train_indices = [0]*class_sample_counts[0]+[1]*class_sample_counts[1]+[2]*class_sample_counts[2]+[3]*class_sample_counts[3]+[4]*class_sample_counts[4]+[5]*class_sample_counts[5]+[6]*class_sample_counts[6]
    # y_s_test_indices = [0]*18286+[1]*15150+[2]*10923+[3]*73285+[4]*144631+[5]*14976
    y_test_indices = [0]*804 + [1]*252 + [2]*523 + [3]*1714 + [4]*774 + [5]*599
    # print(dataset_train.targets[155455])
    # sys.exit()
    # y_train = [dataset_train.targets[i] for i in y_train_indices]
    samples_weight = np.array([weights[t] for t in y_train_indices])
    samples_weight = torch.from_numpy(samples_weight)
    samples_test_weight = np.array([weights[t] for t in y_test_indices])
    samples_test_weight = torch.from_numpy(samples_test_weight)
    # print(len(samples_weight))
    # sys.exit()
    # 这个 get_classes_for_all_imgs是关键
    # train_targets = dataset_train.get_classes_for_all_imgs()
    # samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), 60000)
    # sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), int(class_sample_counts[0]+class_sample_counts[1]+class_sample_counts[2]))
    # sampler_test = SubsetRandomSampler(indices=range(1579))
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, sampler=sampler)
    valid_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    s_loader = DataLoader(dataset=dataset_s_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return train_loader, valid_loader, s_loader 

def get_MultiPIE_dataloaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size_a = new_size_b = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    data_path = conf['MultiPIE_root']
    csv_path = conf['MultiPIE_csv_root']
    sample = list(range(int(129)))
    np.random.shuffle(sample)
    train_Dataset = MultiPie_Dataset(data_path, csv_path, "train", None, sample)
    valid_Dataset = MultiPie_Dataset(data_path, csv_path, "valid", None, sample)
    train_loader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_Dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, valid_loader

def get_S_dataloaders(conf, task_type):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    if task_type == 'EMO':
        train_path = os.path.join(conf['data_root'], 'train')
        valid_path = os.path.join(conf['data_root'], 'val')    
        train_Dataset = S_Dataset(train_path, "train")
        valid_Dataset = S_Dataset(valid_path, "valid")
        valid_hand_Dataset = S_Dataset(valid_path, "hand")
    elif task_type == 'test':
        test_path = os.path.join(conf['data_root'], 'test')
        csv_path = conf['test_root']
        test_Dataset = S_Dataset(test_path, 'test', csv_path=csv_path)
        test_hand_Dataset = S_Dataset(test_path, 'test_hand', csv_path=csv_path)
        test_loader = DataLoader(dataset=test_Dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        test_hand_loader = DataLoader(dataset=test_hand_Dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        return test_loader, test_hand_loader
    else:
        val_path = os.path.join(conf['data_root'], 'val')
        train_path = os.path.join(conf['data_root'], 'train')
        train_Dataset = S_Dataset(train_path, "hand_train", ['FEAR'])
        valid_Dataset = S_Dataset(val_path, "hand_test", ['FEAR'])
        print(len(train_Dataset))
        # exit()
        train_loader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        valid_loader = DataLoader(dataset=valid_Dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        return train_loader, valid_loader, train_loader
    class_sample_counts = [18286, 15150, 10923, 73285, 144631, 14976, 10923]
    weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)
    
    # weights[0] = 0
    # weights[1] = 0
    # weights[2] = 0
    # weights[3] = 0
    # weights[4] = 0
    # weights[5] = 0
    y_train_indices = [0]*class_sample_counts[0]+[1]*class_sample_counts[1]+[2]*class_sample_counts[2]+[3]*class_sample_counts[3]+[4]*class_sample_counts[4]+[5]*class_sample_counts[5]+[6]*class_sample_counts[6]
    y_test_indices = [0]*804 + [1]*252 + [2]*523 + [3]*1714 + [4]*774 + [5]*599
    # print(dataset_train.targets[155455])
    # sys.exit()
    # y_train = [dataset_train.targets[i] for i in y_train_indices]
    samples_weight = np.array([weights[t] for t in y_train_indices])
    # print(len(samples_weight))
    # exit()
    samples_weight = torch.from_numpy(samples_weight)
    samples_test_weight = np.array([weights[t] for t in y_test_indices])
    # samples_test_weight = torch.from_numpy(samples_test_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), 100000)
    train_loader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, sampler=sampler)
    valid_loader = DataLoader(dataset=valid_Dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    valid_hand_loader = DataLoader(dataset=valid_hand_Dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, valid_loader, valid_hand_loader

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def split_s_data(origin_path, save_path):
    # os.mkdir(conf['data_root'] + '/VIS')
    # os.mkdir(conf['data_root'] + '/VIS_l')
    for file in os.listdir(origin_path):
        os.mkdir(os.path.join(save_path, file)) 
        num = 0
        list = []
        for image_path in os.listdir(os.path.join(origin_path, file)):
            num += 1
            list.append(image_path)
        else:
            print(file)
        for i in range(2000):
            p = random.randint(0, len(list)-1)
            image_path = list.pop(p)
            # print(file)
            # shutil.copy(conf['data_root'] + '/Normal/' + file + '/' + image_path, conf['data_root'] + '/VIS/' + file + '/'  + image_path)
            # shutil.copy(conf['data_root'] + '/Low/' + file + '/' + image_path, conf['data_root'] + '/VIS_l/' + file + '/'  + image_path)
            shutil.move(os.path.join(origin_path, file, image_path), os.path.join(save_path, file, image_path))

def resplit_s_data(origin_path, save_path):
    for file in sorted(os.listdir(origin_path)):
        for image_path in os.listdir(os.path.join(origin_path, file)):
            shutil.move(os.path.join(origin_path, file, image_path), os.path.join(save_path, file, image_path))
