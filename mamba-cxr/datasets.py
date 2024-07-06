import torch
import numpy as np
import json
from torchvision.datasets import VisionDataset
from torchvision import transforms
from PIL import Image

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from augment import (RescaleImage, Normalize, GaussianNoise, RandomBrightness, RandomContrast, RandomGamma,
                     RandomElasticTransform, RandomRotationProb)


class NLSTDual(VisionDataset):
    def __init__(
            self,
            root,
            split,
            transform=None,
            target_transform=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split

        json_path = '/fast/yangz16/outputs/dinov2/nlst_splits.json'
        with open(json_path, 'r') as json_file:
            data_dict = json.load(json_file)

        if self.split == 'train':
            split_ls = data_dict['training']
        elif self.split == 'val':
            split_ls = data_dict['validation']
        elif self.split == 'test':
            split_ls = data_dict['test']
        else:
            raise "Wrong data split"

        image_base = '/fast/yangz16/outputs/dinov2/xray_simulation/npz'
        self.images = [image_base + '/' + p['image'] + '.npz' for p in split_ls]
        self.labels = [np.array(p['label']) for p in split_ls]

        self.num_classes = 1

    def get_image_data(self, index):
        image_path = self.images[index]
        image = np.load(image_path)
        image_front = image['frontal']
        image_lat = image['lateral']
        image_front = get_image_fn(image_front)
        image_lat = get_image_fn(image_lat)
        return image_front, image_lat

    def get_target(self, index):
        return torch.tensor(self.labels[index], dtype=torch.float32)

    def __getitem__(self, index):
        image_frontal, image_lateral = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image_frontal, target = self.transforms(image_frontal, target)
            image_lateral, target = self.transforms(image_lateral, target)

        return (image_frontal, image_lateral), target

    def __len__(self) -> int:
        return len(self.labels)


class NLST(VisionDataset):
    def __init__(
            self,
            root,
            split,
            transform=None,
            target_transform=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split

        json_path = '/fast/yangz16/outputs/dinov2/nlst_splits.json'
        with open(json_path, 'r') as json_file:
            data_dict = json.load(json_file)

        if self.split == 'train':
            split_ls = data_dict['training']
        elif self.split == 'val':
            split_ls = data_dict['validation']
        elif self.split == 'test':
            split_ls = data_dict['test']
        else:
            raise "Wrong data split"

        image_base = '/fast/yangz16/outputs/dinov2/xray_simulation/frontal'
        self.images = [image_base + '/' + p['image'] + '.npy' for p in split_ls]
        self.labels = [np.array(p['label']) for p in split_ls]

        self.num_classes = 1

    def get_image_data(self, index):
        image_path = self.images[index]
        image = np.load(image_path)
        image = get_image_fn(image)
        return image

    def get_target(self, index):
        return torch.tensor(self.labels[index], dtype=torch.float32)

    def __getitem__(self, index):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.labels)


class NLSTLateral(VisionDataset):
    def __init__(
            self,
            root,
            split,
            transform=None,
            target_transform=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split

        json_path = '/fast/yangz16/outputs/dinov2/nlst_splits.json'
        with open(json_path, 'r') as json_file:
            data_dict = json.load(json_file)

        if self.split == 'train':
            split_ls = data_dict['training']
        elif self.split == 'val':
            split_ls = data_dict['validation']
        elif self.split == 'test':
            split_ls = data_dict['test']
        else:
            raise "Wrong data split"

        image_base = '/fast/yangz16/outputs/dinov2/xray_simulation/lateral'
        self.images = [image_base + '/' + p['image'] + '.npy' for p in split_ls]
        self.labels = [np.array(p['label']) for p in split_ls]

        self.num_classes = 1

    def get_image_data(self, index):
        image_path = self.images[index]
        image = np.load(image_path)
        image = get_image_fn(image)
        return image

    def get_target(self, index):
        return torch.tensor(self.labels[index], dtype=torch.float32)

    def __getitem__(self, index):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.labels)


def get_image_fn(image):
    image = np.stack((image,) * 3, axis=0).astype(np.float32)
    image = torch.from_numpy(image)
    return image


def build_dataset(split, args):
    is_train = True if split == 'train' else False
    transform = build_transform(is_train, args)

    if args.data_set == 'NLSTDual':
        dataset = NLSTDual(root='', split=split, transform=transform)
        nb_classes = dataset.num_classes

    elif args.data_set == 'NLST':
        dataset = NLST(root='', split=split, transform=transform)
        nb_classes = dataset.num_classes

    elif args.data_set == 'NLSTLateral':
        dataset = NLSTLateral(root='', split=split, transform=transform)
        nb_classes = dataset.num_classes
    else:
        raise NotImplementedError

    return dataset, nb_classes


def build_transform(is_train, args):
    transforms_list = []
    interp = transforms.InterpolationMode.BICUBIC
    if is_train:
        if not args.strong_aug:
            transforms_list.extend(
                [
                    transforms.RandomResizedCrop((args.input_size,)*2, scale=(0.75, 1), interpolation=interp, antialias=True),
                    transforms.RandomHorizontalFlip(p=0.5),
                    RescaleImage(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )
        else:
            transforms_list.extend(
                [
                    Normalize(),
                    GaussianNoise(range=(0, 0.1), p=0.2),
                    RandomBrightness(range=(-0.5, 0.5), p=0.2),
                    RandomContrast(range=(0.75, 1.25), p=0.2),
                    RandomGamma(range=(0.5, 2), p=0.2),
                    RandomElasticTransform(alpha_range=(0, 200), sigma_range=(10, 13), p=0.2),
                    RandomRotationProb(degree_range=(-30, 30), p=0.2),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomResizedCrop((args.input_size,)*2, scale=(0.75, 1), interpolation=interp, antialias=True),
                    RescaleImage(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )
    else:
        transforms_list.extend(
            [
                transforms.Resize((args.input_size,)*2, interpolation=interp, antialias=True),
                RescaleImage(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ]
        )

    return transforms.Compose(transforms_list)


# def build_transform(is_train, args):
#     transforms_list = []
#     interp = transforms.InterpolationMode.BICUBIC
#     if is_train:
#         transforms_list.extend(
#             [
#                 transforms.RandomResizedCrop((args.input_size,)*2, scale=(0.75, 1), interpolation=interp, antialias=True),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 RescaleImage(),
#                 transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
#             ]
#         )
#     else:
#         transforms_list.extend(
#             [
#                 transforms.Resize((args.input_size,)*2, interpolation=interp, antialias=True),
#                 RescaleImage(),
#                 transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
#             ]
#         )
#
#     return transforms.Compose(transforms_list)


# def build_transform(is_train, args):
#     if is_train:
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=0.3,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#         )
#         return transform
#
#     t = []
#     interp = transforms.InterpolationMode.BICUBIC
#     t.extend(
#         [
#             transforms.Resize((args.input_size,) * 2, interpolation=interp, antialias=True),
#             transforms.ToTensor(),
#             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
#         ]
#     )
#
#     return transforms.Compose(t)


# def get_image_fn(image):
#     image = np.stack((image,) * 3, axis=2)
#     image = np.round((image - image.min()) / (image.max() - image.min() + 1e-8) * 255)
#     image = image.astype(np.uint8)
#     image = Image.fromarray(image)
#     return image