import torch
import numpy as np
from torchvision import transforms

from torchvision.transforms.v2 import ElasticTransform
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomResizedCrop


class Normalize:
    def __call__(self, img):
        return (img - img.mean()) / (img.std() + 1e-8)


class RescaleImage:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            # Convert to tensor
            image = torch.from_numpy(image)
        elif torch.is_tensor(image):
            pass
        else:
            raise TypeError

        # Rescale the tensor to [0, 1]
        min_val = image.reshape(image.shape[0], -1).min(dim=1)[0].reshape(-1, 1, 1)
        max_val = image.reshape(image.shape[0], -1).max(dim=1)[0].reshape(-1, 1, 1)
        return (image - min_val) / (max_val - min_val)


class GaussianNoise:
    def __init__(self, range=(0, 0.1), p=0.15):
        self.range = range
        self.p = p

    def __call__(self, img):
        if not np.random.uniform() < self.p:
            return img
        scale = np.random.uniform(self.range[0], self.range[1])
        return img + torch.randn((1, img.shape[1], img.shape[2])) * scale


class RandomBrightness:
    def __init__(self, range=(-0.5, 0.5), p=0.15):
        self.range = range
        self.p = p

    def __call__(self, img):
        if not np.random.uniform() < self.p:
            return img
        scale = np.random.uniform(self.range[0], self.range[1])
        return img + scale


class RandomContrast:
    def __init__(self, range=(0.75, 1.25), p=0.15):
        self.range = range
        self.p = p

    def __call__(self, img):
        if not np.random.uniform() < self.p:
            return img
        scale = np.random.uniform(self.range[0], self.range[1])
        mean_ = img.mean()
        min_ = img.min()
        max_ = img.max()
        img = torch.clamp((img - mean_) * scale + mean_, min_, max_)
        return img


class RandomGamma:
    def __init__(self, range=(0.5, 2), p=0.15):
        self.range = range
        self.p = p

    def __call__(self, img):
        if not np.random.uniform() < self.p:
            return img
        mean_ = img.mean()
        std_ = img.std()

        if np.random.uniform() < 0.5 and self.range[0] < 1:
            scale = np.random.uniform(self.range[0], 1)
        else:
            scale = np.random.uniform(max(1, self.range[0]), self.range[1])

        img = torch.pow((img - img.min()) / (img.max() - img.min() + 1e-8), scale)
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = img * std_ + mean_
        return img


class RandomElasticTransform:
    def __init__(self, alpha_range=(0, 200), sigma_range=(10, 13), p=0.15):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if not np.random.uniform() < self.p:
            return img
        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        img = ElasticTransform(alpha=alpha, sigma=sigma)(img)
        return img


class RandomRotationProb:
    def __init__(self, degree_range=(-30, 30), p=0.15):
        self.degree_range = degree_range
        self.p = p

    def __call__(self, img):
        if not np.random.uniform() < self.p:
            return img
        img = RandomRotation(degrees=self.degree_range)(img)
        return img

