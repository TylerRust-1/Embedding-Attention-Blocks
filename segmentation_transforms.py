import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from torch import nn


class Sequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.transforms = nn.ModuleList(args)

    def forward(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class Identity(nn.Module):
    def forward(self, image, label):
        return image, label


def get_random(value, distribution):
    if distribution == 'normal':
        value = np.random.normal(0, value)
    elif distribution == 'uniform':
        value = random.randint(-value, value)
    elif distribution == 'positive-negative':
        value = value if random.random() < 0.5 else -value
    elif distribution == 'constant':
        pass
    else:
        raise ValueError(distribution)
    return value


class Rotate(nn.Module):
    def __init__(self, angle, fill_value, distribution='constant', interpolation=Image.BICUBIC):
        super().__init__()
        self.angle = angle
        self.fill_value = fill_value
        self.distribution = distribution
        self.interpolation = interpolation

    def forward(self, image, label):
        angle = get_random(self.angle, self.distribution)
        image = image.rotate(angle, self.interpolation, expand=True)
        label = label.rotate(angle, Image.NEAREST, expand=True, fillcolor=self.fill_value)
        return image, label


class ShearX(nn.Module):
    def __init__(self, factor, fill_value, distribution='constant', interpolation=Image.BICUBIC):
        super().__init__()
        self.factor = factor
        self.fill_value = fill_value
        self.distribution = distribution
        self.interpolation = interpolation

    def forward(self, image, label):
        factor = get_random(self.factor, self.distribution)
        image = image.transform(image.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), self.interpolation)
        label = label.transform(label.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), Image.NEAREST, fillcolor=self.fill_value)
        return image, label


class ShearY(nn.Module):
    def __init__(self, factor, fill_value, distribution='constant', interpolation=Image.BICUBIC):
        super().__init__()
        self.factor = factor
        self.fill_value = fill_value
        self.distribution = distribution
        self.interpolation = interpolation

    def forward(self, image, label):
        factor = get_random(self.factor, self.distribution)
        image = image.transform(image.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), self.interpolation)
        label = label.transform(label.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), Image.NEAREST, fillcolor=self.fill_value)
        return image, label


class TranslateX(nn.Module):
    def __init__(self, factor, fill_value, distribution='constant', interpolation=Image.BICUBIC):
        super().__init__()
        self.factor = factor
        self.fill_value = fill_value
        self.distribution = distribution
        self.interpolation = interpolation

    def forward(self, image, label):
        factor = get_random(self.factor, self.distribution)
        image = image.transform(image.size, Image.AFFINE, (1, 0, factor * image.width, 0, 1, 0), self.interpolation)
        label = label.transform(label.size, Image.AFFINE, (1, 0, factor * image.width, 0, 1, 0), Image.NEAREST, fillcolor=self.fill_value)
        return image, label


class TranslateY(nn.Module):
    def __init__(self, factor, fill_value, distribution='constant', interpolation=Image.BICUBIC):
        super().__init__()
        self.factor = factor
        self.fill_value = fill_value
        self.distribution = distribution
        self.interpolation = interpolation

    def forward(self, image, label):
        factor = get_random(self.factor, self.distribution)
        image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, factor * image.height), self.interpolation)
        label = label.transform(label.size, Image.AFFINE, (1, 0, 0, 0, 1, factor * image.height), Image.NEAREST, fillcolor=self.fill_value)
        return image, label


class Resize(nn.Module):
    def __init__(self, scale=None, shorter=None, longer=None, interpolation=Image.BICUBIC):
        super().__init__()
        self.scale = scale
        self.shorter = shorter
        self.longer = longer
        self.interpolation = interpolation

    def forward(self, image, label):
        if self.scale:
            if len(self.scale) == 2:
                scale = random.uniform(*self.scale)
            else:
                scale = random.choice(self.scale)
            size = (round(scale * image.width), round(scale * image.height))
        else:
            size = self.shorter or self.longer
            if isinstance(size, int):
                pass
            elif len(size) == 2:
                size = random.randint(*size)
            else:
                size = random.choice(size)

            w, h = image.size
            if (self.shorter and w < h) or (self.longer and w > h):
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            size = (ow, oh)
        image = image.resize(size, self.interpolation)
        label = label.resize(size, Image.NEAREST)
        return image, label


class Pad(nn.Module):
    def __init__(self, output_size, fill_value, random=True):
        super().__init__()
        self.output_size = output_size
        self.fill_value = fill_value
        self.random = random

    def forward(self, image, label):
        w, h = image.size
        tw, th = self.output_size

        if w >= tw and h >= th:
            return image, label

        left, top, right, bottom = 0, 0, 0, 0
        if w < tw:
            padding = tw - w
            if self.random:
                left = round(padding * random.random())
            else:
                left = padding // 2
            right = padding - left
        if h < th:
            padding = th - h
            if self.random:
                top = round(padding * random.random())
            else:
                top = padding // 2
            bottom = padding - top

        image = ImageOps.expand(image, (left, top, right, bottom))
        label = ImageOps.expand(label, (left, top, right, bottom), self.fill_value)
        return image, label


class Crop(nn.Module):
    def __init__(self, output_size, random=True):
        super().__init__()
        self.output_size = output_size
        self.random = random

    def forward(self, image, label):
        w, h = image.size
        tw, th = self.output_size
        if w <= tw and h <= th:
            return image, label
        if self.random:
            x = random.randint(0, w - tw)
            y = random.randint(0, h - th)
        else:
            x = (w - tw) // 2
            y = (h - th) // 2
        image = image.crop((x, y, x + tw, y + th))
        label = label.crop((x, y, x + tw, y + th))
        return image, label


class Flip(nn.Module):
    def __init__(self, p=0.5, random=True):
        super().__init__()
        self.p = p
        self.random = random

    def forward(self, image, label):
        if not self.random or random.random() < self.p:
            image = ImageOps.mirror(image)
            label = ImageOps.mirror(label)
        return image, label


class VerticalFlip(nn.Module):
    def __init__(self, p=0.5, random=True):
        super().__init__()
        self.p = p
        self.random = random

    def forward(self, image, label):
        if not self.random or random.random() < self.p:
            image = ImageOps.flip(image)
            label = ImageOps.flip(label)
        return image, label


class AutoContrast(nn.Module):
    def forward(self, image, label):
        image = ImageOps.autocontrast(image)
        return image, label


class Sharpness(nn.Module):
    def __init__(self, factor, distribution='constant'):
        super().__init__()
        self.factor = factor
        self.distribution = distribution

    def forward(self, image, label):
        factor = 1 + get_random(self.factor, self.distribution)
        image = ImageEnhance.Sharpness(image).enhance(factor)
        return image, label


class Contrast(nn.Module):
    def __init__(self, factor, distribution='constant'):
        super().__init__()
        self.factor = factor
        self.distribution = distribution

    def forward(self, image, label):
        factor = 1 + get_random(self.factor, self.distribution)
        image = ImageEnhance.Contrast(image).enhance(factor)
        return image, label


class Color(nn.Module):
    def __init__(self, factor, distribution='constant'):
        super().__init__()
        self.factor = factor
        self.distribution = distribution

    def forward(self, image, label):
        factor = 1 + get_random(self.factor, self.distribution)
        image = ImageEnhance.Color(image).enhance(factor)
        return image, label


class Brightness(nn.Module):
    def __init__(self, factor, distribution='constant'):
        super().__init__()
        self.factor = factor
        self.distribution = distribution

    def forward(self, image, label):
        factor = 1 + get_random(self.factor, self.distribution)
        image = ImageEnhance.Brightness(image).enhance(factor)
        return image, label


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = np.array(mean).reshape((1, 1, 3))
        self.std = np.array(std).reshape((1, 1, 3))

    def forward(self, image, label):
        image = (np.array(image, dtype=np.float32) - self.mean) / self.std
        return image, label


class MapLabel(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def forward(self, image, label):
        label = np.array(label)
        label = self.mapping[label]
        label = Image.fromarray(label)
        return image, label


class ToTensor(nn.Module):
    def forward(self, image, label):
        image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1))
        label = torch.from_numpy(np.array(label, dtype=np.long))
        return image, label


class RandAugment(nn.Module):
    def __init__(self, n, m, fill_value):
        super().__init__()
        self.n = n
        in_range = lambda minval, maxval: (m / 30) * (maxval - minval) + minval
        self.transforms = [
            Rotate(in_range(0, 30), fill_value),
            ShearX(in_range(0, 0.3), fill_value),
            ShearY(in_range(0, 0.3), fill_value),
            TranslateX(in_range(0, 0.5), fill_value),
            TranslateY(in_range(0, 0.5), fill_value),
            AutoContrast(),
            Sharpness(in_range(0.1, 1.9)),
            Identity()
        ]

    def forward(self, image, label):
        transforms = random.choices(self.transforms, k=self.n)
        for transform in transforms:
            image, label = transform(image, label)
        return image, label
