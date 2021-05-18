"""
        issert False
Augmenters that apply to a group of augmentations, like selecting
an augmentation from a list, or applying all the augmentations in
a list sequentially

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.HorizontalFlip(),
                          va.VerticalFlip() ])

List of augmenters:
    * Sequential
    * OneOf
    * SomeOf
    * Sometimes

"""

import numpy as np
import PIL
import random
from typing import Any, Union
import hashlib

def _get_transform_hash(transform: Any) -> Union[int, None]:
    """
    Private function to get transform for a function.

    Args:
        transform: Augmentation object

    Return:
        hash of an object or None if the hash does not exist
    """
    if hasattr(transform, "hash"):
        transform_hash = transform.hash
    else:
        transform_hash = hashlib.sha256(
            str(type(transform)).encode('utf-8')
        ).hexdigest()
    return transform_hash



class Sequential(object):
    """
    Composes several augmentations together.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.

        random_order (bool): Whether to apply the augmentations in random order.
    """

    def __init__(self, transforms, random_order=False):
        self.transforms = transforms
        self.rand = random_order
        self.hash = None

    def __call__(self, clip):
        seq_hash = []
        if self.rand:
            rand_transforms = self.transforms[:]
            random.shuffle(rand_transforms)
            for t in rand_transforms:
                transform_hash = _get_transform_hash(t)
                if transform_hash is not None:
                    seq_hash.append(str(transform_hash))
                clip = t(clip)
        else:
            for t in self.transforms:
                transform_hash = _get_transform_hash(t)
                if transform_hash is not None:
                    seq_hash.append(str(transform_hash))
                clip = t(clip)
        self.hash = "_".join(seq_hash)
        return clip


class OneOf(object):
    """
    Selects one augmentation from a list.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.hash = None

    def __call__(self, clip):
        select = random.choice(self.transforms)
        self.hash = _get_transform_hash(select)
        clip = select(clip)
        return clip


class SomeOf(object):
    """
    Selects a given number of augmentation from a list.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations.

        N (int): The number of augmentations to select from the list.

        random_order (bool): Whether to apply the augmentations in random order.

    """

    def __init__(self, transforms, N, random_order=True):
        self.transforms = transforms
        self.rand = random_order
        if N > len(transforms):
            raise TypeError('The number of applied augmentors should be smaller than the given augmentation number')
        else:
            self.N = N
        self.hash = None

    def __call__(self, clip):
        seq_hash = []
        if self.rand:
            indices = [i for i in range(len(self.transforms))]
            selected_indices = [indices.pop(random.randrange(len(indices))) for _ in range(self.N)]
            selected_trans = [self.transforms[i] for i in selected_indices]
            for t in selected_trans:
                transform_hash = _get_transform_hash(t)
                if transform_hash is not None:
                    seq_hash.append(str(transform_hash))
                clip = t(clip)
        else:
            indices = [i for i in range(len(self.transforms))]
            selected_indices = [indices.pop(random.randrange(len(indices))) for _ in range(self.N)]
            selected_indices.sort()
            selected_trans = [self.transforms[i] for i in selected_indices]
            for t in selected_trans:
                transform_hash = _get_transform_hash(t)
                if transform_hash is not None:
                    seq_hash.append(str(transform_hash))
                clip = t(clip)
        self.hash = "_".join(seq_hash)
        return clip


class Sometimes(object):
    """
    Applies an augmentation with a given probability.

    Args:
        p (float): The probability to apply the augmentation.

        transform (an "Augmentor" object): The augmentation to apply.

    Example: Use this this transform as follows:
        sometimes = lambda aug: va.Sometimes(0.5, aug)
        sometimes(va.HorizontalFlip)
    """

    def __init__(self, p, transform):
        self.transform = transform
        self.transform_hash = _get_transform_hash(self.transform)
        if (p > 1.0) | (p < 0.0):
            raise TypeError('Expected p to be in [0.0 <= 1.0], ' +
                            'but got p = {0}'.format(p))
        else:
            self.p = p
        self.hash = None

    def __call__(self, clip):
        if random.random() < self.p:
            clip = self.transform(clip)
            self.hash = self.transform_hash
        else:
            self.hash = None
        return clip
