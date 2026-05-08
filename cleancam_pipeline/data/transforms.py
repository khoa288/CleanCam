"""Image transforms for training and evaluation."""

from typing import Tuple

from torchvision import transforms

from cleancam_pipeline.core.constants import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build training and evaluation transforms.

    Args:
        image_size: Target image size

    Returns:
        Tuple of (train_transform, eval_transform)
    """
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_tf, eval_tf
