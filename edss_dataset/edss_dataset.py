import logging
import os
from typing import Optional, List, Tuple, Literal, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from utils.constants import CLASS_THRESHOLDS

logging.basicConfig(level=logging.INFO)


def preprocess_image(
    image: Union[np.ndarray, EagerTensor], label: int, normalize: bool = True
):
    """
    Preprocess image by applying rescaling, normalization and augmentation.

    Args:
        - image: (np.ndarray, EagerTensor): the image to preprocess (expects grayscale image)
        - label: (int): the label of the image
        - normalize: (bool): whether to normalize the image
        - transformations: (Optional[List[Callable]]): the transformations to apply. If None,
            a set of default transformations will be applied.
    """
    if image.ndim != 2 and (image.ndim == 3 and image.shape[-1] != 1):
        raise ValueError(
            "Image must be grayscale. Expected shapes: (H, W) or (H, W, 1)."
        )

    if normalize:
        if isinstance(image, np.ndarray):
            image = tf.cast(image, tf.float32) / 255
        else:
            image = tf.cast(image, tf.float32)
            image = tf.divide(image, 255)

    return image, label


def get_dataset(
    data_dir: str,
    modality: Union[
        Literal["T1", "T2", "FLAIR"], List[Literal["T1", "T2", "FLAIR"]]
    ] = "T1",
    split: Literal["train", "val", "test"] = "train",
    task: Literal["binary", "multi-class"] = "binary",
    batch_size: int = 32,
    resize: Optional[Tuple[int, int]] = (256, 256),
    normalize: bool = True,
    include_augmented: bool = False,
) -> tf.data.Dataset:
    """
    Load and preprocess the dataset.
    The data will be loaded from `data_dir/modality/task/split`.
    Only `.png` files will be considered. Files must have the label specified in their filepath as '_label_.png'

    Args:
        - data_dir: directory to load the dataset from.
        - split: one of "train", "val", "test". Defaults to "train".
        - task: one of "binary", "multi-class". Defaults to "binary".
        - batch_size: batch size to use for training and preprocessing. Defaults to 32.
        - resize: (Tuple[int, int]): the rescaling factor
        - normalize: (bool): whether to normalize the image
        - include_augmented: (bool): whether to include augmented images

    """
    if isinstance(modality, str):
        modality = [modality]

    images = []
    labels = []

    task_labels = list(CLASS_THRESHOLDS[task].keys())

    for mod in modality:
        dataset_dir = os.path.join(data_dir, mod, task, split)

        logging.info(f"Loading dataset from {dataset_dir}")

        for filename in os.listdir(dataset_dir):
            if not filename.endswith(".png"):
                continue

            # skip augmented files if no augmentation
            if not include_augmented and "aug" in filename:
                continue

            image = tf.io.read_file(os.path.join(dataset_dir, filename))
            image = tf.io.decode_png(image, channels=1)

            if resize is not None:
                image = tf.image.resize(image, resize)

            label = filename.split("_")[-2]

            try:
                label = task_labels.index(label)
            except ValueError:
                logging.warning(
                    f"Label '{label}' not found in CLASS_THRESHOLDS for task '{task}'. Skipping file."
                )
                continue

            images.append(image)

            if task == "multi-class":
                label = tf.one_hot(label, depth=len(CLASS_THRESHOLDS[task].keys()))

            labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.batch(batch_size).map(
        lambda img, lbl: preprocess_image(img, lbl, normalize=normalize),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if split == "train":
        dataset = dataset.shuffle(buffer_size=len(dataset))

    logging.info(
        f"Dataset loaded from {dataset_dir}\n"
        "\t- Modality: {mod}\n"
        f"\t- Task: {task}\n"
        f"\t- Split: {split}\n"
        f"\t- Number of samples: {len(images)}\n"
        f"\t- Batch size: {batch_size}\n"
    )

    return dataset

