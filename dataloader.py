import keras
import keras_cv
import tensorflow as tf

from typing import Tuple

feature_desc = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_shape": tf.io.FixedLenFeature([3], tf.int64),
    "mask": tf.io.FixedLenFeature([], tf.string),
    "mask_shape": tf.io.FixedLenFeature([3], tf.int64),
}


def get_dataset(
    tfrecord_path: str,
    augmentations: bool = False,
    filter_non_zero_prob: float = 0.0,
    batch_size: int = 32,
    prefetch: bool = False,
    shuffle_size: int = 0,
    cache: bool = False,
) -> tf.data.Dataset:
    AUTOTUNE = tf.data.AUTOTUNE

    dset = tf.data.TFRecordDataset(tfrecord_path)
    dset = dset.map(parse_example, num_parallel_calls=AUTOTUNE)
    dset = dset.map(preprocessing, num_parallel_calls=AUTOTUNE)

    if augmentations:
        NUM_CLASSES = 2
        ROTATION_FACTOR = (-0.05, 0.05)
        augment = keras.Sequential(
            [
                keras_cv.layers.RandomFlip(),
                keras_cv.layers.RandomRotation(
                    factor=ROTATION_FACTOR,
                    segmentation_classes=NUM_CLASSES,
                ),
                # keras_cv.layers.RandAugment(
                #     value_range=(0, 1),
                #     geometric=False,
                # ),
            ]
        )

        def augmentation(
            image: tf.Tensor, mask: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            inputs_dict = {
                "images": image,
                "segmentation_masks": tf.cast(mask, dtype=tf.int64),
            }
            augmented_data = augment(inputs_dict, training=True)
            return augmented_data["images"], augmented_data["segmentation_masks"]

        dset = dset.map(augmentation, num_parallel_calls=AUTOTUNE)

    if filter_non_zero_prob > 0:
        rng = tf.random.Generator.from_seed(123, alg="philox")

        def filter_non_zero(image, mask):
            seed = rng.make_seeds(2)[0]
            if tf.math.count_nonzero(mask) == 0:
                return tf.random.stateless_uniform([], seed) > filter_non_zero_prob
            return True

        dset = dset.filter(filter_non_zero)
    dset = dset.map(lambda x, y: (x, tf.cast(y, tf.int32)), num_parallel_calls=AUTOTUNE)
    dset = dset.cache() if cache else dset
    dset = (
        dset.shuffle(shuffle_size, reshuffle_each_iteration=True)
        if shuffle_size
        else dset
    )
    dset = dset.batch(batch_size, num_parallel_calls=AUTOTUNE)
    dset = dset.prefetch(AUTOTUNE) if prefetch else dset
    return dset


def parse_example(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_desc)
    image = tf.io.parse_tensor(example["image"], tf.float32)
    image = tf.reshape(image, example["image_shape"])
    mask = tf.io.parse_tensor(example["mask"], tf.float32)
    mask = tf.reshape(mask, example["mask_shape"])
    return image, mask


def preprocessing(image, mask):
    image = image / 255.0
    return image, mask
