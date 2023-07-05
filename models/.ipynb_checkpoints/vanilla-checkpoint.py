import tensorflow as tf
from tensorflow.keras import layers

from typing import List, Optional, Tuple, Literal, Any, Union

from .conv_layers import Prediction, DeepSupervision

IntOrIntTuple = Union[int, Tuple[int, int, int]]


def UNet(
    encoder: tf.keras.Model,
    skip_names: List[str],
    num_classes: int,
    out_activation: Literal["sigmoid", "softmax"] = "sigmoid",
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Optional[Any] = layers.BatchNormalization,
    upsample_layer: Any = layers.UpSampling2D,
    attention_layer: Optional[Any] = None,
    dropout_rate: float = 0.0,
    name: str = "unet_decoder",
) -> tf.keras.Model:

    # Expects the skip names to be ordered from earlier to deeper levels.
    x = encoder.output

    skip_names = reversed(skip_names)
    for i, skip_name in enumerate(skip_names):
        x_skip = encoder.get_layer(skip_name).output
        x = UpBlock(
            x_skip.shape[-1],
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            upsample_layer=upsample_layer,
            attention_layer=attention_layer,
            dropout_rate=dropout_rate,
            name=name + f"_up_{i}",
        )([x, x_skip])
    x = Prediction(num_classes, out_activation, name=name + "_last")(x)

    return tf.keras.Model(inputs=encoder.inputs, outputs=x, name=name)


def UNetEncoder(
    input_shape: Tuple[int, int, int],
    filters_per_level: List[int],
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Optional[Any] = layers.BatchNormalization,
    pooling_layer: Any = layers.MaxPooling2D,
    dropout_rate: float = 0.0,
    name="unet_encoder",
) -> tf.keras.Model:

    inputs = tf.keras.Input(input_shape, name=name + "_inputs")
    x = Block(
        filters_per_level[0],
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        norm_layer=norm_layer,
        dropout_rate=dropout_rate,
        name=name + "_first_block",
    )(inputs)

    for i, filters in enumerate(filters_per_level[1:-1]):
        x = DownBlock(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            pooling_layer=pooling_layer,
            dropout_rate=dropout_rate,
            name=name + f"_down_{i}",
        )(x)

    x = Block(
        filters_per_level[-1],
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        norm_layer=norm_layer,
        dropout_rate=dropout_rate,
        name=name + "_bottleneck",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


# ---------------------------------------------------------------------------------
# Main blocks.


def UpBlock(
    filters: int,
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Any = layers.BatchNormalization,
    upsample_layer: Any = layers.UpSampling2D,
    attention_layer: Optional[Any] = None,
    dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> tf.Tensor:
    def apply(inputs):

        x, x_skip = inputs

        # Upsample and block to match our paper implementation.
        x = upsample_layer(name=name + "_up")(x)
        x = Block(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            name=name + "_upconv",
        )(x)

        # Filter skip connection values with attention.
        if not attention_layer is None:
            if "Self" in attention_layer.__name__:
                x_skip = attention_layer(filters, name=name + "_sa")(x_skip)
            elif "Cross" in attention_layer.__name__:
                x_skip = attention_layer(filters, name=name + "_ca")([x, x_skip])

        x = layers.Concatenate(name=name + "_concat")([x, x_skip])

        x = Block(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            name=name,
        )(x)

        return x

    return apply


def DownBlock(
    filters: int,
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Any = layers.BatchNormalization,
    pooling_layer: Any = layers.MaxPooling2D,
    dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> tf.Tensor:
    def apply(inputs):
        x = pooling_layer(pool_size=2, name=name + "_pool")(inputs)
        x = Block(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            name=name,
        )(x)
        return x

    return apply


def Block(
    filters: int,
    kernel_size: IntOrIntTuple = 3,
    strides: IntOrIntTuple = 1,
    dilation_rate: IntOrIntTuple = 1,
    padding: str = "same",
    norm_layer: Any = layers.BatchNormalization,
    dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> tf.Tensor:
    def apply(inputs):
        x = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            name=name + "_conv_1",
        )(inputs)
        if not norm_layer is None:
            x = norm_layer(name=name + "_bn_1")(x)
        x = layers.Activation("relu", name=name + "_relu_1")(x)

        x = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            name=name + "_conv_2",
        )(x)
        if not norm_layer is None:
            x = norm_layer(name=name + "_bn_2")(x)
        x = layers.Activation("relu", name=name + "_relu_2")(x)

        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=name + "_drop")(x)
        return x

    return apply


# ---------------------------------------------------------------------------------
# Model utils.


def add_deep_supervision_to_unet(
    unet: tf.keras.Model,
    hidden_layer_names: List[str],
    resize_features: bool = True,
) -> tf.keras.Model:

    # Get the activation layer at the end of the U-Net.
    last_layer = unet.layers[-1]
    out_activation = last_layer.activation
    last_output = last_layer.output
    num_classes = last_output.shape[-1]

    outputs = []
    for i, hidden_layer_name in enumerate(hidden_layer_names):
        hidden_layer_output = unet.get_layer(hidden_layer_name).output
        x = DeepSupervision(
            num_classes,
            out_activation,
            resize_features=resize_features,
            target_shape=last_output.shape[1],
            name=unet.name + f"_deep_supervision_{i}",
        )(hidden_layer_output)
        outputs.append(x)

    outputs.append(last_output)
    model = tf.keras.Model(inputs=unet.inputs, outputs=outputs, name=unet.name)
    return model


def add_head_to_encoder(
    encoder: tf.keras.Model,
    num_classes: int,
    out_activation: Literal["sigmoid", "softmax"],
) -> tf.keras.Model:
    x = encoder.output
    x = layers.GlobalAveragePooling2D(name=encoder.name + "_global_pooling")(x)
    x = layers.Dense(num_classes, activation=None, name=encoder.name + "_logits")(x)
    x = layers.Activation(out_activation, name=encoder.name + "_probs")(x)

    return tf.keras.Model(inputs=encoder.inputs, outputs=x, name=encoder.name)


def remove_head_from_encoder(encoder: tf.keras.Model) -> tf.keras.Model:
    output = encoder.get_layer("unet_encoder_bottleneck_relu_2").output
    return tf.keras.Model(inputs=encoder.inputs, outputs=output, name=encoder.name)


def get_skip_names_from_encoder(encoder: tf.keras.Model) -> List[str]:
    last_layer = encoder.layers[-1]
    last_layer_output = last_layer.output
    output_shape = last_layer_output.shape
    has_dropout = any(layer.name.endswith("_drop") for layer in encoder.layers)
    suffix = "drop" if has_dropout else "relu_2"

    return [
        layer.name
        for layer in encoder.layers
        if (
            layer.name.endswith(suffix)
            and not "bottleneck" in layer.name
            and layer.output.shape[1:-1] != output_shape[1:-1]
        )
    ]


def get_output_names_for_deep_supervision(unet: tf.keras.Model) -> List[str]:
    last_layer = unet.layers[-1]
    last_layer_output = last_layer.output
    output_shape = last_layer_output.shape
    has_dropout = any(layer.name.endswith("_drop") for layer in unet.layers)
    suffix = "drop" if has_dropout else "relu_2"

    return [
        layer.name
        for layer in unet.layers
        if (
            layer.name.endswith(suffix)
            and not "encoder" in layer.name
            and not "bottleneck" in layer.name
            and layer.output.shape[1:-1] != output_shape[1:-1]
        )
    ]
