import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.backend as backend
from tensorflow.keras.utils import get_source_inputs

from functools import partial

from .layers import bottleneckblock, SepResBlock

CONV_INIT = "he_normal"

NORM_DICT = {"bn": layers.BatchNormalization, "gn": layers.GroupNormalization, "ln": layers.LayerNormalization}


def build_backbone(name, **kwargs):

    supported_backbones = {"resnext50": resnext50, "resnext": resnext, 'sepresnet':sepresnet}

    return supported_backbones.get(name.lower(), resnext50)(**kwargs)


def resnext(
    input_tensor=None,
    input_shape=None,
    classes=1000,
    dropout_rate=0,
    activation="gelu",
    stem_kernel_size=7,
    stem_conv_filters=64,
    stem_stride=2,
    stem_convs=1,
    resblocks=[3, 4, 6, 3],
    filters=[256, 512, 1024, 2048],
    groups=32,
    include_top=True,
    top_dense=[2048],
    se_block=True,
    se_ratio=4,
    name="resnext50",
    bottleneck_reduction=2,
    normalization="gn",
    normalization_kw={},
    preact=False,
    **kwargs,
):
    """Instantiates a custom ResNext architecture.

    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
        stem_kernel_size: size of stem convolutions (default 7 for classical resnet)
        stem_conv_filters: depths of the 1st stage conv layers at the bottom of the network.
        stem_stride: stride of each stem conv
        stem_convs: number of conv layer in the stem
        classes: optional number of classes to classify images
        resblocks: number  of repetitions of basic residual blocks in each stage after stage 1 (list of integers)
        bottleneck_reduction: reduction factor of the bottleneck architecture
        filters: list of output depths of each stage (list of int). Default: 2 for ResNext, 4 for classical Resnet
        groups: number of groups in convolution (cardinality)
        dropout_rate: dropout_rate after the Global pooling layer and between dense layers at the top of the network
        SE block: add a "squeeze and excite" block at the end of residual blocks (before adding shortcut branch)
        top_dense: list of depths of dense layers before the classification layer (default [])).

    # Returns
        A Keras model instance.

    """

    # print("Initializing ResNet architecture")

    NORM = NORM_DICT.get(normalization.lower(), NORM_DICT["ln"])
    NORM = partial(NORM, **normalization_kw)

    # Init block type

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = img_input

    # STEM
    for i in range(stem_convs):

        x = layers.Conv2D(
            stem_conv_filters,
            stem_kernel_size,
            strides=stem_stride,
            padding="same",
            kernel_initializer="he_normal",
            name=f"stem_conv_{i}",
            use_bias=False,
        )(x)
        if i < stem_convs - 1:
            x = NORM(name=f"stem_norm_{i}")(x)
            x = layers.Activation(activation, name=f"stem_act_{i}")(x)

    # Only if no preactiavtion in base blocks
    if not preact:
        x = NORM(name=f"stem_norm_{i}")(x)
        x = layers.Activation(activation, name=f"stem_act_{i}")(x)

    # downsampling at the beginning of each stage (when block == 0)
    for stage, blocks in enumerate(resblocks):
        for block in range(blocks):
            strides = 1
            if block == 0:
                # In original ResNet, the downsampling is done by MaxPooling before the first residual block
                # here it is done in the first conv of the first block of the stage
                filters_in = filters[stage - 1] if stage > 0 else stem_conv_filters
                # if stage > 0:
                strides = 2
            else:
                filters_in = filters[stage]

            x = bottleneckblock(
                filters_in=filters_in,
                filters_out=filters[stage],
                activation=activation,
                strides=strides,
                kernel_size=3,
                se_ratio=se_ratio,
                se_block=se_block,
                groups=groups,
                bottleneck_reduction=bottleneck_reduction,
                normalization=normalization,
                normalization_kw=normalization_kw,
                preact=preact,
                name="stage{}_block{}".format(stage + 1, block + 1),
            )(x)

    if include_top:

        if preact:
            x = NORM(name="post_norm")(x)
            x = layers.Activation(activation, name="post_act")(x)

        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

        for i, depth in enumerate(top_dense):
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name="head_dropout_{}".format(i + 1))(x)
            x = layers.Dense(depth, activation=activation, name="head_fc_{}".format(i + 1))(x)

        # Classification layer
        x = layers.Dense(classes, activation="softmax", name="classification")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name=name)

    return model


def sepresnet(
    input_tensor=None,
    input_shape=None,
    classes=1000,
    dropout_rate=0,
    activation="gelu",
    stem_kernel_size=7,
    stem_conv_filters=64,
    stem_stride=2,
    stem_convs=1,
    resblocks=[3, 4, 6, 3],
    filters=[256, 512, 1024, 2048],
    groups=32,
    include_top=True,
    top_dense=[2048],
    se_block=True,
    se_ratio=4,
    downsampling_shortcut='none',
    name="sepresnet",
    normalization="gn",
    normalization_kw={},
    **kwargs,
):
    """Instantiates a custom ResNext architecture.

    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
        stem_kernel_size: size of stem convolutions (default 7 for classical resnet)
        stem_conv_filters: depths of the 1st stage conv layers at the bottom of the network.
        stem_stride: stride of each stem conv
        stem_convs: number of conv layer in the stem
        classes: optional number of classes to classify images
        resblocks: number  of repetitions of basic residual blocks in each stage after stage 1 (list of integers)
        bottleneck_reduction: reduction factor of the bottleneck architecture
        filters: list of output depths of each stage (list of int). Default: 2 for ResNext, 4 for classical Resnet
        groups: number of groups in convolution (cardinality)
        dropout_rate: dropout_rate after the Global pooling layer and between dense layers at the top of the network
        SE block: add a "squeeze and excite" block at the end of residual blocks (before adding shortcut branch)
        top_dense: list of depths of dense layers before the classification layer (default [])).

    # Returns
        A Keras model instance.

    """

    # print("Initializing ResNet architecture")

    NORM = NORM_DICT.get(normalization.lower(), NORM_DICT["ln"])
    NORM = partial(NORM, **normalization_kw)

    # Init block type

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = img_input

    # STEM
    for i in range(stem_convs):

        x = layers.Conv2D(
            stem_conv_filters,
            stem_kernel_size,
            strides=stem_stride,
            padding="same",
            kernel_initializer="he_normal",
            name=f"stem_conv_{i}",
            use_bias=False,
        )(x)
        if i < stem_convs - 1:
            x = NORM(name=f"stem_norm_{i}")(x)
            x = layers.Activation(activation, name=f"stem_act_{i}")(x)


    # downsampling at the beginning of each stage (when block == 0)
    for stage, blocks in enumerate(resblocks):
        for block in range(blocks):
            strides = 1
            if block == 0:
                # In original ResNet, the downsampling is done by MaxPooling before the first residual block
                filters_in = filters[stage - 1] if stage > 0 else stem_conv_filters
                if stage > 0:
                    strides = 2
            else:
                filters_in = filters[stage]

            x = SepResBlock(
                filters_in=filters_in,
                filters_out=filters[stage],
                activation=activation,
                strides=strides,
                kernel_size=3,
                se_ratio=se_ratio,
                se_block=se_block,
                groups=groups,
                downsampling_method=downsampling_shortcut,
                normalization=normalization,
                normalization_kw=normalization_kw,
                name="stage{}_block{}_".format(stage + 1, block + 1),
            )(x)

    if include_top:

        x = NORM(name="post_norm")(x)
        x = layers.Activation(activation, name="post_act")(x)

        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

        for i, depth in enumerate(top_dense):
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name="head_dropout_{}".format(i + 1))(x)
            x = layers.Dense(depth, activation=activation, name="head_fc_{}".format(i + 1))(x)

        # Classification layer
        x = layers.Dense(classes, activation="softmax", name="classification")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name=name)

    return model


def resnext50(**kwargs):
    """classical ResNeXt50"""
    defaults = {
        "resblocks": [3, 4, 6, 3],
        "filters": [256, 512, 1024, 2048],
        "first_kernel_size": 7,
        "stem_conv_filters": 64,
        "groups": 32,
        "SE_block": True,
        "SE_ratio": 4,
        "bottleneck_reduction": 2,
        "top_dense": [2048],
        "include_top": False,
        "name": "resnext50",
        "normalization": "gn",
    }

    defaults.update(kwargs)

    return resnext(**defaults)
