# coding: utf-8

import tensorflow as tf
from tensorflow.keras.regularizers import l2

import numpy as np
import os
from functools import partial
from sklearn.metrics import auc
import tensorflow.keras.layers as layers

from . import utils
from .backbone import build_backbone
from .layers import convblock, pad_with_coord, points_nms
from .loss import compute_image_loss
from .metrics import BinaryRecall, BinaryPrecision


CONV_INIT = "he_normal"
PRE_NMS = 1024

NORM_DICT = {"bn": layers.BatchNormalization, "gn": layers.GroupNormalization, "ln": layers.LayerNormalization}


def ISMENet_head(
    ncls,
    filters_in,
    conv_filters=256,
    kernel_filters=256,
    head_layers=4,
    cls_factor_layers=2,
    activation="gelu",
    name="ISMENet_head",
    normalization="gn",
    normalization_kw={"groups": 32},
    block_params=None,
    **kwargs,
):
    """ISMENet Shared head (class, kernel) + class factor
    For each level i of the FPN, the spatial extent is Si x Si
    the depth is nclass, kernel_filters and 1 for class, kernel and class_factor outputs respectively
    (kernel_filters must be equal to ks**2 * k_depth)
    """

    NORM = NORM_DICT.get(normalization.lower(), NORM_DICT["gn"])
    NORM = partial(NORM, **normalization_kw)

    if name is None:
        name = "ISMENet_head"

    input_tensor = tf.keras.Input(shape=(None, None, filters_in), name=name + "_input")

    # ct_head = input_tensor
    # No preactivation if groupnorm, because depth of input feature map with coord is not a power of 2
    if block_params is None:
        block_params = {
            "preact": False,
            "groups": 1,
            "activation": activation,
            "normalization": normalization,
            "normalization_kw": normalization_kw,
        }

    class_head = input_tensor[..., :-2]  # no need for coordinates in class head
    class_factor_head = input_tensor[..., :-2]
    kernel_head = input_tensor

    for i in range(head_layers):
        if i == 0:
            conv_filters_in = filters_in - 2
        else:
            conv_filters_in = conv_filters

        class_head = convblock(
            filters_in=conv_filters_in,
            filters_out=conv_filters,
            kernel_initializer=CONV_INIT,
            name=name + "_class_{}_".format(i + 1),
            **block_params,
        )(class_head)

    # class_factor_head = class_head

    for i in range(cls_factor_layers):
        if i == 0:
            conv_filters_in = filters_in - 2
        else:
            conv_filters_in = conv_filters

        class_factor_head = convblock(
            filters_in=conv_filters,
            filters_out=conv_filters,
            kernel_initializer=CONV_INIT,
            name=name + "_class_factor_{}_".format(i + 1),
            **block_params,
        )(class_factor_head)

    class_factor_head = NORM(name=name + "_class_factor_final_norm")(class_factor_head)
    class_factor_head = layers.Activation(activation, name=name + "_class_factor_final_act")(class_factor_head)

    class_head = NORM(name=name + "_class_final_norm")(class_head)
    class_head = layers.Activation(activation, name=name + "_class_final_act")(class_head)

    class_head = layers.Conv2D(
        filters=ncls,
        kernel_size=3,
        strides=1,
        padding="same",
        name=name + "_class_logits",
        kernel_initializer=CONV_INIT,
    )(class_head)

    class_factor_head = layers.Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding="same",
        name=name + "_class_factor",
        kernel_initializer=CONV_INIT,
    )(class_factor_head)

    for i in range(head_layers):
        if i == 0:
            conv_filters_in = filters_in
        else:
            conv_filters_in = conv_filters

        kernel_head = convblock(
            filters_in=conv_filters_in,
            filters_out=conv_filters,
            name=name + "_kernel_{}_".format(i + 1),
            **block_params,
        )(kernel_head)

    kernel_head = NORM(name=name + "_kernel_final_norm")(kernel_head)
    kernel_head = layers.Activation(activation, name=name + "_kernel_final_act")(kernel_head)
    kernel_head = layers.Conv2D(
        filters=kernel_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        name=name + "_pred_kernels",
        kernel_initializer=CONV_INIT,
    )(kernel_head)

    return tf.keras.Model(inputs=input_tensor, outputs=[class_head, class_factor_head, kernel_head], name=name)


def ISMENet_mask_head(
    fpn_features,
    nconv=2,
    in_ch=258,
    mid_ch=128,
    geom_ch=128,
    out_ch=256,
    output_level=0,
    activation="gelu",
    normalization="gn",
    normalization_kw={"groups": 32},
):
    """Compute the unified mask representation used in ISMENet and the geometrical feature maps for mass estimation
    fpn_features: list of feature maps from pyramid network P2 to Pn
    Resize, 3x3 conv and add all feature maps
    in_ch: input depth of feature maps
    mid_ch: inner convolution depth
    out_ch: output feature map depth
    output_level: level at which the FPN levels are resized
    """

    outputs = []
    for level, fpn_feature in enumerate(fpn_features):
        if level == len(fpn_features) - 1:
            # Add coordinates to the highest level
            feature = pad_with_coord(fpn_feature)
            channels = in_ch + 2
        else:
            feature = fpn_feature
            channels = in_ch

        # upsampling
        for i in range(level - output_level):
            if i > 0:
                channels = mid_ch

            feature = layers.UpSampling2D(interpolation="bilinear")(feature)
            feature = convblock(
                filters_in=channels,
                filters_out=mid_ch,
                activation=activation,
                normalization=normalization,
                normalization_kw=normalization_kw,
                preact=False,
                name="mask_head_lvl{}_up{}".format(level + 1, i + 1),
            )(feature)

        # downsampling
        for i in range(output_level - level):
            if i > 0:
                channels = mid_ch
            feature = convblock(
                filters_in=channels,
                filters_out=mid_ch,
                activation=activation,
                strides=2,
                normalization=normalization,
                normalization_kw=normalization_kw,
                preact=False,
                name="mask_head_lvl{}_down{}".format(level + 1, i + 1),
            )(feature)

        # 1x1 conv if not down/up
        if level == output_level:
            feature = convblock(
                filters_in=channels,
                filters_out=mid_ch,
                kernel_size=1,
                activation=activation,
                normalization=normalization,
                normalization_kw=normalization_kw,
                preact=False,
                name=f"mask_head_lvl{level + 1}_conv",
            )(feature)

        # print(level, tf.shape(feature))
        outputs.append(feature)

    seg_outputs = layers.Add()(outputs)

    geom_feats = seg_outputs

    for i in range(nconv):
        channels = mid_ch if i == 0 else geom_ch
        geom_feats = convblock(
            filters_in=channels,
            filters_out=geom_ch,
            kernel_size=3,
            activation=activation,
            normalization=normalization,
            normalization_kw=normalization_kw,
            preact=False,
            name=f"geom_feature_{i+1}_",
        )(geom_feats)

    # Last layer with linear activation
    geom_feats = layers.Conv2D(1, kernel_size=1, use_bias=True, strides=1, name="geom_feature_map")(geom_feats)
    # geom_feats = layers.Activation("relu", name="geom_feature_map_act")(geom_feats)

    seg_outputs = convblock(
        filters_in=mid_ch,
        filters_out=out_ch,
        kernel_size=1,
        activation=activation,
        normalization=normalization,
        normalization_kw=normalization_kw,
        preact=False,
        name="mask_head_output",
    )(seg_outputs)

    return seg_outputs, geom_feats


def ISMENet(config):
    """Create the ISMENet model using the given config object"""

    if config.load_backbone:
        backbone = tf.keras.models.load_model(config.backbone)
    else:
        backbone = build_backbone(
            config.backbone,
            **config.backbone_params,
        )

    FPN_inputs = {}
    # use get_output_at(0) instead of .output to avoid Graph disconnected error... does not work in tf2.16 and keras 3 !!
    # We should use output attribute in keras 3
    for lvl, layer in config.connection_layers.items():
        FPN_inputs[lvl] = backbone.get_layer(layer).get_output_at(0)
        # FPN_inputs[lvl] = backbone.get_layer(layer).output

    fpn_features = FPN(FPN_inputs, pyramid_filters=config.FPN_filters, extra_layers=config.extra_FPN_layers)

    head_model = ISMENet_head(
        ncls=config.ncls,
        filters_in=config.FPN_filters + 2,
        head_layers=config.head_layers,
        conv_filters=config.head_filters,
        kernel_filters=config.mask_output_filters * config.kernel_size**2,
        activation=config.activation,
        normalization=config.normalization,
        normalization_kw=config.normalization_kw,
    )

    maxdim = max(config.imshape)
    H = config.imshape[0]
    W = config.imshape[1]

    feat_kernel_list, feat_cls_list, feat_cls_factor_list = [], [], []
    for level, (feature, grid_size) in enumerate(zip(fpn_features, config.grid_sizes)):
        if level == 0:
            # add maxpool for P2 as in fastetimator implementation https://github.com/fastestimator
            feature = layers.MaxPool2D()(feature)
        feature = pad_with_coord(feature)
        feature = tf.image.resize(feature, size=(grid_size * H // maxdim, grid_size * W // maxdim))
        feat_cls, feat_cls_factor, feat_kernel = head_model(feature)
        feat_cls_list.append(feat_cls)
        feat_cls_factor_list.append(feat_cls_factor)
        feat_kernel_list.append(feat_kernel)

    # to build the mask output, we resize and add all FPN levels, except the extra levels > P5
    mask_output, geom_feats = ISMENet_mask_head(
        fpn_features[: -config.extra_FPN_layers],
        nconv=config.geom_feat_convs,
        in_ch=config.FPN_filters,
        mid_ch=config.mask_mid_filters,
        geom_ch=config.geom_feats_filters,
        out_ch=config.mask_output_filters,
        output_level=config.mask_output_level,
        activation=config.activation,
        normalization=config.normalization,
        normalization_kw=config.normalization_kw,
    )

    model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[mask_output, geom_feats, feat_cls_list, feat_cls_factor_list, feat_kernel_list],
        name=config.model_name,
    )
    return model


def FPN(connection_layers, pyramid_filters=256, name="", extra_layers=0, interpolation="bilinear", weight_decay=0):
    """Feature Pyramid Network (original implementation without activation and normalization)
    connection_layers: dict{"C2":layernameX,"C3":layernameY... to C5} dict of keras layers
    if C2 is not provided, P2 layer is not built.
    activation: activation function default 'relu'
    name: base name of the fpn
    interpolation: type of interpolation when upscaling feature maps (default: "bilinear")
    """

    if name is None:
        name = ""

    try:
        C5 = connection_layers["C5"]
        C4 = connection_layers["C4"]
        C3 = connection_layers["C3"]

    except Exception as e:
        print("Error when building FPN: can't get backbone network connection layers")
        print(e, e.args)

    C2 = connection_layers.get("C2", None)

    outputs = []

    P5 = layers.Conv2D(
        pyramid_filters,
        kernel_size=1,
        padding="same",
        use_bias=True,
        strides=1,
        kernel_initializer=CONV_INIT,
        kernel_regularizer=l2(weight_decay),
        name=name + "FPN_C5P5",
    )(C5)

    P5up = layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=name + "FPN_P5upsampled")(P5)

    C4P4 = layers.Conv2D(
        pyramid_filters,
        kernel_size=1,
        padding="same",
        use_bias=True,
        strides=1,
        kernel_initializer=CONV_INIT,
        kernel_regularizer=l2(weight_decay),
        name=name + "FPN_C4P4",
    )(C4)

    P4 = layers.Add(name=name + "FPN_P4add")([P5up, C4P4])

    P4up = layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=name + "FPN_P4upsampled")(P4)

    C3P3 = layers.Conv2D(
        pyramid_filters,
        kernel_size=1,
        padding="same",
        use_bias=True,
        strides=1,
        kernel_initializer=CONV_INIT,
        kernel_regularizer=l2(weight_decay),
        name=name + "FPN_C3P3",
    )(C3)
    P3 = layers.Add(name=name + "FPN_P3add")([P4up, C3P3])

    if C2 is not None:
        P3up = layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=name + "FPN_P3upsampled")(P3)
        P3up = layers.Conv2D(
            pyramid_filters,
            kernel_size=1,
            padding="same",
            use_bias=True,
            strides=1,
            kernel_initializer=CONV_INIT,
            kernel_regularizer=l2(weight_decay),
            name=name + "FPN_P3upconv",
        )(P3up)
        C2P2 = layers.Conv2D(
            pyramid_filters,
            kernel_size=1,
            padding="same",
            use_bias=True,
            strides=1,
            kernel_initializer=CONV_INIT,
            kernel_regularizer=l2(weight_decay),
            name=name + "FPN_C2P2",
        )(C2)
        P2 = layers.Add(name=name + "FPN_P2add")([P3up, C2P2])

        outputs.append(
            layers.Conv2D(
                pyramid_filters,
                kernel_size=3,
                padding="same",
                use_bias=True,
                strides=1,
                kernel_initializer=CONV_INIT,
                kernel_regularizer=l2(weight_decay),
                name=name + "FPN_P2",
            )(P2)
        )

    outputs.append(
        layers.Conv2D(
            pyramid_filters,
            kernel_size=3,
            padding="same",
            use_bias=True,
            strides=1,
            kernel_initializer=CONV_INIT,
            kernel_regularizer=l2(weight_decay),
            name=name + "FPN_P3",
        )(P3)
    )

    outputs.append(
        layers.Conv2D(
            pyramid_filters,
            kernel_size=3,
            padding="same",
            use_bias=True,
            strides=1,
            kernel_initializer=CONV_INIT,
            kernel_regularizer=l2(weight_decay),
            name=name + "FPN_P4_",
        )(P4)
    )

    outputs.append(
        layers.Conv2D(
            pyramid_filters,
            kernel_size=3,
            padding="same",
            use_bias=True,
            strides=1,
            kernel_initializer=CONV_INIT,
            kernel_regularizer=l2(weight_decay),
            name=name + "FPN_P5_",
        )(P5)
    )

    for i in range(extra_layers):
        x = layers.Conv2D(
            pyramid_filters,
            kernel_size=3,
            padding="same",
            use_bias=True,
            strides=1,
            kernel_initializer=CONV_INIT,
            kernel_regularizer=l2(weight_decay),
            name=name + "FPN_P{}_".format(i + 6),
        )(outputs[-1])
        outputs.append(x)

    return outputs


def FPNv2(
    connection_layers,
    pyramid_filters=256,
    activation="gelu",
    name="",
    extra_layers=0,
    interpolation="bilinear",
    normalization="gn",
    normalization_kw={"groups": 32},
):
    """Feature Pyramid Network with normalization and activation
    connection_layers: dict{"C2":layernameX,"C3":layernameY... to C5} dict of keras layers
    if C2 is not provided, P2 layer is not built.
    activation: activation function default 'gelu'
    name: base name of the fpn
    interpolation: type of interpolation when upscaling feature maps (default: "bilinear")
    A difference with classic FPN is that we add activation and normalization layer
    """

    if name is None:
        name = ""

    try:
        C5 = connection_layers["C5"]
        C4 = connection_layers["C4"]
        C3 = connection_layers["C3"]

    except Exception as e:
        print("Error when building FPN: can't get backbone network connection layers")
        print(e, e.args)
        raise

    input_shapes = [C5.shape[-1], C4.shape[-1], C3.shape[-1]]
    C2 = connection_layers.get("C2", None)
    if C2 is not None:
        input_shapes.append(C2.shape[-1])
    outputs = []

    # tf.print(input_shapes)

    # We must first add Normalization + activation for models using preactivation

    P5 = convblock(
        filters_in=input_shapes[0],
        filters_out=pyramid_filters,
        kernel_size=(1, 1),
        preact=False,
        activation=activation,
        name=name + "FPN_C5P5_",
        normalization=normalization,
        normalization_kw=normalization_kw,
    )(C5)

    P5up = layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=name + "FPN_P5upsampled")(P5)

    C4P4 = convblock(
        filters_in=input_shapes[1],
        filters_out=pyramid_filters,
        kernel_size=(1, 1),
        preact=False,
        activation=activation,
        name=name + "FPN_C4P4_",
        normalization=normalization,
        normalization_kw=normalization_kw,
    )(C4)

    P4 = layers.Add(name=name + "FPN_P4add")([P5up, C4P4])

    P4up = layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=name + "FPN_P4upsampled")(P4)
    C3P3 = convblock(
        filters_in=input_shapes[2],
        filters_out=pyramid_filters,
        kernel_size=(1, 1),
        preact=False,
        activation=activation,
        name=name + "FPN_C3P3_",
        normalization=normalization,
        normalization_kw=normalization_kw,
    )(C3)
    P3 = layers.Add(name=name + "FPN_P3add")([P4up, C3P3])

    if C2 is not None:
        P3up = layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=name + "FPN_P3upsampled")(P3)
        P3up = convblock(
            filters_in=pyramid_filters,
            filters_out=pyramid_filters,
            kernel_size=(3, 3),
            preact=False,
            activation=activation,
            name=name + "FPN_P3upconv_",
            normalization=normalization,
            normalization_kw=normalization_kw,
        )(P3up)
        C2P2 = convblock(
            filters_in=input_shapes[3],
            filters_out=pyramid_filters,
            kernel_size=1,
            preact=False,
            activation=activation,
            name=name + "FPN_C2P2_",
            normalization=normalization,
            normalization_kw=normalization_kw,
        )(C2)
        P2 = layers.Add(name=name + "FPN_P2add")([P3up, C2P2])

        outputs.append(
            convblock(
                filters_in=pyramid_filters,
                filters_out=pyramid_filters,
                kernel_size=3,
                preact=False,
                activation=activation,
                name=name + "FPN_P2_",
                normalization=normalization,
                normalization_kw=normalization_kw,
                attention_kernel=7,
            )(P2)
        )

    outputs.append(
        convblock(
            filters_in=pyramid_filters,
            filters_out=pyramid_filters,
            kernel_size=3,
            preact=False,
            activation=activation,
            name=name + "FPN_P3_",
            normalization=normalization,
            normalization_kw=normalization_kw,
        )(P3)
    )

    outputs.append(
        convblock(
            filters_in=pyramid_filters,
            filters_out=pyramid_filters,
            kernel_size=3,
            preact=False,
            activation=activation,
            name=name + "FPN_P4_",
            normalization=normalization,
            normalization_kw=normalization_kw,
        )(P4)
    )

    outputs.append(
        convblock(
            filters_in=pyramid_filters,
            filters_out=pyramid_filters,
            kernel_size=3,
            preact=False,
            activation=activation,
            name=name + "FPN_P5_",
            normalization=normalization,
            normalization_kw=normalization_kw,
        )(P5)
    )

    for i in range(extra_layers):
        x = convblock(
            filters_in=pyramid_filters,
            filters_out=pyramid_filters,
            kernel_size=3,
            preact=False,
            activation=activation,
            strides=2,
            name=name + "FPN_P{}_".format(i + 6),
            normalization=normalization,
            normalization_kw=normalization_kw,
        )(outputs[-1])
        outputs.append(x)

    return outputs


@tf.function()
def flatten_predictions(predictions, ncls=1, kernel_depth=256):
    """Flatten and concat head predictions for each FPN level
    predictions: [pred_cls_list, pred_kernel_list] each list contains predictions by level
    TODO: add strides tensor as output to filter binary masks by size
    """

    num_lvl = len(predictions[0])

    flat_pred_cls = [[]] * num_lvl
    flat_cls_factor = [[]] * num_lvl
    flat_pred_kernel = [[]] * num_lvl

    # Extract and flatten the i-th predicted tensors
    for lvl in range(num_lvl):
        x, y = tf.shape(predictions[0][lvl])[0], tf.shape(predictions[0][lvl])[1]
        # tf.print(tf.shape(predictions[0][lvl]), x*y)
        flat_pred_cls[lvl] = tf.reshape(predictions[0][lvl], [x * y, ncls])
        flat_pred_kernel[lvl] = tf.reshape(predictions[2][lvl], [x * y, kernel_depth])
        flat_cls_factor[lvl] = tf.reshape(predictions[1][lvl], [x * y, 1])

    # Concat predictions -> one big vector for all scales
    flat_pred_cls = tf.concat(flat_pred_cls, 0)
    flat_pred_kernel = tf.concat(flat_pred_kernel, 0)
    flat_cls_factor = tf.concat(flat_cls_factor, 0)

    return flat_pred_cls, flat_cls_factor, flat_pred_kernel


@tf.function
def compute_one_image_masks(
    inputs, cls_threshold=0.5, mask_threshold=0.5, nms_threshold=0.5, kernel_size=1, kernel_depth=256, max_inst=400, min_area=0, sigma_nms=0.5
):
    """given predicted class results and predicted kernels, compute the output one encoded mask tensor"""

    flat_pred_cls = inputs[0]
    flat_cls_factor_pred = inputs[1]
    flat_pred_kernel = inputs[2]
    masks_head_output = inputs[3]  # [H, W, mask_ch]
    geom_feats = inputs[4]  # [H, W, 1]

    # Only one prediction by pixel
    cls_labels = tf.argmax(flat_pred_cls, axis=-1)
    flat_pred_cls = tf.reduce_max(flat_pred_cls, axis=-1)
    # cls_preds = tf.sigmoid(flat_pred_cls)

    positive_idx = tf.where(flat_pred_cls >= cls_threshold)

    # topk = tf.math.top_k(flat_pred_cls, k=max_inst*2)
    # positive_idx = tf.where(topk.values >= cls_threshold)
    # positive_idx = tf.gather_nd(positive_idx, positive_idx) # take only indices where the values are > cls_threshold

    cls_scores = tf.gather_nd(flat_pred_cls, positive_idx)
    cls_labels = tf.gather_nd(cls_labels, positive_idx)
    cls_factors = tf.gather_nd(flat_cls_factor_pred, positive_idx)

    kernel_preds = tf.gather(flat_pred_kernel, positive_idx[:, 0])  # shape [N, kernel_depth*kernel_size**2]
    kernel_preds = tf.transpose(kernel_preds)  # [kernel_depth*kernel_size**2, N]
    kernel_preds = tf.reshape(
        kernel_preds, (kernel_size, kernel_size, kernel_depth, -1)
    )  # SHAPE [ks, ks, cin, cout] where cin=kernel_depth, cout=N isntances

    seg_preds = tf.sigmoid(tf.nn.conv2d(masks_head_output[tf.newaxis, ...], kernel_preds, strides=1, padding="SAME"))[
        0
    ]  # results is shape [H, W, N]

    seg_preds = tf.transpose(seg_preds, perm=[2, 0, 1])  # reshape to [N, H, W]
    binary_masks = tf.where(seg_preds >= mask_threshold, 1.0, 0.0)  # [N, H, W]

    mask_sum = tf.reduce_sum(binary_masks, axis=[1, 2])  # area of each instance (one instance per slice) -> [N]

    # scale the category score by mask confidence
    mask_scores = tf.math.divide_no_nan(tf.reduce_sum(seg_preds * binary_masks, axis=[1, 2]), mask_sum)  # [N]

    scores = cls_scores * mask_scores  # [N]

    if min_area > 0:
        indices = tf.where(mask_sum > min_area)
        cls_labels = tf.gather(cls_labels, indices[:, 0])
        scores = tf.gather(scores, indices[:, 0])
        cls_factors = tf.gather(cls_factors, indices[:, 0])
        seg_preds = tf.gather(seg_preds, indices[:, 0])
        binary_masks = tf.gather(binary_masks, indices[:, 0])
        mask_sum = tf.gather(mask_sum, indices[:, 0])

    # Filters the detected instances
    seg_preds, scores, cls_labels, cls_factors = matrix_nms(
        cls_labels,
        scores,
        cls_factors,
        seg_preds,
        binary_masks,
        mask_sum,
        post_nms_k=max_inst,
        score_threshold=nms_threshold,
        sigma=sigma_nms
    )

    # multiply each inst. by the geom factors and take the average i.e. geom_factor**1.5 / area**1.5
    # Note, maybe we should use binary masks here ?
    densities = tf.transpose(seg_preds, [1, 2, 0]) * geom_feats  # [H, W, N] * [H, W, 1] -> [H, W, N]
    densities = tf.reduce_sum(densities, axis=(0, 1))  # [N] sum values for each feature map.
    densities = densities * tf.squeeze(cls_factors)

    seg_preds = tf.RaggedTensor.from_tensor(seg_preds)

    return seg_preds, scores, cls_labels, densities


@tf.function
def matrix_nms(
    cls_labels,
    scores,
    cls_factors,
    seg_preds,
    binary_masks,
    mask_sum,
    sigma=0.5,
    pre_nms_k=PRE_NMS,
    post_nms_k=300,
    score_threshold=0.5,
    mode='gaussian',
):

    """ Matrix NMS algorithm as defined in SOLOv2 paper
    TODO: instance with no overlapping but with low score are discarded
    """
    # Select only first pre_nms_k instances (sorted by scores)
    num_selected = tf.minimum(pre_nms_k, tf.shape(scores)[0])
    indices = tf.argsort(scores, direction="DESCENDING")[:num_selected]

    # keep the selected masks, scores and labels (and mask areas)
    seg_preds = tf.gather(seg_preds, indices)
    seg_masks = tf.gather(binary_masks, indices)
    cls_labels, scores = tf.gather(cls_labels, indices), tf.gather(scores, indices)  # [N]
    cls_factors = tf.gather(cls_factors, indices)
    mask_sum = tf.gather(mask_sum, indices)  # [N]

    # calculate iou between different masks
    seg_masks = tf.reshape(seg_masks, shape=(num_selected, -1))  # [N, H*W]
    intersection = tf.matmul(seg_masks, seg_masks, transpose_b=True)  # [N, N]
    mask_sum = tf.tile(mask_sum[tf.newaxis, ...], multiples=[num_selected, 1])  # [N,N]
    union = mask_sum + tf.transpose(mask_sum) - intersection
    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.linalg.band_part(iou, 0, -1) - tf.linalg.band_part(iou, 0, 0)  # equivalent of np.triu(diagonal=1)

    # iou decay and compensation
    labels_match = tf.tile(cls_labels[tf.newaxis, ...], multiples=[num_selected, 1])
    labels_match = tf.where(labels_match == tf.transpose(labels_match), 1.0, 0.0)
    labels_match = tf.linalg.band_part(labels_match, 0, -1) - tf.linalg.band_part(labels_match, 0, 0)

    decay_iou = iou * labels_match  # iou with any object from same class
    compensate_iou = tf.reduce_max(decay_iou, axis=0)
    # tf.print("\nciou", compensate_iou, summarize=-1)

    compensate_iou = tf.tile(compensate_iou[..., tf.newaxis], multiples=[1, num_selected])

    # matrix nms
    if mode == 'gaussian':
        inv_sigma = 1.0 / sigma
        decay_coefficient = tf.reduce_min(tf.exp(-inv_sigma * (decay_iou**2 - compensate_iou**2)), axis=0)
    else:
        decay_coefficient = tf.reduce_min(tf.math.divide_no_nan(1 - decay_iou, 1 - compensate_iou), axis=0)

    # tf.print("\ndecay", decay_coefficient, summarize=-1)
    # tf.print("\nscores", scores, summarize=-1)

    # instance with no overlapping are not discarded based on score here. They are filtered earlier
    # decayed_scores = tf.where(decay_coefficient < 1, scores * decay_coefficient, score_threshold)
    decayed_scores = scores * decay_coefficient
    # tf.print("\ndecayed_scores", decayed_scores, summarize=-1)
    indices = tf.where(decayed_scores >= score_threshold)
    # indices = tf.where(decay_coefficient >= score_threshold)
    scores = tf.gather_nd(scores, indices)
    seg_preds = tf.gather(seg_preds, tf.reshape(indices, [-1]))

    num_selected = tf.minimum(post_nms_k, tf.shape(scores)[0])
    # select the final predictions
    sorted_indices = tf.argsort(scores, direction="DESCENDING")[:num_selected]
    scores = tf.gather(scores, tf.reshape(sorted_indices, [-1]))
    cls_labels = tf.gather(cls_labels, tf.reshape(sorted_indices, [-1]))
    cls_factors = tf.gather(cls_factors, tf.reshape(sorted_indices, [-1]))

    return seg_preds, scores, cls_labels, cls_factors


@tf.function
def compute_masks(
    flat_cls_pred,
    flat_cls_factor_pred,
    flat_kernel_pred,
    mask_features,
    geom_output,
    cls_threshold=0.5,
    mask_threshold=0.5,
    nms_threshold=0.5,
    kernel_depth=256,
    kernel_size=1,
    max_inst=400,
    min_area=0,
    sigma_nms=0.5
):
    """Compute mask
    inputs:
        list of predicted class features form ISMENet head
        list of predicted kernel features  from ISMENet head
        feature maps from mask_head
        ...

    """
    prediction_function = partial(
        compute_one_image_masks,
        cls_threshold=cls_threshold,
        mask_threshold=mask_threshold,
        nms_threshold=nms_threshold,
        kernel_size=kernel_size,
        kernel_depth=kernel_depth,
        max_inst=max_inst,
        min_area=min_area,
        sigma_nms=sigma_nms
    )

    seg_preds, scores, cls_labels, densities = tf.map_fn(
        prediction_function,
        [flat_cls_pred, flat_cls_factor_pred, flat_kernel_pred, mask_features, geom_output],
        fn_output_signature=(
            tf.RaggedTensorSpec(shape=(None, None, None), dtype=tf.float32, ragged_rank=1),
            tf.RaggedTensorSpec(shape=(None), dtype=tf.float32, ragged_rank=0),
            tf.RaggedTensorSpec(shape=(None), dtype=tf.int64, ragged_rank=0),
            tf.RaggedTensorSpec(shape=(None), dtype=tf.float32, ragged_rank=0),
        ),
    )
    # return predicted segmentation masks as [B, N , H, W] ragged tensor
    return seg_preds, scores, cls_labels, densities


class ISMENetModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(ISMENetModel, self).__init__(**kwargs)
        self.config = config
        self.model = ISMENet(config)
        self.kernel_depth = config.mask_output_filters * config.kernel_size**2
        self.ncls = config.ncls
        self.strides = config.strides
        # self.mask_size = (self.config.imshape[0] // self.config.mask_stride,
        #                   self.config.imshape[1] // self.config.mask_stride)

        # losses
        self.seg_loss = tf.keras.metrics.Mean(name="seg_loss", dtype=tf.float32)
        self.cls_loss = tf.keras.metrics.Mean(name="cls_loss", dtype=tf.float32)
        self.density_loss = tf.keras.metrics.Mean(name="density_loss", dtype=tf.float32)
        self.total_loss = tf.keras.metrics.Mean(name="total_loss", dtype=tf.float32)

        # Metrics
        self.precision = BinaryPrecision(name="precision")
        self.recall = BinaryRecall(name="recall")
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.precision, self.recall, self.accuracy, self.cls_loss, self.seg_loss, self.density_loss, self.total_loss]

    def call(self, inputs, training=False, **kwargs):
        default_kwargs = {
            "score_threshold": 0.5,
            "seg_threshold": 0.5,
            "nms_threshold": 0.5,
            "max_detections": 400,
            "point_nms": False,
            "min_area": 0,
            "sigma_nms": 0.5
        }

        default_kwargs.update(kwargs)

        mask_head_output, geom_output, feat_cls_list, feat_cls_factor_list, feat_kernel_list = self.model(
            inputs, training=training
        )

        # Apply Points NMS in inference
        for lvl, _ in enumerate(feat_cls_list):
            if default_kwargs.get("point_nms", False):
                feat_cls_list[lvl] = points_nms(tf.sigmoid(feat_cls_list[lvl]))
            else:
                feat_cls_list[lvl] = tf.sigmoid(feat_cls_list[lvl])

        flatten_predictions_func = partial(flatten_predictions, ncls=self.ncls, kernel_depth=self.kernel_depth)
        flat_cls_pred, flat_cls_factor_pred, flat_kernel_pred = tf.map_fn(
            flatten_predictions_func,
            [feat_cls_list, feat_cls_factor_list, feat_kernel_list],
            fn_output_signature=(tf.float32, tf.float32, tf.float32),
        )

        seg_preds, scores, cls_labels, densities = compute_masks(
            flat_cls_pred,
            flat_cls_factor_pred,
            flat_kernel_pred,
            mask_head_output,
            geom_output,
            cls_threshold=default_kwargs["score_threshold"],
            mask_threshold=default_kwargs["seg_threshold"],
            nms_threshold=default_kwargs["nms_threshold"],
            kernel_depth=self.config.mask_output_filters,
            kernel_size=self.config.kernel_size,
            max_inst=default_kwargs["max_detections"],
            min_area=default_kwargs["min_area"],
            sigma_nms=default_kwargs["sigma_nms"]
        )

        return seg_preds, scores, cls_labels, densities

    @tf.function
    def train_step(self, data):
        gt_img = data[1]
        gt_mask_img = data[2]
        gt_boxes = data[3]
        gt_cls_ids = data[4]
        gt_labels = data[5]
        gt_densities = data[6]

        nx = tf.shape(data[1])[1]
        ny = tf.shape(data[1])[2]

        # Compute flattened targets

        compute_targets_func = partial(
            utils.compute_cls_targets,
            shape=(nx, ny),
            strides=self.strides,
            grid_sizes=self.config.grid_sizes,
            scale_ranges=self.config.scale_ranges,
            offset_factor=self.config.offset_factor,
        )
        class_targets, label_targets, density_targets = tf.map_fn(
            compute_targets_func,
            (gt_boxes, gt_labels, gt_cls_ids, gt_mask_img, gt_densities),
            fn_output_signature=(tf.int32, tf.int32, tf.float32),
        )

        # OHE class targets and delete bg slice
        class_targets = tf.one_hot(class_targets, self.ncls + 1)[..., 1:]
        if self.ncls == 1:
            class_targets = class_targets[..., tf.newaxis]

        with tf.GradientTape() as tape:
            mask_head_output, geom_output, feat_cls_list, feat_cls_factor_list, feat_kernel_list = self.model(
                gt_img, training=True
            )

            flatten_predictions_func = partial(flatten_predictions, ncls=self.ncls, kernel_depth=self.kernel_depth)
            # Flattened tensor over locations and levels [B, locations, ncls] and [B, locations, kern_depth]
            flat_cls_pred, flat_cls_factor_pred, flat_kernel_pred = tf.map_fn(
                flatten_predictions_func,
                [feat_cls_list, feat_cls_factor_list, feat_kernel_list],
                fn_output_signature=(tf.float32, tf.float32, tf.float32),
            )

            flat_cls_pred = tf.sigmoid(flat_cls_pred)

            loss_function = partial(
                compute_image_loss,
                weights=self.config.lossweights,
                kernel_size=self.config.kernel_size,
                kernel_depth=self.config.mask_output_filters,
                max_pos=self.config.max_pos_samples,
                compute_cls_loss=self.config.compute_cls_loss,
                compute_seg_loss=self.config.compute_seg_loss,
                compute_density_loss=self.config.compute_density_loss
            )
            cls_loss, seg_loss, density_loss = tf.map_fn(
                loss_function,
                [
                    class_targets,
                    label_targets,
                    density_targets,
                    gt_mask_img,
                    flat_cls_pred,
                    flat_cls_factor_pred,
                    flat_kernel_pred,
                    mask_head_output,
                    geom_output,
                ],
                fn_output_signature=(tf.float32, tf.float32, tf.float32),
            )

            # avg over batch size
            cls_loss = tf.reduce_mean(cls_loss)
            seg_loss = tf.reduce_mean(seg_loss)
            density_loss = tf.reduce_mean(density_loss)

            # Update loss
            self.seg_loss.update_state(seg_loss)
            self.cls_loss.update_state(cls_loss)
            self.density_loss.update_state(density_loss)

            total_loss = cls_loss + seg_loss + density_loss
            self.total_loss.update_state(total_loss)

            grads = tape.gradient(total_loss, self.trainable_variables)

            self.optimizer.apply_gradients(
                (grad, var) for (grad, var) in zip(grads, self.trainable_variables) if grad is not None
            )

        # Update Metrics
        self.precision.update_state(class_targets, flat_cls_pred)
        self.recall.update_state(class_targets, flat_cls_pred)
        self.accuracy.update_state(class_targets, flat_cls_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        gt_img = data[1]
        gt_mask_img = data[2]
        gt_boxes = data[3]
        gt_cls_ids = data[4]
        gt_labels = data[5]
        gt_densities = data[6]

        nx = tf.shape(data[1])[1]
        ny = tf.shape(data[1])[2]

        # Compute targets
        compute_targets_func = partial(
            utils.compute_cls_targets,
            shape=(nx, ny),
            strides=self.strides,
            grid_sizes=self.config.grid_sizes,
            scale_ranges=self.config.scale_ranges,
            offset_factor=self.config.offset_factor,
        )
        class_targets, label_targets, density_targets = tf.map_fn(
            compute_targets_func,
            (gt_boxes, gt_labels, gt_cls_ids, gt_mask_img, gt_densities),
            fn_output_signature=(tf.int32, tf.int32, tf.float32),
        )

        # OHE class targets and delete bg
        class_targets = tf.one_hot(class_targets, self.ncls + 1)[..., 1:]
        if self.ncls == 1:
            class_targets = class_targets[..., tf.newaxis]

        mask_head_output, geom_output, feat_cls_list, feat_cls_factor_list, feat_kernel_list = self.model(gt_img, training=True)

        flatten_predictions_func = partial(flatten_predictions, ncls=self.ncls, kernel_depth=self.kernel_depth)
        # Flattened tensor over locations and levels [B, locations, ncls] and [B, locations, kern_depth]
        flat_cls_pred, flat_cls_factor_pred, flat_kernel_pred = tf.map_fn(
            flatten_predictions_func,
            [feat_cls_list, feat_cls_factor_list, feat_kernel_list],
            fn_output_signature=(tf.float32, tf.float32, tf.float32),
        )

        flat_cls_pred = tf.sigmoid(flat_cls_pred)

        loss_function = partial(
            compute_image_loss,
            weights=self.config.lossweights,
            kernel_size=self.config.kernel_size,
            kernel_depth=self.config.mask_output_filters,
            max_pos=self.config.max_pos_samples,
            compute_cls_loss=self.config.compute_cls_loss,
            compute_seg_loss=self.config.compute_seg_loss,
            compute_density_loss=self.config.compute_density_loss
        )
        cls_loss, seg_loss, density_loss = tf.map_fn(
            loss_function,
            [
                class_targets,
                label_targets,
                density_targets,
                gt_mask_img,
                flat_cls_pred,
                flat_cls_factor_pred,
                flat_kernel_pred,
                mask_head_output,
                geom_output,
            ],
            fn_output_signature=(tf.float32, tf.float32, tf.float32),
        )

        # avg over batch size
        cls_loss = tf.reduce_mean(cls_loss)
        seg_loss = tf.reduce_mean(seg_loss)
        density_loss = tf.reduce_mean(density_loss)

        # Update Metrics
        self.precision.update_state(class_targets, flat_cls_pred)
        self.recall.update_state(class_targets, flat_cls_pred)
        self.accuracy.update_state(class_targets, flat_cls_pred)

        # Update loss
        self.seg_loss.update_state(seg_loss)
        self.cls_loss.update_state(cls_loss)
        self.density_loss.update_state(density_loss)
        self.total_loss.update_state(cls_loss + seg_loss + density_loss)

        return {m.name: m.result() for m in self.metrics}


def train(
    model,
    train_dataset,
    epochs,
    batch_size=1,
    val_dataset=None,
    steps_per_epoch=None,
    validation_steps=None,
    optimizer=None,
    callbacks=None,
    initial_epoch=0,
    prefetch=tf.data.AUTOTUNE,
    buffer=None,
):
    # Dataset returns name, image, mask, bboxes, classes, labels

    if buffer is None:
        buffer = len(train_dataset)

    train_dataset = train_dataset.shuffle(buffer, reshuffle_each_iteration=True)
    train_dataset = train_dataset.ragged_batch(batch_size)
    # tf.data.experimental.dense_to_ragged_batch(batch_size))
    train_dataset = train_dataset.repeat(epochs)

    if val_dataset is not None:
        val_dataset = val_dataset.ragged_batch(batch_size)
        val_dataset = val_dataset.repeat(epochs)
        # tf.data.experimental.dense_to_ragged_batch(batch_size))
        if (validation_steps is None or validation_steps == 0):
            validation_steps = len(val_dataset)

    if steps_per_epoch is None:
        steps_per_epoch = len(train_dataset)

    print("Length of the batched dataset:", len(train_dataset))
    print("number of epochs:", epochs)
    print("number of training steps per epoch", steps_per_epoch)
    print("number of validation steps per epoch", validation_steps)

    train_dataset = train_dataset.prefetch(prefetch)
    if val_dataset is not None:
        val_dataset = val_dataset.prefetch(prefetch)

    # Here lr can be a scheduler
    if optimizer is not None:
        model.optimizer = optimizer
        model.compile(optimizer=model.optimizer)
    else:
        if model.optimizer is None:
            model.optimizer = tf.keras.optimizers.Adam()
            model.compile(optimizer=model.optimizer)

    print("Training using {} optimizer with lr0={}".format(model.optimizer, model.optimizer.lr.numpy()))

    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    return history


def eval_AP(model, dataset, maxiter, threshold=0.75, exclude=[], verbose=False, **kwargs):
    """compute AP score and mAP
    inputs:
    model: ISMENet model
    dataset: tf.dataset (e.g. dataset attribute of the DataLoader class)
    maxiter: number of images used
    threshold: iou threshold
    byclass: default True. If False compute the AP based only on mask iou

    outputs:
    AP
    sorted scores
    precision
    interpolated precision
    recall

    TODO: add per class AP
    TODO: deal with multiple positive predictions for the same GT object
    TODO: evaluate mass prediction (add resolution, etc.)
    """


    default_kwargs = {
        "score_threshold": 0.5,
        "seg_threshold": 0.5,
        "nms_threshold": 0.5,
        "max_detections": 400,
        "point_nms": False,
    }

    default_kwargs.update(kwargs)

    gt_accumulator = 0

    for i, data in enumerate(dataset):
        if i >= maxiter:
            break

        imgname = data[0].numpy()
        gt_img = data[1]
        gt_mask_img = data[2]
        # gt_boxes = data[3]
        gt_cls_ids = data[4]
        gt_labels = data[5]
        gt_mass = data[6]

        # ratio = (self.input_shape[0] // self.mask_stride) / new_nx

        if gt_img.ndim < 4:
            gt_img = gt_img[tf.newaxis]

        seg_preds, scores, cls_labels, _ = model(gt_img, training=False, **default_kwargs)
        seg_preds = seg_preds[0, ...].to_tensor()  # because the model outputs ragged_tensors
        scores = scores[0, ...]
        cls_labels = cls_labels[0, ...] + 1

        labels, _ = tf.unique(tf.reshape(gt_mask_img, [-1]))
        ngt = tf.size(labels) - 1
        gt_accumulator += ngt

        gt_masks = tf.one_hot(gt_mask_img, ngt + 1)[..., 1:]
        gt_masks = tf.reshape(gt_masks, [-1, ngt])
        gt_masks = tf.transpose(gt_masks, [1, 0])  # [Ngt, H*W]

        pred_masks = tf.where(seg_preds > default_kwargs["seg_threshold"], 1, 0)
        pred_masks = tf.reshape(pred_masks, shape=(pred_masks.shape[0], -1))  # [Npred, H*W]

        if verbose:
            print(
                f"Processing image {imgname} {i+1}/{maxiter}: containing {ngt} objets. Detected : {pred_masks.shape[0]}", end="\r"
            )

        gt_masks = tf.cast(gt_masks, tf.int32)

        intersection = tf.matmul(gt_masks, pred_masks, transpose_b=True)

        gt_sums = tf.reduce_sum(gt_masks, axis=1)  # Mask area -> [Ngt]
        gt_sums = tf.tile(gt_sums[tf.newaxis, ...], multiples=[pred_masks.shape[0], 1])  # [Npred, Ngt]

        pred_sums = tf.reduce_sum(pred_masks, axis=1)
        pred_sums = tf.tile(pred_sums[tf.newaxis, ...], multiples=[ngt, 1])  # [Ngt, Npred]

        union = tf.transpose(gt_sums) + pred_sums - intersection
        iou = tf.math.divide_no_nan(tf.cast(intersection, tf.float32), tf.cast(union, tf.float32))

        # iou représente la matrice des ious entre GT et PRED -> [Ngt, Npred]

        gt_inds = tf.argmax(
            iou, axis=0
        )  # indices des gt_masks donnant la meilleure valeur de iou pour chaque predictions [Npred]: il peut y avoir des doublons !
        # pred_inds = tf.argmax(iou, axis=1)  # indices des pred_masks donnant la meilleure valeur de iou pour chaque gt_masks [Ngt]

        if i == 0:
            # iou_per_gt = tf.reduce_max(iou, axis=1)   # iou max de chaque gt mask [Ngt]
            iou_per_pred = tf.reduce_max(iou, axis=0)  # iou max de chaque pred mask [Npred]
            TPFP_per_pred = tf.where(iou_per_pred > threshold, 1, 0)  # storing if a predicted mask is a TP or a FP [Npred]
            # TPFN_per_gt = tf.where(iou_per_gt > threshold, True, False)  # storing if a gt mask match with a pred_mask (-> TP) or not (-> FN) [Ngt]
            # scores_per_gt = tf.gather(scores, pred_inds)  # Get pred score of the best iou pred_mask for each gt [Ngt]
            pred_scores = scores  # [Npred]
            pred_cls_labels = cls_labels  # [Npred]
            gt_cls_labels_per_pred = tf.gather(
                gt_cls_ids, gt_inds
            )  # the gt labels associated to the pred box (gt with best iou with pred mask) [Npred]
            gt_cls_labels = gt_cls_ids
            # pred_class_per_gt =  tf.gather(cls_labels, pred_inds)     # Get pred classes of the best iou pred_mask for each gt [Ngt]

        else:
            # iou_per_gt = tf.concat([iou_per_gt, tf.reduce_max(iou, axis=1)], axis=-1)
            iou_per_pred = tf.concat([iou_per_pred, tf.reduce_max(iou, axis=0)], axis=-1)
            TPFP_per_pred = tf.concat([TPFP_per_pred, tf.where(iou_per_pred > threshold, 1, 0)], axis=-1)
            # TPFN_per_gt = tf.concat([TPFN_per_gt, tf.where(iou_per_gt > threshold, True, False)], axis=-1)
            # scores_per_gt = tf.concat([scores_per_gt, tf.gather(scores, pred_inds)], axis=-1)
            pred_scores = tf.concat([pred_scores, scores], axis=-1)
            pred_cls_labels = tf.concat([pred_cls_labels, cls_labels], axis=-1)
            gt_cls_labels_per_pred = tf.concat([gt_cls_labels_per_pred, tf.gather(gt_cls_ids, gt_inds)], axis=-1)
            gt_cls_labels = tf.concat([gt_cls_labels, gt_cls_ids], axis=-1)
            # pred_class_per_gt = tf.concat([pred_class_per_gt, tf.gather(cls_labels, pred_inds)], axis=-1)

    print("")

    # Sort predictions by score
    sorted_pred_indices = tf.argsort(pred_scores, direction="DESCENDING")
    sorted_iou_per_pred = tf.gather(iou_per_pred, sorted_pred_indices)
    sorted_TPFP_per_pred = tf.gather(TPFP_per_pred, sorted_pred_indices)
    sorted_scores = tf.gather(pred_scores, sorted_pred_indices)
    sorted_pred_cls_labels = tf.cast(tf.gather(pred_cls_labels, sorted_pred_indices), tf.int32)
    sorted_gt_cls_labels_per_pred = tf.cast(tf.gather(gt_cls_labels_per_pred, sorted_pred_indices), tf.int32)

    # Compute TP/FP using class labels AND iou threshold
    sorted_TPFP_per_pred = tf.where(sorted_gt_cls_labels_per_pred == sorted_pred_cls_labels, sorted_TPFP_per_pred, 0)

    npred = sorted_scores.shape[0]

    # list of all cls ids in gt dataset
    cls_ids, _, gt_count_per_cls = tf.unique_with_counts(gt_cls_labels)
    cls_ids = cls_ids.numpy().tolist()
    results = {}

    # AP per class
    for i, cls_id in enumerate(cls_ids):
        if cls_id in exclude:
            continue
        indexes = tf.where(gt_cls_labels_per_pred == cls_id)
        if indexes.numpy().size == 0:
            continue
        sorted_TPFP_per_pred_per_cls = tf.gather(sorted_TPFP_per_pred, indexes)
        iou_per_cls = tf.gather(sorted_iou_per_pred, indexes)
        TP = tf.cumsum(sorted_TPFP_per_pred_per_cls)
        FP = tf.cumsum(1 - sorted_TPFP_per_pred_per_cls)
        precision = tf.math.divide_no_nan(TP, (TP + FP))
        recall = TP / gt_count_per_cls[i]
        scores = tf.gather(sorted_scores, indexes)
        # Add 1 and 0 to precision and 0 and 1 to recall...
        if recall.ndim > 1:
            recall = tf.squeeze(recall, axis=1)
            precision = tf.squeeze(precision, axis=1)
            scores = tf.squeeze(scores, axis=1)
            iou_per_cls = tf.squeeze(iou_per_cls, axis=1)

        recall = tf.concat([[0], recall], axis=0)
        precision = tf.concat([[1], precision], axis=0)
        AP = auc(recall, precision)
        results[cls_id] = {
            "AP": AP,
            "precision": precision.numpy(),
            "recall": recall.numpy(),
            "count": gt_count_per_cls[i].numpy(),
            "scores": scores.numpy(),
            "iou": iou_per_cls.numpy(),
            "TPFP": sorted_TPFP_per_pred_per_cls.numpy(),
        }

    # For all classes (also excluded classes !)
    TP = tf.cumsum(sorted_TPFP_per_pred)
    FP = tf.cumsum(1 - sorted_TPFP_per_pred)
    precision = tf.math.divide_no_nan(TP, (TP + FP))
    recall = TP / gt_accumulator

    results["all"] = {
        "AP": auc(recall, precision),
        "precision": precision.numpy(),
        "recall": recall.numpy(),
        "count": gt_accumulator.numpy(),
        "scores": sorted_scores.numpy(),
        "iou": sorted_iou_per_pred.numpy(),
        "TPFP": sorted_TPFP_per_pred.numpy(),
    }

    if verbose:
        print("class |  AP  | count")
        for val, cls_id in results.items():
            print(f"{cls_id:5d} | {val['AP']:.3f} | {val['count'].numpy():5d}")

    return results
