import tensorflow as tf
import tensorflow.keras.backend as backend

# from tensorflow.keras.backend import epsilon
# import sys


@tf.function()
def compute_image_loss(
    inputs,
    weights=[1.0, 1.0, 1.0],
    kernel_size=1,
    kernel_depth=256,
    max_pos=512,
    compute_seg_loss=True,
    compute_cls_loss=True,
    compute_density_loss=True,
    label_smoothing=0.1,
):
    """Compute the loss for one image
    inputs:
    cls_targets: class targets (flattened over FPN and spatial dims, one-hot) -> shape [locations, ncls]
    labels_targets: labels of positive locations ((flattened over FPN and spatial dims) [locations]
    gt_mask: gt masks with labeled instances (H, W)
    cls_pred: predicted class logits (flattened over FPN and spatial dims -> shape [locations, ncls])
    cls_factor_pred: cls factor predictions from shared heads
    kernel_pred: predicted kernel maps (flattened over FPN and spatial dims -> shape [locations, ks*ks*mask_head_depth])
    mask_head_pred: output of the mask head [H, W, mask_head_depth]
    geom_pred: output of the mask head [H, W, 1]
    max_pos: max positive GT mask (to limit memory footprint)
    To compute mask loss we take predicted kernels at gt positive locations

    """

    # bboxes loss (one box per location inside objects only)
    # takes locations where the gt class is not bg

    (
        cls_targets,
        labels_targets,
        density_targets,
        gt_masks,
        cls_pred,
        cls_factor_pred,
        kernel_pred,
        mask_head_pred,
        geom_factor_pred,
    ) = inputs

    seg_loss = 0.0
    density_loss = 0.0
    cls_loss = 0.0

    if compute_seg_loss or compute_density_loss:

        labels, _, _ = tf.unique_with_counts(labels_targets)
        max_label = tf.reduce_max(labels)

        pos_idx = tf.where(labels_targets > 0)[:, 0]

        if (max_pos is not None or max_pos > 0) and tf.shape(pos_idx)[0] > max_pos:
            pos_idx = tf.random.shuffle(pos_idx)
            pos_idx = pos_idx[:max_pos]

        ohe_masks = tf.one_hot(gt_masks, max_label + 1)[..., 1:]  # max label + 1 for bg

        # Here, each gt pixel is associated with a mask. Pixels of the same object are associated with the same masks
        # note that the tensor can be pretty big, as there are several positive pixels per instance, so that it is quite redondant

        # mask_targets = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=ohe_masks.shape[:-1])
        # for i in tf.range(tf.shape(pos_idx)):
        #     mask_targets = mask_targets.write(
        #         tf.cast(i, tf.int32), ohe_masks[..., labels_targets[pos_idx[i]] - 1]
        #     )  # labels - 1 because label 1 is at slice 0

        # mask_targets = mask_targets.stack()

        # Get the kernels corresponding to positive GT -
        kernel_pred = tf.gather(kernel_pred, pos_idx)  # shape [n_inst, kernel_depth*kernel_size**2]
        kernel_pred = tf.transpose(kernel_pred)
        kernel_pred = tf.reshape(kernel_pred, (kernel_size, kernel_size, kernel_depth, -1))  # SHAPE [ks, ks, cin, cout]
        seg_preds = tf.sigmoid(
            tf.nn.conv2d(mask_head_pred[tf.newaxis, ...], kernel_pred, strides=1, padding="SAME")
        )  # shape [1, H, W, N]

        if compute_density_loss:
            # compute densities predictions - filter the undetected masks (i.e. slices where seg_preds ~ 0), so that the geom feature doesn't try to compensate
            density_indexes = tf.where(tf.reduce_sum(seg_preds, axis=(1, 2)) > 1.0)
            density_pred = (
                tf.gather(seg_preds, density_indexes[:, 1], axis=-1) * geom_factor_pred[tf.newaxis, ...]
            )  # [1, H, W, N] * [1, H, W, 1] -> [1, H, W, N]
            density_pred = tf.squeeze(
                tf.reduce_sum(density_pred, axis=(1, 2)), axis=0
            )  # [1, N] sum values for each feature map.
            cls_factor_pred = tf.gather(cls_factor_pred, pos_idx)  # [N, 1]
            cls_factor_pred = tf.squeeze(tf.gather(cls_factor_pred, density_indexes[:, 1]), axis=-1)
            # density_pred = tf.squeeze(density_pred, axis=0) * cls_factor_pred  # [N]
            density_pred = density_pred * cls_factor_pred  # [N]
            density_targets = tf.gather(density_targets, pos_idx)  # [N, 1]
            density_targets = tf.squeeze(tf.gather(density_targets, density_indexes[:, 1]))
            density_loss = MAPEIgnoringNaN(density_targets, density_pred)

        if compute_seg_loss:
            seg_preds = tf.transpose(seg_preds[0], perm=[2, 0, 1])  # reshape to [ninstances, H, W]
            for i in tf.range(tf.shape(pos_idx)[0]):
                seg_loss = seg_loss + dice_loss(seg_preds[i, ...], ohe_masks[..., labels_targets[pos_idx[i]] - 1])

            seg_loss = tf.math.divide_no_nan(seg_loss, tf.cast(tf.shape(pos_idx)[0], tf.float32))

    if compute_cls_loss:
        cls_loss = focal_loss(cls_pred, cls_targets)
        # cls_loss = tf.keras.losses.CategoricalFocalCrossentropy(
        #     alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=label_smoothing, axis=-1
        # )(cls_targets[tf.newaxis, ...], cls_pred[tf.newaxis, ...])

    return cls_loss * weights[0], seg_loss * weights[1], density_loss * weights[2]


@tf.function()
def focal_loss(pred, gt, alpha=0.25, gamma=2.0):
    # We flatten the ohe cls tensors
    pred, gt = tf.reshape(pred, (-1, 1)), tf.reshape(gt, (-1, 1))
    anchor_obj_count = tf.cast(tf.math.count_nonzero(gt), pred.dtype)
    alpha_factor = tf.ones_like(gt) * alpha
    alpha_factor = tf.where(tf.equal(gt, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(tf.equal(gt, 1), 1 - pred, pred)
    focal_weight = alpha_factor * focal_weight**gamma / (anchor_obj_count + 1)
    return tf.losses.BinaryCrossentropy(reduction="sum", from_logits=False)(gt, pred, sample_weight=focal_weight)


@tf.function()
def dice_loss(pred, gt):
    a = tf.reduce_sum(pred * gt)
    b = tf.reduce_sum(pred * pred)
    c = tf.reduce_sum(gt * gt)
    dice = tf.math.divide_no_nan((2 * a), (b + c))
    return 1 - dice  # tf.where(dice > 0, dice, 1.)


@tf.function()
def masked_MSE(masked_y_true, y_pred, mask):
    # calculate MSE loss using a mask
    return backend.sum(tf.math.squared_difference(masked_y_true, y_pred)) / tf.maximum(
        backend.sum(tf.cast(mask, tf.float32)), 1
    )


@tf.function()
def masked_MAPE(masked_y_true, y_pred, mask, eps=backend.epsilon()):
    # calculate Mean Absolute Percentage Error using a mask - better if all data is != 0 !
    return backend.sum(tf.math.abs(y_pred - masked_y_true) / tf.maximum(tf.math.abs(masked_y_true), eps)) / tf.maximum(
        backend.sum(tf.cast(mask, tf.float32)), 1
    )


@tf.function()
def masked_MAE(masked_y_true, y_pred, mask):
    # calculate Mean Absolute Error using a mask
    return backend.sum(tf.math.abs(y_pred - masked_y_true)) / tf.maximum(backend.sum(tf.cast(mask, tf.float32)), 1)


@tf.function()
def MSEIgnoringNaN(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    masked_y_true = tf.where(mask, y_true, y_pred)
    loss = masked_MSE(masked_y_true, y_pred, mask)
    return loss


@tf.function()
def MAPEIgnoringNaN(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    masked_y_true = tf.where(mask, y_true, y_pred)
    loss = masked_MAPE(masked_y_true, y_pred, mask)
    return loss


@tf.function()
def MAEIgnoringNaN(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    masked_y_true = tf.where(mask, y_true, y_pred)
    loss = masked_MAE(masked_y_true, y_pred, mask)
    return loss
