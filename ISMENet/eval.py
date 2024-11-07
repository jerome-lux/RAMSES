# coding: utf-8

import tensorflow as tf
import numpy as np
from sklearn.metrics import auc


def eval(
    model,
    dataset,
    annotations,
    maxiter,
    scaling=1,
    threshold=0.75,
    exclude=[],
    verbose=False,
    remove_wiggles=True,
    **kwargs,
):
    """Compute AP score and mAP
    inputs:
    model: ISMENet model
    dataset:  an instance of tf.Dataset
    annotations: a pandas DataFrame containing the data for each instance and image
    scaling: image downsampling ratio (input_shape / original_imshape)
    maxiter: number of images used
    threshold: iou threshold
    remove_wiggles: remove wigglesz in the PR curve before computing AUC

    outputs:
    AP
    sorted scores
    precision
    interpolated precision
    recall

    Note: the dataset outputs mass targets (i.e. normalized by mask resolution).
    To compute actual mass, one must know the actual original image resolution and shape.
    Note: images are resized in the tf.dataset (using either padding or crop if the option crop_to_aspect_ratio is True)
    Now we only use a single ratio
    # TODO: include height and width info in the dataloader attributes?

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

    # Get the resolutions for each image
    unique_indexes = annotations.reset_index().groupby(["baseimg"])["index"].min().to_list()
    resolutions = {annotations.iloc[i]["baseimg"]: annotations.iloc[i]["res"] for i in unique_indexes}

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
        gt_mass_targets = data[6]

        if gt_img.ndim < 4:
            gt_img = gt_img[tf.newaxis]

        seg_preds, scores, cls_labels, norm_masses = model(gt_img, training=False, **default_kwargs)
        seg_preds = seg_preds[0, ...].to_tensor()  # because the model outputs ragged_tensors
        scores = scores[0, ...]
        cls_labels = cls_labels[0, ...] + 1
        masses = masses[0, ...]

        mask_stride = gt_img.shape[1] / seg_preds.shape[0]
        scale_factor = 10 * scaling / mask_stride

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
                f"Processing image {imgname} {i+1}/{maxiter}: containing {ngt} objets. Detected : {pred_masks.shape[0]}",
                end="\r",
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
            TPFP_per_pred = tf.where(
                iou_per_pred > threshold, 1, 0
            )  # storing if a predicted mask is a TP or a FP [Npred]
            # TPFN_per_gt = tf.where(iou_per_gt > threshold, True, False)  # storing if a gt mask match with a pred_mask (-> TP) or not (-> FN) [Ngt]
            # scores_per_gt = tf.gather(scores, pred_inds)  # Get pred score of the best iou pred_mask for each gt [Ngt]
            pred_scores = scores  # [Npred]
            pred_cls_labels = cls_labels  # [Npred]
            pred_masses = norm_masses / (scale_factor * resolutions[imgname])

            gt_cls_labels_per_pred = tf.gather(
                gt_cls_ids, gt_inds
            )  # the gt labels associated to the pred box (gt with best iou with pred mask) [Npred]
            gt_cls_labels = gt_cls_ids
            gt_mass_per_pred = tf.gather(gt_mass_targets, gt_inds) / (scale_factor * resolutions[imgname])
            # gt_mass = gt_mass_targets / (scale_factor * resolutions[imgname])
            # pred_class_per_gt =  tf.gather(cls_labels, pred_inds)     # Get pred classes of the best iou pred_mask for each gt [Ngt]

        else:
            # iou_per_gt = tf.concat([iou_per_gt, tf.reduce_max(iou, axis=1)], axis=-1)
            iou_per_pred = tf.concat([iou_per_pred, tf.reduce_max(iou, axis=0)], axis=-1)
            TPFP_per_pred = tf.concat([TPFP_per_pred, tf.where(iou_per_pred > threshold, 1, 0)], axis=-1)
            # TPFN_per_gt = tf.concat([TPFN_per_gt, tf.where(iou_per_gt > threshold, True, False)], axis=-1)
            # scores_per_gt = tf.concat([scores_per_gt, tf.gather(scores, pred_inds)], axis=-1)
            pred_scores = tf.concat([pred_scores, scores], axis=-1)
            pred_cls_labels = tf.concat([pred_cls_labels, cls_labels], axis=-1)
            pred_masses = tf.concat([pred_masses, norm_masses / (scale_factor * resolutions[imgname])], axis=-1)
            gt_cls_labels_per_pred = tf.concat([gt_cls_labels_per_pred, tf.gather(gt_cls_ids, gt_inds)], axis=-1)
            gt_cls_labels = tf.concat([gt_cls_labels, gt_cls_ids], axis=-1)
            gt_mass_per_pred = tf.concat(
                [gt_mass_per_pred, tf.gather(gt_mass_targets, gt_inds) / (scale_factor * resolutions[imgname])], axis=-1
            )
            # gt_mass = tf.concat([gt_mass, gt_mass_targets / (scale_factor * resolutions[imgname])], axis=-1)
            # pred_class_per_gt = tf.concat([pred_class_per_gt, tf.gather(cls_labels, pred_inds)], axis=-1)

    print("")

    # Sort predictions by score
    sorted_pred_indices = tf.argsort(pred_scores, direction="DESCENDING")
    sorted_iou_per_pred = tf.gather(iou_per_pred, sorted_pred_indices)
    sorted_TPFP_per_pred = tf.gather(TPFP_per_pred, sorted_pred_indices)
    sorted_scores = tf.gather(pred_scores, sorted_pred_indices)
    sorted_pred_cls_labels = tf.cast(tf.gather(pred_cls_labels, sorted_pred_indices), tf.int32)
    sorted_gt_cls_labels_per_pred = tf.cast(tf.gather(gt_cls_labels_per_pred, sorted_pred_indices), tf.int32)
    sorted_pred_masses = tf.gather(pred_masses, sorted_pred_indices)
    sorted_gt_masses_per_pred = tf.gather(gt_mass_per_pred, sorted_pred_indices)

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

        if recall.ndim > 1:
            recall = tf.squeeze(recall, axis=1)
            precision = tf.squeeze(precision, axis=1)
            scores = tf.squeeze(scores, axis=1)
            iou_per_cls = tf.squeeze(iou_per_cls, axis=1)

        # Add 1 and 0 to precision and 0 and 1 to recall... is it really needed?
        recall = tf.concat([[0], recall], axis=0).numpy()
        precision = tf.concat([[1], precision], axis=0).numpy()
        if remove_wiggles:
            precision = np.maximum.accumulate(precision[::-1])[::-1]
        AP = auc(recall, precision)

        # Mass error MAPE and MAE
        sorted_pred_masses_per_cls = tf.gather(sorted_pred_masses, indexes)
        sorted_gt_masses_per_pred_per_cls = tf.gather(sorted_gt_masses_per_pred, indexes)
        MAE = tf.math.abs(sorted_pred_masses_per_cls - sorted_gt_masses_per_pred_per_cls)
        MMAPE = tf.math.divide_no_nan(MAE, sorted_gt_masses_per_pred_per_cls)
        MMAPE = tf.reduce_mean(MMAPE)
        MAE = tf.reduce_mean(MAE)

        results[cls_id] = {
            "AP": AP,
            "precision": precision,
            "recall": recall,
            "count": gt_count_per_cls[i].numpy(),
            "scores": scores.numpy(),
            "iou": iou_per_cls.numpy(),
            "TPFP": sorted_TPFP_per_pred_per_cls.numpy(),
            "MassMAPE": MMAPE.numpy(),
            "MassMAE": MAE.numpy(),
        }

    # For all classes (also excluded classes !)
    TP = tf.cumsum(sorted_TPFP_per_pred)
    FP = tf.cumsum(1 - sorted_TPFP_per_pred)
    precision = tf.math.divide_no_nan(TP, (TP + FP)).numpy()
    recall = TP / gt_accumulator
    recall = recall.numpy()
    if remove_wiggles:
        precision = np.maximum.accumulate(precision[::-1])[::-1]

    # Mass error MAPE and MAE
    MAE = tf.math.abs(sorted_pred_masses - sorted_gt_masses_per_pred)
    MMAPE = tf.math.divide_no_nan(MAE, sorted_gt_masses_per_pred)
    MMAPE = tf.reduce_mean(MMAPE)
    MAE = tf.reduce_mean(MAE)

    results["all"] = {
        "AP": auc(recall, precision),
        "precision": precision.numpy(),
        "recall": recall.numpy(),
        "count": gt_accumulator.numpy(),
        "scores": sorted_scores.numpy(),
        "iou": sorted_iou_per_pred.numpy(),
        "TPFP": sorted_TPFP_per_pred.numpy(),
        "MassMAPE": MMAPE.numpy(),
        "MassMAE": MAE.numpy(),
    }

    if verbose:
        print("class |  AP  | count | Mass MAPE | Mass MAE")
        for val, cls_id in results.items():
            print(
                f"{cls_id:5d} | {val['AP']:.3f} | {val['count'].numpy():5d} |  {val['MassMAPE']:4.f}  |  {val['MassMAE']:4.f}"
            )

    return results
