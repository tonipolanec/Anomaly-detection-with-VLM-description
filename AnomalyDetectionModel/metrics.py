"""Anomaly metrics."""
import cv2
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    precision, recall, _ = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auc_pr = metrics.auc(recall, precision)

    print('Imagewise Threshold', thresholds)

    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )


    optimal_threshold = thresholds[np.argmax(F1_scores)]
    print('Pixelwise Threshold', optimal_threshold)
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)

    prec = metrics.precision_score(flat_ground_truth_masks.astype(int), predictions)
    rec = metrics.recall_score(flat_ground_truth_masks.astype(int), predictions)
    print(f'\nPrecision: {prec:.3f}\nRecall: {rec:.3f}')

    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


import pandas as pd
from skimage import measure
def compute_pro(masks, amaps, num_th=200):

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        # Handle empty pros list
        if len(pros) == 0:
            continue  # Skip this threshold if no regions found

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()

        # Handle division by zero
        if inverse_masks.sum() == 0:
            fpr = 0.0
        else:
            fpr = fp_pixels / inverse_masks.sum()

        # Only add row if values are valid
        pro_mean = np.mean(pros)
        if not np.isnan(pro_mean) and not np.isnan(fpr):
            new_row = pd.DataFrame([{"pro": pro_mean, "fpr": fpr, "threshold": th}])
            df = pd.concat([df, new_row], ignore_index=True)

    # Handle empty DataFrame
    if df.empty:
        return 0.0

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]

    if df.empty or df["fpr"].max() == 0:
        return 0.0

    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc