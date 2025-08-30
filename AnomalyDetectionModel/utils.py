import csv
import logging
import os
import random

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
import cv2

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
    draw_contours=False,
    contour_color=(255, 0, 0),
    contour_thickness=1,
    score_threshold=1.4044,
    pixel_score_threshold=0.75,
    threshold_offset=50,
    fill_anomaly=True,
    overlay_color=(255, 0, 0),
    overlay_alpha=0.5,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
        full_pixel_threshold: [float] Full pixel-level threshold. If in (0,1], treated as normalized; otherwise as 0-255.
        threshold_offset: [int] Added to Otsu threshold (in 0-255 space) to make the mask stricter. Ignored if
            full_pixel_threshold is provided.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    # Prepare normalization for score-based binary prediction
    use_score_pred = isinstance(anomaly_scores, (list, tuple, np.ndarray)) and len(anomaly_scores) == len(image_paths)
    score_min = score_max = None
    if use_score_pred:
        scores_np = np.asarray(anomaly_scores, dtype=np.float32)
        score_min = float(scores_np.min())
        score_max = float(scores_np.max())

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
                # Ensure mask has correct shape (C, H, W) for transpose
                if mask.ndim == 2:
                    mask = mask[np.newaxis, :, :]  # Add channel dimension
                elif mask.ndim == 3 and mask.shape[0] != 3:
                    mask = mask.transpose(2, 0, 1)  # Assume (H, W, C) -> (C, H, W)
            else:
                mask = np.zeros_like(image)

        # Extract category and filename from the image path
        path_parts = image_path.replace('\\', '/').split('/')
        # Find the category (it's usually the second-to-last part before the filename)
        category = path_parts[-2] if len(path_parts) > 1 else "unknown"
        filename = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(filename)[0]

        # Create unique filename with category prefix
        savename = f"{category}_{filename_without_ext}.png"
        savename = os.path.join(savefolder, savename)

        predicted_label = "Unknown"
        predicted_as_anomaly = False
        this_score = float(anomaly_score)
        norm_score = None
        if use_score_pred and anomaly_score != "-1":
            norm_score = 0.0 if score_max == score_min else (this_score - score_min) / (score_max - score_min)

        if this_score is not None:
            predicted_label = (
                "Anomaly" if (this_score > score_threshold) else "Normal")
            predicted_as_anomaly = (this_score > score_threshold)

        #print('predicted_label', predicted_label)
        #print('this_score', this_score)

        # Create anomaly map
        try:

            overlay_rgb = image.transpose(1, 2, 0).astype(np.uint8)

            if predicted_as_anomaly:
                seg = np.asarray(segmentation, dtype=np.float32)


                max_seg_value = seg.max()
                min_seg_value = seg.min()
                seg_range = max_seg_value - min_seg_value
                if seg_range == 0:
                    seg_range = 1
                adjusted_threshold = pixel_score_threshold * seg_range + min_seg_value

                # Create mask where segmentation exceeds threshold
                anomaly_mask = seg > adjusted_threshold

                # Create red overlay with alpha
                red_overlay = np.zeros_like(overlay_rgb)
                red_overlay[..., 0] = overlay_color[0]  # Red channel
                red_overlay[..., 1] = overlay_color[1]  # Green channel
                red_overlay[..., 2] = overlay_color[2]  # Blue channel

                # Blend overlay with original image where mask is True
                overlay_rgb = np.where(
                    anomaly_mask[..., None],
                    (overlay_rgb * (1 - overlay_alpha) + red_overlay * overlay_alpha).astype(np.uint8),
                    overlay_rgb
                )

        except Exception:
            pass

        # Build figure with 3 or 4 columns including overlay
        num_cols = 3 + int(masks_provided)
        f, axes = plt.subplots(1, num_cols)
        col = 0
        axes[col].imshow(image.transpose(1, 2, 0))
        axes[col].set_title("Image")
        axes[col].axis('off')
        col += 1
        if masks_provided:
            axes[col].imshow(mask.transpose(1, 2, 0))
            axes[col].set_title("Ground Truth")
            axes[col].axis('off')
            col += 1
        axes[col].imshow(segmentation)
        axes[col].set_title("Anomaly Map")
        axes[col].axis('off')
        col += 1
        axes[col].imshow(overlay_rgb)
        if norm_score is not None:
            axes[col].set_title(f"{predicted_label} (score={this_score:.3f})")
        else:
            axes[col].set_title(f"{predicted_label}")
        axes[col].axis('off')

        f.set_size_inches(3 * num_cols, 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()



def create_storage_folder(
    main_folder_path, project_folder, group_folder, run_name, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder, run_name)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.
    #! MODIFIED

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """

    return torch.device("cpu")

    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
