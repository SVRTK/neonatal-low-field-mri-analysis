#!/usr/bin/python

from __future__ import print_function
import sys
import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet
from monai.inferers import sliding_window_inference
from monai.transforms import Flip


def save_nifti_like(data, ref_nii, out_name, dtype=None):
    data = np.asarray(data)
    data = np.squeeze(data)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if dtype is not None:
        data = data.astype(dtype)
    header = ref_nii.header.copy()
    header.set_data_shape(data.shape)
    if dtype is not None:
        header.set_data_dtype(dtype)
    out_nii = nib.Nifti1Image(data, ref_nii.affine, header)
    nib.save(out_nii, out_name)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.unet = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=17,
            features=(32, 32, 64, 64, 128, 128),
            dropout=0.00,
        )

    def forward(self, x):
        return self.unet(x)


def normalise_with_mask(img, mask, eps=1e-8):
    img = img.astype(np.float32)
    mask = mask > 0
    out = np.zeros_like(img, dtype=np.float32)
    vals = img[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax > vmin:
        out[mask] = (img[mask] - vmin) / (vmax - vmin + eps)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def histogram_match_inside_mask(source, template, mask):
    """
    Match source intensity distribution to template inside mask.
    Outside mask remains zero.
    """
    source = source.astype(np.float32)
    template = template.astype(np.float32)
    mask = mask > 0
    out = np.zeros_like(source, dtype=np.float32)

    src = source[mask]
    tmpl = template[mask]
    src = src[np.isfinite(src)]
    tmpl = tmpl[np.isfinite(tmpl)]

    if src.size < 2 or tmpl.size < 2:
        out[mask] = source[mask]
        return out

    s_values, bin_idx, s_counts = np.unique(source[mask], return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template[mask], return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    out[mask] = interp_t_values[bin_idx].astype(np.float32)
    return out


def swap_lr_segmentation_channels(model_output):
    """
    model_output channels:
        0      = enhanced image
        1..16  = segmentation logits for labels 0..15
    Swap L/R label channels after Flip(1) TTA.
    """
    out = model_output.clone()
    label_pairs = [(2, 3), (4, 5), (6, 7), (9, 10), (12, 13)]
    for lab_a, lab_b in label_pairs:
        ch_a = lab_a + 1
        ch_b = lab_b + 1
        out[:, ch_a, :, :, :] = model_output[:, ch_b, :, :, :]
        out[:, ch_b, :, :, :] = model_output[:, ch_a, :, :, :]
    return out


def compute_segmentation_certainty(seg_logits, mask):
    prob = torch.softmax(seg_logits, dim=1)
    confidence, pred_label = torch.max(prob, dim=1)
    m = mask[:, 0, :, :, :] > 0.5
    confidence = confidence * m.float()
    pred_label = pred_label * m.long()
    return pred_label, confidence


def robust_certainty_from_difference(diff, mask, eps=1e-8):
    """
    Convert absolute flip-difference map to certainty map in [0,1].
    1 = identical original/flip predictions, 0 = high disagreement.
    """
    diff = np.asarray(diff, dtype=np.float32)
    mask = mask > 0
    out = np.zeros_like(diff, dtype=np.float32)
    vals = diff[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    scale = float(np.percentile(vals, 95))
    if scale <= eps:
        out[mask] = 1.0
    else:
        out[mask] = 1.0 - np.clip(diff[mask] / (scale + eps), 0.0, 1.0)
    return out.astype(np.float32)


COMBINED_ROIS = {
    "csf": [1],
    "cgm": [2, 3],
    "wm": [4, 5],
    "dgm": [6, 7, 8],
    "ventricles_cavum": [9, 10, 11],
    "posterior_fossa": [12, 13, 14, 15],
}


def write_qc_summary(csv_name, seg, seg_certainty, pred_certainty, contrast_abs, native_img, enhanced_img, mask):
    mask = mask > 0
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "roi", "labels", "n_voxels",
            "mean_segmentation_certainty",
            "mean_prediction_certainty",
            "mean_qc_contrast_abs_difference",
            "median_qc_contrast_abs_difference",
            "p95_qc_contrast_abs_difference",
            "mean_native_64mt_intensity",
            "mean_enhanced_intensity",
        ])

        for roi_name, labs in COMBINED_ROIS.items():
            roi = np.isin(seg, labs) & mask
            nvox = int(np.sum(roi))
            if nvox > 0:
                row = [
                    roi_name,
                    "+".join(str(x) for x in labs),
                    nvox,
                    float(np.mean(seg_certainty[roi])),
                    float(np.mean(pred_certainty[roi])),
                    float(np.mean(contrast_abs[roi])),
                    float(np.median(contrast_abs[roi])),
                    float(np.percentile(contrast_abs[roi], 95)),
                    float(np.mean(native_img[roi])),
                    float(np.mean(enhanced_img[roi])),
                ]
            else:
                row = [roi_name, "+".join(str(x) for x in labs), 0] + [np.nan] * 7
            writer.writerow(row)


def main():
    if len(sys.argv) != 10:
        print("Usage:")
        print("python3 run_monai_basiunet_enhancement_1case-2026-gpu.py weights.pth input_64mt.nii.gz bet_mask.nii.gz enhanced.nii.gz labels.nii.gz qc-segmentation.nii.gz qc-prediction.nii.gz qc-contrast.nii.gz qc-summary.csv")
        sys.exit(1)

    model_weights_path = sys.argv[1]
    input_img_name = sys.argv[2]
    input_mask_name = sys.argv[3]
    output_img_name = sys.argv[4]
    output_seg_name = sys.argv[5]
    output_qc_seg_name = sys.argv[6]
    output_qc_pred_name = sys.argv[7]
    output_qc_contrast_name = sys.argv[8]
    output_qc_csv_name = sys.argv[9]

    print(" - loading image and mask")
    img_nii = nib.load(input_img_name)
    mask_nii = nib.load(input_mask_name)
    img_raw = img_nii.get_fdata().astype(np.float32)
    bet_mask = (mask_nii.get_fdata() > 0).astype(np.float32)

    img_norm = normalise_with_mask(img_raw, bet_mask)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" - device:", device)

    print(" - defining model")
    generator = Generator().to(device)

    print(" - loading weights")
    with torch.no_grad():
        generator.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
        generator.eval()

    print(" - running enhancement with Flip(1) TTA")
    run_inputs = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    run_mask = torch.tensor(bet_mask).unsqueeze(0).unsqueeze(0).float().to(device)
    flp_run = Flip(1)

    with torch.no_grad():
        run_outputs = sliding_window_inference(run_inputs, (128, 128, 128), 4, generator, overlap=0.8)

        fl_inputs = flp_run(run_inputs.clone())
        fl_outputs = sliding_window_inference(fl_inputs, (128, 128, 128), 4, generator, overlap=0.8)
        fl_outputs = flp_run(fl_outputs.clone())
        fl_outputs = swap_lr_segmentation_channels(fl_outputs)

        avg_outputs = 0.5 * (run_outputs + fl_outputs)

    native_mask_np = bet_mask.astype(np.float32)

    # Image outputs before histogram matching, for prediction consistency.
    img_orig_norm = run_outputs.detach().cpu()[0, 0].numpy().astype(np.float32)
    img_flip_norm = fl_outputs.detach().cpu()[0, 0].numpy().astype(np.float32)
    img_avg_norm = avg_outputs.detach().cpu()[0, 0].numpy().astype(np.float32)

    img_orig_norm[img_orig_norm < 0] = 0
    img_flip_norm[img_flip_norm < 0] = 0
    img_avg_norm[img_avg_norm < 0] = 0
    img_avg_norm = img_avg_norm * native_mask_np

    # Histogram-match averaged enhanced output back to original 64mT intensity scale.
    img_enhanced_native = histogram_match_inside_mask(img_avg_norm, img_raw, bet_mask)
    img_enhanced_native[img_enhanced_native < 0] = 0
    img_enhanced_native = img_enhanced_native * native_mask_np

    # Segmentation from averaged logits.
    avg_seg_logits = avg_outputs[:, 1:17, :, :, :]
    pred_label, seg_confidence = compute_segmentation_certainty(avg_seg_logits, run_mask)
    seg_np = pred_label.detach().cpu()[0].numpy().astype(np.uint8)
    seg_conf_np = seg_confidence.detach().cpu()[0].numpy().astype(np.float32) * native_mask_np
    seg_np = (seg_np * native_mask_np.astype(np.uint8)).astype(np.uint8)

    # Prediction certainty from original-vs-flip enhanced image disagreement.
    pred_diff = np.abs(img_orig_norm - img_flip_norm).astype(np.float32) * native_mask_np
    pred_certainty = robust_certainty_from_difference(pred_diff, bet_mask)

    # Contrast change map on final hist-matched native-scale enhanced image.
    qc_contrast = np.abs(img_enhanced_native - img_raw).astype(np.float32) * native_mask_np

    print(" - saving outputs")
    save_nifti_like(img_enhanced_native, img_nii, output_img_name, np.float32)
    save_nifti_like(seg_np, img_nii, output_seg_name, np.uint8)
    save_nifti_like(seg_conf_np, img_nii, output_qc_seg_name, np.float32)
    save_nifti_like(pred_certainty, img_nii, output_qc_pred_name, np.float32)
    save_nifti_like(qc_contrast, img_nii, output_qc_contrast_name, np.float32)

    write_qc_summary(
        output_qc_csv_name,
        seg_np,
        seg_conf_np,
        pred_certainty,
        qc_contrast,
        img_raw,
        img_enhanced_native,
        bet_mask,
    )

    print(" - done")


if __name__ == "__main__":
    main()
