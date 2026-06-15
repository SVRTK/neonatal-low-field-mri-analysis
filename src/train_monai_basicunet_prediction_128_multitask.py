#!/usr/bin/python

from __future__ import print_function

import os
import sys
import warnings

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, SSIMLoss
from monai.networks.nets import BasicUNet
from monai.transforms import Compose, LoadImaged, RandAffined, ScaleIntensityd, ToTensord, Flip

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


ROI_NAMES = {
    0: "background",
    1: "csf",
    2: "cortical_gm_R",
    3: "cortical_gm_L",
    4: "wm_R",
    5: "wm_L",
    6: "deep_gm_R",
    7: "deep_gm_L",
    8: "internal_wm_background",
    9: "ventricle_R",
    10: "ventricle_L",
    11: "cavum",
    12: "cerebellum_R",
    13: "cerebellum_L",
    14: "vermis",
    15: "brainstem",
}


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


def sobel3d_kernels(device, dtype):
    d = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
    s = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
    kx = d[None, None, :, None, None] * s[None, None, None, :, None] * s[None, None, None, None, :]
    ky = s[None, None, :, None, None] * d[None, None, None, :, None] * s[None, None, None, None, :]
    kz = s[None, None, :, None, None] * s[None, None, None, :, None] * d[None, None, None, None, :]
    return kx / 32.0, ky / 32.0, kz / 32.0


def gradient_magnitude_3d(x):
    kx, ky, kz = sobel3d_kernels(x.device, x.dtype)
    channels = x.shape[1]
    kx = kx.repeat(channels, 1, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1, 1)
    kz = kz.repeat(channels, 1, 1, 1, 1)
    gx = F.conv3d(x, kx, padding=1, groups=channels)
    gy = F.conv3d(x, ky, padding=1, groups=channels)
    gz = F.conv3d(x, kz, padding=1, groups=channels)
    return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-8)


def ensure_mask(mask, like):
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    mask = (mask > 0.5).float()
    return mask.to(device=like.device, dtype=like.dtype)


def masked_mean(x, mask, eps=1e-6):
    return (x * mask).sum() / (mask.sum() + eps)


def masked_l1(pred, target, mask):
    return masked_mean(torch.abs(pred - target), mask)


def masked_ssim(pred, target, mask, ssim_fn):
    return ssim_fn(pred * mask, target * mask)


def masked_edge(pred, target, mask):
    pred_s = F.avg_pool3d(pred, kernel_size=3, stride=1, padding=1)
    target_s = F.avg_pool3d(target, kernel_size=3, stride=1, padding=1)
    return masked_mean(torch.abs(gradient_magnitude_3d(pred_s) - gradient_magnitude_3d(target_s)), mask)


def tissue_mean_contrast_loss(pred_img, native_img, target_label, mask, n_labels=16, eps=1e-6):
    mask = ensure_mask(mask, pred_img)

    if native_img.dim() == 4:
        native_img = native_img.unsqueeze(1)
    native_img = native_img.to(device=pred_img.device, dtype=pred_img.dtype)

    lab = target_label[:, 0, :, :, :] if target_label.dim() == 5 else target_label

    loss = pred_img.new_tensor(0.0)
    count = 0
    for lab_idx in range(1, n_labels):
        roi = ((lab == lab_idx).unsqueeze(1).float() * mask).to(pred_img.device)
        if roi.sum() > 20:
            pred_mean = (pred_img * roi).sum() / (roi.sum() + eps)
            native_mean = (native_img * roi).sum() / (roi.sum() + eps)
            loss = loss + torch.abs(pred_mean - native_mean)
            count += 1

    return loss / count if count > 0 else pred_img.new_tensor(0.0)


def save_nifti_like(data, ref_nii, out_path, dtype=None):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.squeeze(np.asarray(data))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if dtype is not None:
        data = data.astype(dtype)
    header = ref_nii.header.copy()
    header.set_data_shape(data.shape)
    if dtype is not None:
        header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(data, ref_nii.affine, header), out_path)


def swap_lr_segmentation_channels(model_output):
    out = model_output.clone()
    for lab_a, lab_b in [(2, 3), (4, 5), (6, 7), (9, 10), (12, 13)]:
        ch_a = lab_a + 1
        ch_b = lab_b + 1
        out[:, ch_a, :, :, :] = model_output[:, ch_b, :, :, :]
        out[:, ch_b, :, :, :] = model_output[:, ch_a, :, :, :]
    return out


def predict_with_flip_tta(generator, run_inputs):
    flp_run = Flip(1)
    run_outputs = generator(run_inputs)
    fl_inputs = flp_run(run_inputs.clone())
    fl_outputs = generator(fl_inputs)
    fl_outputs = flp_run(fl_outputs.clone())
    fl_outputs = swap_lr_segmentation_channels(fl_outputs)
    return 0.5 * (run_outputs + fl_outputs)


def get_prediction_arrays(model_output, mask_np):
    pred_img = model_output.detach().cpu()[0, 0].numpy().astype(np.float32)
    pred_img[pred_img < 0] = 0
    pred_img = pred_img * mask_np.astype(np.float32)

    seg_logits = model_output[:, 1:17, :, :, :]
    pred_label = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1)
    pred_seg = pred_label.detach().cpu()[0].numpy().astype(np.uint8)
    pred_seg = (pred_seg * mask_np.astype(np.uint8)).astype(np.uint8)
    return pred_img, pred_seg


files_path = sys.argv[1]
check_path = sys.argv[2]
json_file = sys.argv[3]
results_path = sys.argv[4]
res = int(sys.argv[5])
status_train_proc = int(sys.argv[6])
status_load_check = int(sys.argv[7])
max_iterations = int(sys.argv[8])
roi_type = sys.argv[9]

root_dir = files_path
os.chdir(root_dir)
os.makedirs(results_path, exist_ok=True)

degree_min = -0.15
degree_max = 0.15

train_transforms = Compose([
    LoadImaged(keys=["image", "prediction", "mask", "label"]),
    ScaleIntensityd(keys=["image", "prediction"], minv=0.0, maxv=1.0),
    RandAffined(
        keys=["image", "prediction", "mask", "label"],
        rotate_range=[(degree_min, degree_max), (degree_min, degree_max), (degree_min, degree_max)],
        mode=("bilinear", "bilinear", "nearest", "nearest"),
        padding_mode="border",
        prob=0.7,
    ),
    ToTensord(keys=["image", "prediction", "mask", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "prediction", "mask", "label"]),
    ScaleIntensityd(keys=["image", "prediction"], minv=0.0, maxv=1.0),
    ToTensord(keys=["image", "prediction", "mask", "label"]),
])

run_transforms = Compose([
    LoadImaged(keys=["image"]),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    ToTensord(keys=["image"]),
])

print("Loading data ...")
datasets = files_path + json_file

if status_train_proc > 0:
    train_datalist = load_decathlon_datalist(datasets, True, "training")
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms, cache_num=50, cache_rate=1.0, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    val_datalist = load_decathlon_datalist(datasets, True, "validation")
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms, cache_num=20, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
else:
    run_datalist = load_decathlon_datalist(datasets, True, "running")
    run_ds = CacheDataset(data=run_datalist, transform=run_transforms, cache_num=20, cache_rate=1.0, num_workers=0)
    run_loader = DataLoader(run_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print("Defining the model ...")
generator = Generator().to(device)

ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0, win_size=7)
seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)

lambda_l1 = 0.50
lambda_ssim = 0.20
lambda_edge = 0.10
lambda_seg = 0.20
lambda_tissue = 0.00


def total_loss(pred, target, mask):
    mask = ensure_mask(mask, pred)
    loss_l1 = masked_l1(pred, target, mask)
    loss_ssim = masked_ssim(pred, target, mask, ssim_loss)
    loss_edge = masked_edge(pred, target, mask)
    loss_img = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim + lambda_edge * loss_edge
    return loss_img, loss_l1, loss_ssim, loss_edge


def total_multitask_loss(model_output, native_img, target_img, target_label, mask):
    pred_img = model_output[:, 0:1, :, :, :]
    pred_seg_logits = model_output[:, 1:17, :, :, :]

    loss_img, loss_l1, loss_ssim, loss_edge = total_loss(pred_img, target_img, mask)

    if target_label.dim() == 4:
        target_label = target_label.unsqueeze(1)
    target_label = target_label.long()

    loss_seg = seg_loss_fn(pred_seg_logits, target_label)
    loss_tissue = tissue_mean_contrast_loss(pred_img, native_img, target_label, mask, n_labels=16)
    total = loss_img + lambda_seg * loss_seg + lambda_tissue * loss_tissue
    return total, loss_img, loss_l1, loss_ssim, loss_edge, loss_seg, loss_tissue


generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=1e-5)

torch.backends.cudnn.benchmark = True

if status_load_check > 0:
    print("Loading the checkpoint ...")
    checkpoint_name = roi_type + ("_latest_metric_model.pth" if status_load_check == 2 else "_best_metric_model.pth")
    generator.load_state_dict(torch.load(os.path.join(check_path, checkpoint_name), map_location=device), strict=False)
    generator.eval()

if status_train_proc > 0:
    print("Training ...")
    eval_num = 100
    global_step = 0
    mse_val_best = 100.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    def validation(epoch_iterator_val):
        generator.eval()
        loss_vals = []
        with torch.no_grad():
            for _, batch in enumerate(epoch_iterator_val):
                val_inputs = batch["image"].unsqueeze(0).to(device)
                val_pred = batch["prediction"].unsqueeze(0).to(device)
                val_mask = batch["mask"].unsqueeze(0).to(device)
                val_label = batch["label"].unsqueeze(0).to(device)
                val_outputs = generator(val_inputs)
                loss, _, _, _, _, _, _ = total_multitask_loss(val_outputs, val_inputs, val_pred, val_label, val_mask)
                loss_vals.append(loss.item())
                epoch_iterator_val.set_description("Validate (%d Steps) (loss=%2.5f)" % (global_step, loss))
        return float(np.mean(loss_vals)), loss_vals

    def train(global_step, train_loader, mse_val_best, global_step_best):
        generator.train()
        epoch_loss = 0.0
        step = 0
        epoch_loss_train_list = []
        epoch_loss_val_list = []
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

        while global_step < max_iterations:
            for _, batch in enumerate(epoch_iterator):
                if global_step >= max_iterations:
                    break
                step += 1

                x = batch["image"].unsqueeze(0).to(device)
                y = batch["prediction"].unsqueeze(0).to(device)
                m = batch["mask"].unsqueeze(0).to(device)
                lab = batch["label"].unsqueeze(0).to(device)

                logit_map = generator(x)
                loss, loss_img, loss_l1, loss_ssim, loss_edge, loss_seg, loss_tissue = total_multitask_loss(logit_map, x, y, lab, m)

                loss.backward()
                epoch_loss += loss.item()
                epoch_loss_train_list.append(loss.item())
                generator_optimizer.step()
                generator_optimizer.zero_grad()

                print(
                    f"Step {global_step} | Total {loss.item():.4f} | Img {loss_img.item():.4f} | "
                    f"L1 {loss_l1.item():.4f} | SSIM {loss_ssim.item():.4f} | "
                    f"Edge {loss_edge.item():.4f} | Seg {loss_seg.item():.4f} | Native64 {loss_tissue.item():.4f}"
                )
                epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

                if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps)", dynamic_ncols=True)
                    mse_val, epoch_loss_val_list = validation(epoch_iterator_val)
                    epoch_loss /= step
                    epoch_loss_values.append(epoch_loss)
                    metric_values.append(mse_val)

                    torch.save(generator.state_dict(), os.path.join(root_dir, roi_type + "_latest_metric_model.pth"))

                    if mse_val < mse_val_best:
                        mse_val_best = mse_val
                        global_step_best = global_step
                        torch.save(generator.state_dict(), os.path.join(root_dir, roi_type + "_best_metric_model.pth"))
                        print("Model Was Saved ! Current Best Avg. loss: {} Current Avg. loss: {}".format(mse_val_best, mse_val))
                    else:
                        print("Model Was Not Saved ! Current Best Avg. loss: {} Current Avg. loss: {}".format(mse_val_best, mse_val))

                global_step += 1
        return global_step, mse_val_best, global_step_best, epoch_loss_train_list, epoch_loss_val_list

    train_loss_list = []
    val_loss_list = []
    while global_step < max_iterations:
        global_step, mse_val_best, global_step_best, epoch_loss_train_list, epoch_loss_val_list = train(global_step, train_loader, mse_val_best, global_step_best)
        train_loss_list.append(epoch_loss_train_list)
        val_loss_list.append(epoch_loss_val_list)

    print("Generating validation results ...")
    generator.eval()
    for case_num in range(len(val_datalist)):
        img_name = val_datalist[case_num]["image"]
        case_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        out_name = os.path.join(results_path, "cnn-output-" + case_name)
        out_seg_name = os.path.join(results_path, "cnn-seg-output-" + case_name)
        print(case_num, out_name)

        img_tmp_info = nib.load(img_name)
        with torch.no_grad():
            img = val_ds[case_num]["image"].unsqueeze(0)
            run_inputs = torch.unsqueeze(img, 1).to(device)
            run_mask = val_ds[case_num]["mask"].unsqueeze(0).unsqueeze(0).to(device)
            out = generator(run_inputs)
            mask_np = run_mask.detach().cpu()[0, 0].numpy() > 0
            out_img, out_seg = get_prediction_arrays(out, mask_np)
            save_nifti_like(out_img, img_tmp_info, out_name, np.float32)
            save_nifti_like(out_seg, img_tmp_info, out_seg_name, np.uint8)

else:
    print("Running ...")
    generator.eval()

    for case_num in range(len(run_datalist)):
        img_name = run_datalist[case_num]["image"]
        case_name = os.path.split(img_name)[1]
        out_name = os.path.join(results_path, "cnn-output-flipavg-" + case_name)
        out_seg_name = os.path.join(results_path, "cnn-seg-output-flipavg-" + case_name)
        print(case_num, out_name)

        img_tmp_info = nib.load(img_name)
        with torch.no_grad():
            img = run_ds[case_num]["image"].unsqueeze(0)
            run_inputs = torch.unsqueeze(img, 1).to(device)
            run_mask = (run_inputs > 0).float()
            native_mask = run_mask.detach().cpu()[0, 0].numpy() > 0

            avg_outputs = predict_with_flip_tta(generator, run_inputs)
            out_img_avg, out_seg_avg = get_prediction_arrays(avg_outputs, native_mask)

            save_nifti_like(out_img_avg, img_tmp_info, out_name, np.float32)
            save_nifti_like(out_seg_avg, img_tmp_info, out_seg_name, np.uint8)
