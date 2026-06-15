#!/usr/bin/env python3

from __future__ import print_function

import argparse
import csv
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_cdt as cdt
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from skimage.filters import gaussian
import trimesh

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="nibabel"
)

warnings.filterwarnings(
    "ignore",
    message=".*from_dict class method deprecated.*"
)



LEFT_LABELS = [5, 7, 10]
RIGHT_LABELS = [4, 6, 9]
SPLIT_LABELS = [8]

LABEL_NAMES = {
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

VOLUME_GROUPS = [
    ("csf", "midline", [1]),
    ("cortical_gm", "right", [2]),
    ("cortical_gm", "left", [3]),
    ("cortical_gm", "bilateral", [2, 3]),
    ("wm", "right", [4]),
    ("wm", "left", [5]),
    ("wm", "bilateral", [4, 5]),
    ("deep_gm", "right", [6]),
    ("deep_gm", "left", [7]),
    ("deep_gm", "bilateral", [6, 7, 8]),
    ("internal_wm_background", "midline", [8]),
    ("ventricles_cavum", "right", [9]),
    ("ventricles_cavum", "left", [10]),
    ("ventricles_cavum", "midline", [11]),
    ("ventricles_cavum", "combined", [9, 10, 11]),
    ("cerebellum", "right", [12]),
    ("cerebellum", "left", [13]),
    ("posterior_fossa", "midline", [14, 15]),
    ("posterior_fossa", "combined", [12, 13, 14, 15]),
    ("total_brain", "combined", list(range(1, 16))),
]


def save_nifti_like(data, ref_nii, out_name, dtype=np.uint8):
    data = np.asarray(data)
    if dtype is not None:
        data = data.astype(dtype)
    hdr = ref_nii.header.copy()
    hdr.set_data_shape(data.shape)
    hdr.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, ref_nii.affine, hdr), out_name)


def write_gii_mesh(mesh, gifti_file, structure=None):
    coord = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    carray = nib.gifti.GiftiDataArray(
        coord,
        intent="NIFTI_INTENT_POINTSET",
    )
    farray = nib.gifti.GiftiDataArray(
        faces,
        intent="NIFTI_INTENT_TRIANGLE",
    )

    img = nib.gifti.GiftiImage(darrays=[carray, farray])

    if structure is not None:
        meta_value = {
            "CORTEX_LEFT": "CortexLeft",
            "CORTEX_RIGHT": "CortexRight",
        }.get(structure, structure)

        img.meta = nib.gifti.GiftiMetaData.from_dict({
            "AnatomicalStructurePrimary": meta_value
        })

        carray.meta = nib.gifti.GiftiMetaData.from_dict({
            "AnatomicalStructurePrimary": meta_value
        })
        farray.meta = nib.gifti.GiftiMetaData.from_dict({
            "AnatomicalStructurePrimary": meta_value
        })

    nib.save(img, gifti_file)

    if structure is not None:
        try:
            subprocess.run(
                ["wb_command", "-set-structure", gifti_file, structure],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            print(f"WARNING: wb_command -set-structure failed for {gifti_file}: {e}")


def split_labels_by_nearest_hemi(lab, left_mask, right_mask, labels_to_split):
    split_mask = np.isin(lab, labels_to_split)
    if not np.any(split_mask):
        return left_mask, right_mask

    if np.any(left_mask) and np.any(right_mask):
        dist_left = distance_transform_edt(~left_mask.astype(bool))
        dist_right = distance_transform_edt(~right_mask.astype(bool))
        left_mask = left_mask.copy()
        right_mask = right_mask.copy()
        left_mask[split_mask & (dist_left <= dist_right)] = 1
        right_mask[split_mask & (dist_right < dist_left)] = 1
    else:
        yy = np.indices(lab.shape)[1]
        mid = lab.shape[1] // 2
        left_mask = left_mask.copy()
        right_mask = right_mask.copy()
        left_mask[split_mask & (yy >= mid)] = 1
        right_mask[split_mask & (yy < mid)] = 1
    return left_mask, right_mask


def create_hemi_masks(seg_nii):
    lab = np.rint(seg_nii.get_fdata()).astype(np.uint16)
    left_mask = np.isin(lab, LEFT_LABELS).astype(np.uint8)
    right_mask = np.isin(lab, RIGHT_LABELS).astype(np.uint8)
    left_mask, right_mask = split_labels_by_nearest_hemi(lab, left_mask, right_mask, SPLIT_LABELS)

    overlap = (left_mask > 0) & (right_mask > 0)
    if np.any(overlap):
        right_mask[overlap] = 0

    bilateral = np.zeros_like(lab, dtype=np.uint8)
    bilateral[left_mask > 0] = 1
    bilateral[right_mask > 0] = 2
    return lab, left_mask.astype(np.uint8), right_mask.astype(np.uint8), bilateral


def seg2surf(mask, sigma=0.5, level=0.55):
    mask = mask.astype(bool)
    cc, nc = compute_cc(mask, connectivity=2, return_num=True)
    if nc == 0:
        raise RuntimeError("No connected component found in mask")

    sizes = np.array([np.count_nonzero(cc == i) for i in range(1, nc + 1)])
    cc_id = 1 + int(np.argmax(sizes))
    mask = cc == cc_id

    sdf = -cdt(mask) + cdt(~mask)
    sdf = gaussian(sdf.astype(np.float32), sigma=sigma)
    verts, faces, _, _ = marching_cubes(-sdf, level=-level, method="lewiner", allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True, validate=True)

    if not mesh.is_empty:
        comps = mesh.split(only_watertight=False)
        if len(comps) > 1:
            mesh = max(comps, key=lambda c: len(c.faces))
    return mesh


def smooth_mesh(mesh, iterations=20, step=0.3):
    if iterations <= 0:
        return mesh
    return trimesh.smoothing.filter_laplacian(
        mesh,
        lamb=step,
        iterations=iterations,
        implicit_time_integration=False,
        volume_constraint=False,
    )


def save_mesh(mesh, path, structure=None):
    if path.endswith(".gii") or path.endswith(".surf.gii"):
        write_gii_mesh(mesh, path, structure=structure)
    else:
        mesh.export(path)


def face_areas(mesh):
    v = mesh.vertices
    f = mesh.faces
    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def vertex_area_barycentric(mesh):
    areas = face_areas(mesh)
    va = np.zeros(len(mesh.vertices), dtype=np.float64)
    for j in range(3):
        np.add.at(va, mesh.faces[:, j], areas / 3.0)
    return va


def cotangent_laplacian_mean_curvature(mesh, eps=1e-12):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    lap = np.zeros((len(V), 3), dtype=np.float64)

    def cot(a, b):
        cr = np.linalg.norm(np.cross(a, b))
        if cr < eps:
            return 0.0
        return float(np.dot(a, b) / cr)

    for tri in F:
        i, j, k = tri
        vi, vj, vk = V[i], V[j], V[k]
        cot_i = cot(vj - vi, vk - vi)
        cot_j = cot(vi - vj, vk - vj)
        cot_k = cot(vi - vk, vj - vk)
        w_jk = 0.5 * cot_i
        w_ik = 0.5 * cot_j
        w_ij = 0.5 * cot_k
        lap[j] += w_ik * (vi - vj); lap[i] += w_ik * (vj - vi)
        lap[i] += w_jk * (vk - vi); lap[k] += w_jk * (vi - vk)
        lap[i] += w_ij * (vj - vi); lap[j] += w_ij * (vi - vj)

    area = np.maximum(vertex_area_barycentric(mesh), eps)
    hn = lap / (2.0 * area[:, None])
    H = 0.5 * np.linalg.norm(hn, axis=1)
    return np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)


def gaussian_curvature_angle_deficit(mesh, eps=1e-12):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    angle_sum = np.zeros(len(V), dtype=np.float64)
    area = vertex_area_barycentric(mesh)

    for tri in F:
        pts = V[tri]
        for loc, idx in enumerate(tri):
            a = pts[(loc + 1) % 3] - pts[loc]
            b = pts[(loc + 2) % 3] - pts[loc]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > eps and nb > eps:
                angle_sum[idx] += np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    K = (2.0 * np.pi - angle_sum) / np.maximum(area, eps)
    return np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)


def mesh_metrics(mesh, surface_name):
    H = cotangent_laplacian_mean_curvature(mesh)
    K = gaussian_curvature_angle_deficit(mesh)
    area = float(mesh.area)
    hull_area = float(mesh.convex_hull.area) if mesh.convex_hull is not None else np.nan
    si = area / hull_area if hull_area and hull_area > 0 else np.nan
    return {
        "surface": surface_name,
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "surface_area_mm2": area,
        "convex_hull_area_mm2": hull_area,
        "sulcation_index_area_over_hull": si,
        "mean_curvature_mean_abs": float(np.mean(np.abs(H))),
        "mean_curvature_median_abs": float(np.median(np.abs(H))),
        "mean_curvature_p95_abs": float(np.percentile(np.abs(H), 95)),
        "gaussian_curvature_mean_abs": float(np.mean(np.abs(K))),
        "gaussian_curvature_median_abs": float(np.median(np.abs(K))),
        "gaussian_curvature_p95_abs": float(np.percentile(np.abs(K), 95)),
    }


def write_surface_metrics_csv(path, left_metrics, right_metrics):
    rows = [left_metrics, right_metrics]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def volume_cc(lab, labels, voxel_volume_cc):
    return float(np.count_nonzero(np.isin(lab, labels)) * voxel_volume_cc)


def write_volumetry_csv(path, lab, voxel_volume_cc):
    rows = []
    for label_id in range(1, 16):
        rows.append({
            "structure": LABEL_NAMES[label_id],
            "side": "label",
            "labels": str(label_id),
            "volume_cc": volume_cc(lab, [label_id], voxel_volume_cc),
        })
    for name, side, labels in VOLUME_GROUPS:
        rows.append({
            "structure": name,
            "side": side,
            "labels": "+".join(str(x) for x in labels),
            "volume_cc": volume_cc(lab, labels, voxel_volume_cc),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure", "side", "labels", "volume_cc"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate HILO/64mT WM surfaces and volumetric/surface metrics.")
    parser.add_argument("-s", "--seg_vol", required=True)
    parser.add_argument("--left_mesh", required=True)
    parser.add_argument("--right_mesh", required=True)
    parser.add_argument("--surface_csv", required=True)
    parser.add_argument("--volumes_csv", required=True)
    parser.add_argument("--save_left_mask", default=None)
    parser.add_argument("--save_right_mask", default=None)
    parser.add_argument("--save_bilateral_mask", default=None)
    parser.add_argument("-n", "--nb_smoothing_iter", type=int, default=20)
    parser.add_argument("-dt", "--delta", type=float, default=0.3)
    args = parser.parse_args()

    seg_nii = nib.load(args.seg_vol)
    lab, left_mask, right_mask, bilateral_mask = create_hemi_masks(seg_nii)

    if args.save_left_mask:
        save_nifti_like(left_mask, seg_nii, args.save_left_mask, np.uint8)
    if args.save_right_mask:
        save_nifti_like(right_mask, seg_nii, args.save_right_mask, np.uint8)
    if args.save_bilateral_mask:
        save_nifti_like(bilateral_mask, seg_nii, args.save_bilateral_mask, np.uint8)

    print(" - generating left WM surface")
    left_mesh = smooth_mesh(seg2surf(left_mask), iterations=args.nb_smoothing_iter, step=args.delta)
    left_mesh.apply_transform(seg_nii.affine)
    save_mesh(left_mesh, args.left_mesh, structure="CORTEX_LEFT")

    print(" - generating right WM surface")
    right_mesh = smooth_mesh(seg2surf(right_mask), iterations=args.nb_smoothing_iter, step=args.delta)
    right_mesh.apply_transform(seg_nii.affine)
    save_mesh(right_mesh, args.right_mesh, structure="CORTEX_RIGHT")

    print(" - computing surface metrics")
    write_surface_metrics_csv(args.surface_csv, mesh_metrics(left_mesh, "left"), mesh_metrics(right_mesh, "right"))

    voxel_volume_cc = abs(float(np.linalg.det(seg_nii.affine[:3, :3]))) / 1000.0
    print(" - computing volumes")
    write_volumetry_csv(args.volumes_csv, lab, voxel_volume_cc)
    print(" - done")


if __name__ == "__main__":
    main()
