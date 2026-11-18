import os
import sys
import base64
from io import BytesIO

import nibabel as nib
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy.stats import norm
from plotly.offline import get_plotlyjs
from html import escape
from matplotlib.colors import ListedColormap

REPORT_TITLE = "HiLo-BOUNTI report for neonatal MRI"
GA_MIN = 38
GA_MAX = 47

# Provisional: label 8 is excluded. Set True if training definitions count it as WM.
INCLUDE_LABEL_8_IN_WM = False

LABEL_COLORS = {0: (0, 0, 0, 0.0),
 1: (248, 235, 134, 1.0),
 2: (127, 10, 162, 1.0),
 3: (131, 4, 118, 1.0),
 4: (186, 122, 194, 1.0),
 5: (149, 120, 203, 1.0),
 6: (149, 35, 4, 1.0),
 7: (117, 3, 83, 1.0),
 8: (198, 198, 198, 1.0),
 9: (5, 185, 122, 1.0),
 10: (11, 230, 208, 1.0),
 11: (167, 75, 22, 1.0),
 12: (128, 0, 70, 1.0),
 13: (120, 0, 139, 1.0),
 14: (180, 9, 0, 1.0),
 15: (86, 14, 7, 1.0)}

COEFFICIENTS = {'total_brain_volume': {'a50': -0.9477627,
                        'b50': 107.56605,
                        'c50': -2362.3138,
                        'astd': -0.4374869,
                        'bstd': 38.1641171,
                        'cstd': -790.36946},
 'eCSF_volume': {'a50': -0.1569451,
                 'b50': 22.2029542,
                 'c50': -522.18313,
                 'astd': 0,
                 'bstd': 0.07474302,
                 'cstd': 10.9336082},
 'total_cortical_GM_volume': {'a50': -0.3613115,
                              'b50': 39.8438898,
                              'c50': -904.7165,
                              'astd': -0.1715811,
                              'bstd': 14.8020026,
                              'cstd': -305.35685},
 'total_WM_volume': {'a50': -0.4316201,
                     'b50': 41.4165146,
                     'c50': -826.73007,
                     'astd': -0.2010979,
                     'bstd': 17.2697737,
                     'cstd': -355.34186},
 'total_DGM_volume': {'a50': -0.0237324,
                      'b50': 2.65614802,
                      'c50': -55.898379,
                      'astd': -0.000915,
                      'bstd': 0.09828144,
                      'cstd': -1.7170555},
 'lateral_ventricles_volume': {'a50': 0.03117799,
                               'b50': -2.0915285,
                               'c50': 37.8050479,
                               'astd': 0,
                               'bstd': 0.30995667,
                               'cstd': -11.38877},
 'cavum_volume': {'a50': 0.00117261,
                  'b50': -0.108986,
                  'c50': 2.96397571,
                  'astd': 0,
                  'bstd': 0.00412874,
                  'cstd': 0.0798022},
 'cerebellum_volume': {'a50': 0.05039961,
                       'b50': -2.3856128,
                       'c50': 31.3791493,
                       'astd': 0,
                       'bstd': 0.05534913,
                       'cstd': -0.2260206},
 'vermis_volume': {'a50': 0.0013006,
                   'b50': 0.06040254,
                   'c50': -2.099316,
                   'astd': 0,
                   'bstd': -0.00919,
                   'cstd': 0.64095588},
 'brainstem_volume': {'a50': -0.0035726,
                      'b50': 0.62416788,
                      'c50': -13.122025,
                      'astd': 0,
                      'bstd': -0.0167961,
                      'cstd': 1.19442088}}

ROI_LABELS = {'total_brain_volume': [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
 'eCSF_volume': [1],
 'total_cortical_GM_volume': [2, 3],
 'total_WM_volume': [4, 5],
 'total_DGM_volume': [6, 7],
 'lateral_ventricles_volume': [9, 10],
 'cavum_volume': [11],
 'cerebellum_volume': [12, 13],
 'vermis_volume': [14],
 'brainstem_volume': [15]}

DISPLAY_NAMES = {'total_brain_volume': 'Total brain volume',
 'eCSF_volume': 'Extracerebral CSF volume',
 'total_cortical_GM_volume': 'Total cortical GM volume',
 'total_WM_volume': 'Total WM volume',
 'total_DGM_volume': 'Total deep GM volume',
 'lateral_ventricles_volume': 'Total lateral ventricles volume',
 'cavum_volume': 'Cavum volume',
 'cerebellum_volume': 'Cerebellum volume',
 'vermis_volume': 'Vermis volume',
 'brainstem_volume': 'Brainstem volume'}

ORDERED_VARIABLES = ['total_brain_volume',
 'eCSF_volume',
 'total_cortical_GM_volume',
 'total_WM_volume',
 'total_DGM_volume',
 'lateral_ventricles_volume',
 'cavum_volume',
 'cerebellum_volume',
 'vermis_volume',
 'brainstem_volume']

if INCLUDE_LABEL_8_IN_WM:
    ROI_LABELS["total_WM_volume"].append(8)
    ROI_LABELS["total_brain_volume"].append(8)

label_rgba = np.zeros((16, 4), dtype=float)
for _idx, (r, g, b, a) in LABEL_COLORS.items():
    label_rgba[_idx] = [r/255.0, g/255.0, b/255.0, a]
jet_transparent = ListedColormap(label_rgba)

def compute_volume_cc(label_matrix, labels, voxel_dims):
    voxel_volume_mm3 = float(voxel_dims[0] * voxel_dims[1] * voxel_dims[2])
    n_vox = int(np.isin(label_matrix, labels).sum())
    return (n_vox * voxel_volume_mm3) / 1000.0

def model_mean_std(variable, ga):
    c = COEFFICIENTS[variable]
    ga_arr = np.asarray(ga, dtype=float)
    mean = c["a50"] * ga_arr**2 + c["b50"] * ga_arr + c["c50"]
    std = c["astd"] * ga_arr**2 + c["bstd"] * ga_arr + c["cstd"]
    std = np.maximum(std, 1e-6)
    return mean, std

def centile_graph(variable, ga, measured_cc):
    x = np.linspace(GA_MIN, GA_MAX, 200)
    m, s = model_mean_std(variable, x)

    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    y5 = m - 1.645 * s
    y95 = m + 1.645 * s

    finite = np.isfinite(x) & np.isfinite(m) & np.isfinite(y5) & np.isfinite(y95)
    x = x[finite]
    m = m[finite]
    y5 = y5[finite]
    y95 = y95[finite]

    x_plot = x.astype(float).tolist()
    m_plot = m.astype(float).tolist()
    y5_plot = y5.astype(float).tolist()
    y95_plot = y95.astype(float).tolist()
    y_all = np.concatenate([m, y5, y95, np.array([measured_cc], dtype=float)])
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    pad = max((y_max - y_min) * 0.08, 1e-3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=m_plot, mode="lines", line=dict(color="black", width=2), name="50th"))
    fig.add_trace(go.Scatter(x=x_plot, y=y5_plot, mode="lines", line=dict(color="grey", dash="dot"), name="5th"))
    fig.add_trace(go.Scatter(x=x_plot, y=y95_plot, mode="lines", line=dict(color="grey", dash="dot"), name="95th"))
    fig.add_trace(go.Scatter(x=[float(ga)], y=[float(measured_cc)], mode="markers",
                             marker=dict(color="red", size=10, symbol="x"), name="Measured"))
    fig.update_layout(
        title={"text": DISPLAY_NAMES[variable], "x": 0.5, "xanchor": "center"},
        xaxis_title="Scan PMA [weeks]",
        yaxis_title="volume [cc]",
        xaxis=dict(range=[GA_MIN, GA_MAX], gridcolor="lightgrey"),
        yaxis=dict(range=[0, y_max + pad], gridcolor="lightgrey"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def evaluate_measurements(label_matrix, voxel_dims, ga):
    results = []
    for variable in ORDERED_VARIABLES:
        measured_cc = compute_volume_cc(label_matrix, ROI_LABELS[variable], voxel_dims)
        mean_cc, std_cc = model_mean_std(variable, ga)
        mean_cc = float(mean_cc)
        std_cc = float(std_cc)
        z = (measured_cc - mean_cc) / std_cc
        centile = norm.cdf(z) * 100.0
        results.append({
            "variable": variable,
            "name": DISPLAY_NAMES[variable].replace(" volume", "").replace("Volume", ""),
            "volume_cc": measured_cc,
            "oe_ratio": float(measured_cc / mean_cc) if mean_cc != 0 else float("nan"),
            "mean_cc": mean_cc,
            "std_cc": std_cc,
            "z": float(z),
            "centile": float(centile),
            "graph": centile_graph(variable, ga, measured_cc),
        })
    return results

def _select_slice_indices(label_data, axis, n_slices=7):
    other_axes = [0, 1, 2]
    other_axes.remove(axis)
    presence = np.any(label_data > 0, axis=tuple(other_axes))
    idx = np.where(presence)[0]
    if len(idx) == 0:
        size = label_data.shape[axis]
        return np.linspace(int(size * 0.35), int(size * 0.65), n_slices).astype(int).tolist()
    if len(idx) == 1:
        return [int(idx[0])] * n_slices
    # Focus on the central brain and avoid edge-adjacent slices
    q = np.linspace(0.30, 0.70, n_slices)
    sel = np.quantile(idx, q)
    sel = np.clip(np.round(sel).astype(int), idx.min(), idx.max())
    return sel.tolist()

def _show_slice(ax, img2d, lab2d=None, alpha=0.4):
    ax.imshow(img2d.T, cmap="gray", origin="lower")
    if lab2d is not None:
        ax.imshow(lab2d.T, cmap=jet_transparent, origin="lower", alpha=alpha, vmin=0, vmax=15)
    ax.axis("off")

def plot_brain_image(t2w_data, label_data):
    n_slices = 7
    fig, axs = plt.subplots(6, n_slices, figsize=(24, 18))
    alpha = 0.4

    axial_idx = _select_slice_indices(label_data, axis=2, n_slices=n_slices)
    coronal_idx = _select_slice_indices(label_data, axis=1, n_slices=n_slices)
    sagittal_idx = _select_slice_indices(label_data, axis=0, n_slices=n_slices)

    for i, sl in enumerate(axial_idx):
        _show_slice(axs[0, i], t2w_data[:, :, sl], None, alpha)
        _show_slice(axs[1, i], t2w_data[:, :, sl], label_data[:, :, sl], alpha)

    for i, sl in enumerate(coronal_idx):
        _show_slice(axs[2, i], t2w_data[:, sl, :], None, alpha)
        _show_slice(axs[3, i], t2w_data[:, sl, :], label_data[:, sl, :], alpha)

    for i, sl in enumerate(sagittal_idx):
        _show_slice(axs[4, i], t2w_data[sl, :, :], None, alpha)
        _show_slice(axs[5, i], t2w_data[sl, :, :], label_data[sl, :, :], alpha)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=140)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return image_b64

def render_report(case_id, ga, scan_date, brain_image_b64, results):
    rows_html = "".join(
        (
            (f"<tr style='color:#c00000;font-weight:700;'><td>{r['name']}</td><td>{r['volume_cc']:.2f}</td><td>{r['oe_ratio']:.3f}</td>"
             f"<td>{r['centile']:.2f}</td><td>{r['z']:.2f}</td></tr>")
            if (r['centile'] < 5 or r['centile'] > 95)
            else
            (f"<tr><td>{r['name']}</td><td>{r['volume_cc']:.2f}</td><td>{r['oe_ratio']:.3f}</td>"
             f"<td>{r['centile']:.2f}</td><td>{r['z']:.2f}</td></tr>")
        )
        for r in results
    )
    graphs_html = "".join(f"<div class='graph'>{r['graph']}</div>" for r in results)
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>{REPORT_TITLE}</title>
<script>{get_plotlyjs()}</script>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
.info-table {{ width: 100%; max-width: 1100px; border-collapse: collapse; }}
.info-table td, .info-table th {{ border: 1px solid #bbb; padding: 6px 8px; text-align: left; font-size: 13px; }}
.brain-image {{ width: 100%; max-width: 1100px; height: auto; }}
.graph-container {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 12px; }}
.graph {{ border: 1px solid #ddd; padding: 4px; }}
</style>
</head>
<body>
<h1>{REPORT_TITLE}</h1>
<table class='info-table'>
<tr><td>Case ID</td><td>{escape(case_id)}</td></tr>
<tr><td>Scan PMA</td><td>{ga:.2f} weeks</td></tr>
<tr><td>Scan date</td><td>{escape(scan_date)}</td></tr>
</table>

<br>
<img src='data:image/png;base64,{brain_image_b64}' alt='Segmentation overlay' class='brain-image'>
<br><br>
<table class='info-table'>
<tr><th>ROI</th><th>Volume [cc]</th><th>O/E</th><th>Centile</th><th>Z-score</th></tr>
{rows_html}
</table>
<br><br>
<div class='graph-container'>{graphs_html}</div>
<p style='font-size:12px; margin-top:30px'>This software has been developed for research purposes only, and hence should not be used as a diagnostic tool. In no event shall the authors or distributors be liable to any direct, indirect, special, incidental, or consequential damages arising of the use of this software, its documentation, or any derivatives thereof, even if the authors have been advised of the possibility of such damage.</p>
</body></html>"""

def main():
    if len(sys.argv) != 7:
        raise SystemExit("Usage: python script.py <case_id> <pma_weeks> <scan_date> <input_img_nii> <input_lab_nii> <output_html>")
    case_id = sys.argv[1]
    ga = float(sys.argv[2])
    if not np.isfinite(ga) or not GA_MIN <= ga <= GA_MAX:
        raise SystemExit("Scan PMA must be between 38 and 47 weeks.")
    scan_date = sys.argv[3]
    img_path = sys.argv[4]
    lab_path = sys.argv[5]
    output_html = sys.argv[6]

    img = nib.load(img_path)
    lab = nib.load(lab_path)
    if img.ndim != 3 or lab.ndim != 3 or img.shape != lab.shape or not np.allclose(img.affine, lab.affine, atol=1e-4):
        raise SystemExit("Image and segmentation must be 3D and share the same grid.")
    raw_labels = lab.get_fdata()
    if not np.all(np.isfinite(raw_labels)) or not np.allclose(raw_labels, np.rint(raw_labels)):
        raise SystemExit("Segmentation must contain finite integer label values.")
    if np.any((raw_labels < 0) | (raw_labels > 15)):
        raise SystemExit("Expected labels 0–15.")
    img_data = np.asarray(img.get_fdata())
    lab_data = np.rint(np.asarray(lab.get_fdata())).astype(np.int16)
    voxel_dims = img.header.get_zooms()[:3]

    results = evaluate_measurements(lab_data, voxel_dims, ga)
    brain_image_b64 = plot_brain_image(img_data, lab_data)
    html = render_report(case_id, ga, scan_date, brain_image_b64, results)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(output_html)

if __name__ == "__main__":
    main()
