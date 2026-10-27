#!/usr/bin/env python3
"""Brain biometry report: five-column measurement images, table and centile graphs.
Install: pip install numpy scipy nibabel matplotlib plotly
Default coefficients and mapping are embedded. Inputs must already be aligned.
"""
from __future__ import annotations
import argparse,base64,csv,html,io,json,warnings
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODELS=json.loads('[{"measurement": "BPD_brain", "a": -0.09249069342672686, "b": 9.27619163059321, "c": -139.90098055341437, "a_s": 0.0, "b_s": 0.19016297247282035, "c_s": -4.680781096404219, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "BPD_skull", "a": -0.1216843086949518, "b": 11.845483365825515, "c": -187.03263082103743, "a_s": 0.0, "b_s": 0.14393203206528143, "c_s": -2.4674700873850766, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "BPD_skin", "a": -0.1313746948089966, "b": 12.738987077514514, "c": -202.46710338834941, "a_s": 0.0, "b_s": 0.15542114272494775, "c_s": -2.764750154225822, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "OFD_brain", "a": 0.057338793129761804, "b": -3.088644517191254, "c": 134.65120648001658, "a_s": 0.0, "b_s": -0.0749529837082642, "c_s": 7.087204812965442, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "OFD_skull", "a": 0.0628581620968021, "b": -3.4660402213038024, "c": 148.62424009830184, "a_s": 0.0, "b_s": -0.1148129050698572, "c_s": 9.147639063245073, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "OFD_skin", "a": 0.06202113202709823, "b": -3.302622314465019, "c": 148.56815626873683, "a_s": 0.0, "b_s": -0.12320609022621067, "c_s": 9.716450503664056, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Cerebellum_TCD", "a": -0.0922377168242144, "b": 8.779645968549739, "c": -152.145212732749, "a_s": 0.0, "b_s": 0.07201143617910433, "c_s": -0.9757233804481054, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "R_cerebellar_hemisphere_TS", "a": -0.04869164091223208, "b": 4.513202796056004, "c": -79.52122797300137, "a_s": 0.0, "b_s": 0.028702850123145832, "c_s": -0.26691845468706676, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "L_cerebellar_hemisphere_TS", "a": -0.04022427865598126, "b": 3.897905629738282, "c": -68.34343897885338, "a_s": 0.0, "b_s": -0.04177317847069514, "c_s": 2.9749912589075422, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Vermis_height", "a": -0.03785083265992264, "b": 3.6034188309899386, "c": -61.26659481271645, "a_s": 0.0, "b_s": -0.003440315205511662, "c_s": 1.00037504050879, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Vermis_APD", "a": -0.0034829872004840685, "b": 0.5505731056426479, "c": 0.35368355869399437, "a_s": 0.0, "b_s": 0.006020079621448887, "c_s": 0.461482244762457, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Pons_AP", "a": 0.005206831836207368, "b": -0.21750024070938895, "c": 13.77071373568861, "a_s": 0.0, "b_s": -0.024208847948789292, "c_s": 1.558624730047033, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Mesencephalon_width_AP_including_tectum", "a": -0.002622067532656912, "b": 0.4528653270492915, "c": -1.5208734212564625, "a_s": 0.0, "b_s": -0.01578337756865407, "c_s": 1.1593794162590498, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Medulla_oblongata_AP", "a": 0.007080944012480076, "b": -0.4338309406999531, "c": 16.288711384364753, "a_s": 0.0, "b_s": -0.014488349418783011, "c_s": 0.9986818360666763, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Cavum_septum_pellucidum_TS", "a": -0.0030861621162092553, "b": 0.33397018700336767, "c": -4.228274378341334, "a_s": 0.0, "b_s": 0.005334169445128307, "c_s": -0.032117681725883446, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "L_Deep_grey_nuclei_AP", "a": 0.02597165868957072, "b": -1.589349447922765, "c": 59.337117101786255, "a_s": 0.0, "b_s": -0.051284496252381837, "c_s": 3.6079835238232, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "L_Deep_grey_nuclei_TS", "a": -0.0227128441122196, "b": 2.2636984640760365, "c": -34.06004250032959, "a_s": 0.0, "b_s": 0.04728628720668261, "c_s": -1.1942235466340247, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "R_Deep_grey_nuclei_AP", "a": 0.017153991757203505, "b": -0.8347878876920715, "c": 43.176542049681856, "a_s": 0.0, "b_s": -0.0660756695516242, "c_s": 4.215266848874683, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "R_Deep_grey_nuclei_TS", "a": -0.03286507158733349, "b": 3.132749215671935, "c": -51.767280859372406, "a_s": 0.0, "b_s": 0.017672065745078642, "c_s": 0.09281346714725303, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "R_thalamus_AP", "a": 0.007820076910332426, "b": -0.2538588249452035, "c": 22.376221462726598, "a_s": 0.0, "b_s": 0.01771313911374555, "c_s": 0.2119619116571323, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "R_thalamus_TS", "a": -0.009531638243894705, "b": 1.0337492866051867, "c": -12.510725721182043, "a_s": 0.0, "b_s": 0.014088601751686289, "c_s": -0.10128753629199982, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "L_thalamus_AP", "a": 0.00378589576125745, "b": 0.11454427168356823, "c": 15.401384803506472, "a_s": 0.0, "b_s": -0.03748110689711388, "c_s": 2.4687153109828004, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "L_thalamus_TS", "a": -0.011544121773679871, "b": 1.1883568756903455, "c": -16.360533160104755, "a_s": 0.0, "b_s": 0.029204809207679965, "c_s": -0.7265034086755575, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Basal_ganglia_width", "a": -0.06232975031225041, "b": 5.971689944291841, "c": -97.73443846107241, "a_s": 0.0, "b_s": 0.06880199288257498, "c_s": -1.3001329276995064, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Thalamic_width", "a": -0.0484780184274725, "b": 4.6791736778044655, "c": -74.8005027358792, "a_s": 0.0, "b_s": 0.05516462676810693, "c_s": -0.9622226280069979, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Corpus_callosum_AP", "a": 0.017553477853983552, "b": -0.8212234497231414, "c": 43.86624941332296, "a_s": 0.0, "b_s": -0.04026837294116928, "c_s": 3.2389649454932448, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "CC_thickness_genu", "a": -0.0014696774070106268, "b": 0.1743785368398587, "c": -1.6913270331292387, "a_s": 0.0, "b_s": -0.006434534620949842, "c_s": 0.40107556831873326, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "CC_thickness_body", "a": -0.0015117353682714133, "b": 0.16760706253562024, "c": -2.1521630986339835, "a_s": 0.0, "b_s": -0.0036576788465196007, "c_s": 0.25473404773623753, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "CC_thickness_isthmus", "a": -0.004790410664243081, "b": 0.43764414490419856, "c": -7.761604260476462, "a_s": 0.0, "b_s": -0.005876615234349375, "c_s": 0.3492786337013161, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "CC_thickness_splenium", "a": 0.0021604512560630695, "b": -0.12033223299608055, "c": 4.473992396076549, "a_s": 0.0, "b_s": -0.011520578737490352, "c_s": 0.6463915287630504, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Tectum_length", "a": -0.01640462195589171, "b": 1.5706556949297061, "c": -26.347910969126765, "a_s": 0.0, "b_s": -0.005086357613814397, "c_s": 0.6232677193878633, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Sup_colliculus_thickness_AP", "a": -0.005837325735892143, "b": 0.5447456315252985, "c": -9.840958899749959, "a_s": 0.0, "b_s": 0.0009787574905760306, "c_s": 0.08458194683723053, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Inf_colliculus_thickness_AP", "a": 0.000469323889842389, "b": 0.008188309009860659, "c": 1.4153499466538149, "a_s": 0.0, "b_s": 0.0007539014985651873, "c_s": 0.07521753521054929, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Third_ventricle_width", "a": -0.011854188878942445, "b": 1.0728095804808355, "c": -20.505696548052057, "a_s": 0.0, "b_s": -0.009522389122300036, "c_s": 0.5900859700358997, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Fourth_ventricle_width", "a": -0.015166538218885327, "b": 1.4485789000021048, "c": -23.70643302045123, "a_s": 0.0, "b_s": 0.009413779477728472, "c_s": -0.007916367422486137, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Fourth_ventricle_AP", "a": 0.004675594764262906, "b": -0.30867205431326084, "c": 10.480833941154103, "a_s": 0.0, "b_s": -0.02136740827286976, "c_s": 1.1403888517179068, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "R_Lat_ventricle_frontal_horn_width", "a": -0.022313837516713807, "b": 1.9775005903214558, "c": -39.75836097914048, "a_s": 0.0, "b_s": 0.008571284362377612, "c_s": -0.07595450495349376, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "L_Lat_ventricle_frontal_horn_width", "a": -0.016390429500218594, "b": 1.4620289225595775, "c": -28.390223167606337, "a_s": 0.0, "b_s": -0.00045919918730317116, "c_s": 0.2420865194216935, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Bifrontal_horn_width", "a": -0.015161735182335895, "b": 1.591694246429991, "c": -19.803361692669096, "a_s": 0.0, "b_s": 0.05444736410493934, "c_s": -1.4424144645614523, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "R_Lat_ventricle_occipital_horn_width", "a": -0.0065189603001035495, "b": 0.7107124144932008, "c": -12.133527815213014, "a_s": 0.0, "b_s": 0.08999985106781254, "c_s": -3.3337995349475023, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "L_Lat_ventricle_occipital_horn_width", "a": -0.014009392171917621, "b": 1.3513571635109596, "c": -25.970527884466346, "a_s": 0.0, "b_s": 0.08481359720833125, "c_s": -3.0969690632199756, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Bioccipital_horn_width", "a": -0.056007091979150485, "b": 5.436709232011474, "c": -85.93395588760916, "a_s": 0.0, "b_s": 0.0755035606388419, "c_s": -1.5594958380903219, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}, {"measurement": "Pituitary_height", "a": -0.01745546857433125, "b": 1.5835642365467786, "c": -30.65662589379853, "a_s": 0.0, "b_s": -0.007899449668570665, "c_s": 0.5615765787141392, "n": 45, "sd_degree": 1, "pma_min": 38.43, "pma_max": 46.71}]')
PROTOCOL=json.loads('{"BPD_brain": {"label_1": 1, "label_2": 2, "plane": "axial"}, "BPD_skull": {"label_1": 3, "label_2": 4, "plane": "axial"}, "BPD_skin": {"label_1": 5, "label_2": 6, "plane": "axial"}, "OFD_brain": {"label_1": 7, "label_2": 8, "plane": "sagittal"}, "OFD_skull": {"label_1": 9, "label_2": 10, "plane": "sagittal"}, "OFD_skin": {"label_1": 11, "label_2": 12, "plane": "sagittal"}, "Cerebellum_TCD": {"label_1": 13, "label_2": 14, "plane": "axial"}, "R_cerebellar_hemisphere_TS": {"label_1": 13, "label_2": 15, "plane": "axial"}, "L_cerebellar_hemisphere_TS": {"label_1": 16, "label_2": 14, "plane": "axial"}, "Vermis_height": {"label_1": 17, "label_2": 18, "plane": "sagittal"}, "Vermis_APD": {"label_1": 19, "label_2": 20, "plane": "sagittal"}, "Pons_AP": {"label_1": 21, "label_2": 22, "plane": "sagittal"}, "Mesencephalon_width_AP_including_tectum": {"label_1": 23, "label_2": 24, "plane": "sagittal"}, "Medulla_oblongata_AP": {"label_1": 25, "label_2": 26, "plane": "sagittal"}, "Cavum_septum_pellucidum_TS": {"label_1": 27, "label_2": 28, "plane": "axial"}, "L_Deep_grey_nuclei_AP": {"label_1": 29, "label_2": 30, "plane": "axial"}, "L_Deep_grey_nuclei_TS": {"label_1": 31, "label_2": 32, "plane": "axial"}, "R_Deep_grey_nuclei_AP": {"label_1": 33, "label_2": 34, "plane": "axial"}, "R_Deep_grey_nuclei_TS": {"label_1": 35, "label_2": 36, "plane": "axial"}, "R_thalamus_AP": {"label_1": 37, "label_2": 38, "plane": "axial"}, "R_thalamus_TS": {"label_1": 39, "label_2": 40, "plane": "axial"}, "L_thalamus_AP": {"label_1": 41, "label_2": 42, "plane": "axial"}, "L_thalamus_TS": {"label_1": 43, "label_2": 44, "plane": "axial"}, "Basal_ganglia_width": {"label_1": 45, "label_2": 46, "plane": "axial"}, "Thalamic_width": {"label_1": 47, "label_2": 48, "plane": "axial"}, "Corpus_callosum_AP": {"label_1": 49, "label_2": 50, "plane": "sagittal"}, "CC_thickness_genu": {"label_1": 51, "label_2": 52, "plane": "sagittal"}, "CC_thickness_body": {"label_1": 53, "label_2": 54, "plane": "sagittal"}, "CC_thickness_isthmus": {"label_1": 55, "label_2": 56, "plane": "sagittal"}, "CC_thickness_splenium": {"label_1": 57, "label_2": 58, "plane": "sagittal"}, "Tectum_length": {"label_1": 59, "label_2": 60, "plane": "sagittal"}, "Sup_colliculus_thickness_AP": {"label_1": 61, "label_2": 62, "plane": "sagittal"}, "Inf_colliculus_thickness_AP": {"label_1": 63, "label_2": 64, "plane": "sagittal"}, "Third_ventricle_width": {"label_1": 65, "label_2": 66, "plane": "axial"}, "Fourth_ventricle_width": {"label_1": 67, "label_2": 68, "plane": "axial"}, "Fourth_ventricle_AP": {"label_1": 69, "label_2": 70, "plane": "axial"}, "R_Lat_ventricle_frontal_horn_width": {"label_1": 71, "label_2": 72, "plane": "coronal"}, "L_Lat_ventricle_frontal_horn_width": {"label_1": 73, "label_2": 74, "plane": "coronal"}, "Bifrontal_horn_width": {"label_1": 75, "label_2": 76, "plane": "axial"}, "R_Lat_ventricle_occipital_horn_width": {"label_1": 77, "label_2": 78, "plane": "coronal"}, "L_Lat_ventricle_occipital_horn_width": {"label_1": 79, "label_2": 80, "plane": "coronal"}, "Bioccipital_horn_width": {"label_1": 81, "label_2": 82, "plane": "axial"}, "Pituitary_height": {"label_1": 83, "label_2": 84, "plane": "sagittal"}}')
DISCLAIMER='This software has been developed for research purposes only, and hence should not be used as a diagnostic tool. In no event shall the authors or distributors be liable to any direct, indirect, special, incidental, or consequential damages arising of the use of this software, its documentation, or any derivatives thereof, even if the authors have been advised of the possibility of such damage.'
BASES={'axial':(np.array([1.,0,0]),np.array([0.,1,0])),
'coronal':(np.array([1.,0,0]),np.array([0.,0,1])),
'sagittal':(np.array([0.,1,0]),np.array([0.,0,1]))}


def group_rank(name):
    n=name.lower()
    if name.startswith(('BPD_','OFD_')):return 0
    if 'cerebell' in n or 'vermis' in n:return 1
    if any(x in n for x in ('deep_grey','thalam','basal_ganglia')):return 2
    if name.startswith(('Corpus_callosum','CC_')):return 3
    if any(x in n for x in ('pons','mesencephalon','medulla','tectum','colliculus','pituitary')):return 4
    return 5


def transform(points,a):
    return np.asarray(points)@a[:3,:3].T+a[:3,3]


def affine_mm(image,assume_mm=False):
    unit=image.header.get_xyzt_units()[0];factors={'mm':1.,'meter':1000.,'micron':.001}
    if unit=='unknown':
        if not assume_mm:raise ValueError('Unknown spatial units; use --assume-mm only if the affine is in mm')
        warnings.warn('Unknown spatial units: explicitly assumed mm');factor=1.
    elif unit in factors:factor=factors[unit]
    else:raise ValueError('Unsupported spatial units: '+unit)
    a=np.asarray(image.affine,dtype=float).copy();a[:3,:]*=factor
    if not np.all(np.isfinite(a)) or abs(np.linalg.det(a[:3,:3]))<1e-12:raise ValueError('Invalid affine')
    return a


def load_models(path=None):
    if path is None:return MODELS
    with Path(path).open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    keys={'measurement','a','b','c','a_s','b_s','c_s','pma_min','pma_max'}
    if not rows or not keys.issubset(rows[0]):raise ValueError('Coefficient CSV has missing columns')
    if len({r['measurement'] for r in rows})!=len(rows):raise ValueError('Duplicate model names')
    for r in rows:
        for k in keys-{'measurement'}:r[k]=float(r[k])
    if set(r['measurement'] for r in rows)!=set(PROTOCOL):raise ValueError('Models must match the protocol')
    return rows


def score(observed,pma,model,allow_extrapolation=False):
    expected=float(np.polyval([model[k] for k in ('a','b','c')],pma))
    sd=float(np.polyval([model[k] for k in ('a_s','b_s','c_s')],pma))
    outside=not model['pma_min']<=pma<=model['pma_max'];notes=[]
    if outside:notes.append('Outside reference age range'+(' (extrapolated)' if allow_extrapolation else ' (scores withheld)'))
    if not np.isfinite(expected) or expected<=0:notes.append('Nonpositive/invalid expected value')
    if not np.isfinite(sd) or sd<=0:notes.append('Nonpositive/invalid fitted SD')
    if observed is None:notes.append('Measurement unavailable')
    valid=(not outside or allow_extrapolation) and np.isfinite(expected) and expected>0 and np.isfinite(sd) and sd>0
    result={'expected_mm':expected if valid else None,'SD_mm':sd if valid else None,
            'o_e':None,'z_score':None,'centile':None,'status':'; '.join(notes) or 'OK'}
    if observed is not None and valid:
        z=(observed-expected)/sd
        result.update(o_e=observed/expected,z_score=float(z),centile=float(100*norm.cdf(z)))
    return result


def oblique_basis(points,reference):
    u,v=BASES[reference];d=points[1]-points[0];d/=np.linalg.norm(d)
    n0=np.cross(u,v);n=n0-np.dot(n0,d)*d
    if np.linalg.norm(n)<1e-8:
        n=np.cross(d,v)
        if np.linalg.norm(n)<1e-8:n=np.cross(d,u)
    n/=np.linalg.norm(n)
    if np.dot(n,n0)<0:n=-n
    x=u-np.dot(u,n)*n
    if np.linalg.norm(x)<1e-8:x=np.cross(v,n)
    x/=np.linalg.norm(x)
    return x,np.cross(n,x)


def measurement_image(data,inverse,corners,points,plane,vmin,vmax,spacing=.7):
    u,v=oblique_basis(points,plane);centre=points.mean(axis=0);relative=corners-centre
    lo=np.array([(relative@u).min(),(relative@v).min()]);hi=np.array([(relative@u).max(),(relative@v).max()])
    middle=(lo+hi)/2;side=float(np.max(hi-lo));lo=middle-side/2;hi=middle+side/2
    count=max(2,min(900,int(np.ceil(side/spacing))+1))
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],count),np.linspace(lo[1],hi[1],count))
    vox=transform(centre+xx[...,None]*u+yy[...,None]*v,inverse)
    sliced=ndi.map_coordinates(data,np.moveaxis(vox,-1,0),order=1,mode='constant',cval=vmin,prefilter=False)
    fig=plt.figure(figsize=(4,4),dpi=110,layout='none');ax=fig.add_axes([0,0,1,1]);ax.axis('off')
    ax.imshow(sliced,cmap='gray',origin='lower',extent=[lo[0],hi[0],lo[1],hi[1]],vmin=vmin,vmax=vmax,interpolation='nearest')
    delta=points-centre;ax.plot(delta@u,delta@v,color='red',linewidth=3,solid_capstyle='round')
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110,pad_inches=0,facecolor='black');plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def centile_graphs(models,records,pma):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import get_plotlyjs
    lookup={m['measurement']:m for m in models};cols=6;rows=int(np.ceil(len(records)/cols))
    fig=make_subplots(rows=rows,cols=cols,subplot_titles=[r['measurement'].replace('_',' ') for r in records],
        horizontal_spacing=.035,vertical_spacing=.045)
    for i,r in enumerate(records):
        m=lookup[r['measurement']];row=i//cols+1;col=i%cols+1;x=np.linspace(m['pma_min'],m['pma_max'],200)
        mu=np.polyval([m[k] for k in ('a','b','c')],x);sd=np.polyval([m[k] for k in ('a_s','b_s','c_s')],x)
        valid=(sd>0)&np.isfinite(sd);mu=np.where(valid,mu,np.nan);sd=np.where(valid,sd,np.nan)
        for y,name,color,width in [(mu,'50th','#111111',2),(mu+norm.ppf(.05)*sd,'5th','#999999',1.5),(mu+norm.ppf(.95)*sd,'95th','#999999',1.5)]:
            fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name=name,line=dict(color=color,width=width),showlegend=i==0,
                hovertemplate='Scan PMA %{x:.2f} weeks<br>%{y:.2f} mm<extra>'+name+'</extra>'),row=row,col=col)
        if r['observed_mm'] is not None:
            fig.add_trace(go.Scatter(x=[pma],y=[r['observed_mm']],mode='markers',name='Observed',marker=dict(color='red',size=9,symbol='x'),showlegend=i==0),row=row,col=col)
        fig.update_xaxes(title_text='Scan PMA [weeks]',row=row,col=col,showgrid=True,gridcolor='#eeeeee')
        fig.update_yaxes(title_text='mm',row=row,col=col,showgrid=True,gridcolor='#eeeeee')
    fig.update_layout(height=rows*310,width=2100,template='plotly_white',margin=dict(l=45,r=25,t=60,b=40),
        legend=dict(orientation='h',y=1.025,x=0),font=dict(size=11),paper_bgcolor='white',plot_bgcolor='white')
    for ann in fig.layout.annotations:ann.font.size=12
    return '<script>'+get_plotlyjs()+'</script>'+fig.to_html(full_html=False,include_plotlyjs=False,config={'displaylogo':False})


def generate_report(t2w_path,landmarks_path,scan_pma,subject_id,scan_date,output_html,
                    coefficients=None,assume_mm=False,allow_extrapolation=False):
    import nibabel as nib
    if not np.isfinite(scan_pma) or scan_pma<=0:raise ValueError('scan_pma must be decimal weeks')
    models=load_models(coefficients);lookup={m['measurement']:m for m in models}
    t2=nib.load(str(t2w_path));lab=nib.load(str(landmarks_path))
    if len(t2.shape)!=3 or len(lab.shape)!=3:raise ValueError('Inputs must be 3D')
    data=t2.get_fdata(dtype=np.float32);labels=lab.get_fdata(dtype=np.float64)
    if not np.all(np.isfinite(labels)) or np.any(labels<0) or not np.allclose(labels,np.rint(labels),atol=1e-5,rtol=0):raise ValueError('Landmark image must contain discrete nonnegative integer labels')
    labels=np.rint(labels).astype(np.int64);a=affine_mm(t2,assume_mm);la=affine_mm(lab,assume_mm)
    wanted={p[k] for p in PROTOCOL.values() for k in ('label_1','label_2')}
    present=sorted(wanted&set(int(v) for v in np.unique(labels)))
    means=ndi.center_of_mass(np.ones(labels.shape,dtype=np.uint8),labels,present) if present else []
    centres={label:transform(np.asarray(c),la) for label,c in zip(present,means)}
    finite=np.isfinite(data)
    if not finite.any():raise ValueError('No finite T2w intensities')
    vmin=float(data[finite].min());vmax=float(data[finite].max())
    if vmax<=vmin:raise ValueError('Constant T2w intensity range')
    data=np.where(finite,data,vmin);inverse=np.linalg.inv(a)
    corners=transform(np.array(np.meshgrid(*[(0,s-1) for s in data.shape],indexing='ij')).reshape(3,-1).T,a)
    names=sorted(PROTOCOL,key=group_rank);records=[];tiles=[];esc=lambda x:html.escape(str(x),quote=True)
    for name in names:
        p=PROTOCOL[name];missing=[p[k] for k in ('label_1','label_2') if p[k] not in centres]
        observed=None;image='';qc=[]
        if missing:qc.append('Missing landmark labels: '+', '.join(map(str,missing)))
        else:
            points=np.stack([centres[p[k]] for k in ('label_1','label_2')]);distance=float(np.linalg.norm(points[1]-points[0]))
            if distance<1e-8:qc.append('Coincident landmark centroids')
            else:
                observed=distance;vox=transform(points,inverse)
                if np.any(vox<0) or np.any(vox>np.asarray(data.shape)-1):qc.append('Endpoint outside T2w FOV; check alignment')
                else:image=measurement_image(data,inverse,corners,points,p['plane'],vmin,vmax)
        result=score(observed,scan_pma,lookup[name],allow_extrapolation)
        if qc:result['status']='; '.join(qc+([] if result['status']=='OK' else [result['status']]))
        records.append({'subject_id':subject_id,'scan_date':scan_date,'scan_pma':scan_pma,'measurement':name,'observed_mm':observed,**result})
        body='<img alt="'+esc(name)+'" src="data:image/png;base64,'+image+'">' if image else '<div class="unavailable">Image unavailable</div>'
        tiles.append('<figure><figcaption>'+esc(name.replace('_',' '))+'</figcaption>'+body+'</figure>')
    def fmt(value,decimals=2):return '&mdash;' if value is None or not np.isfinite(value) else f'{value:.{decimals}f}'
    table_rows=[]
    for r in records:
        flag=r['centile'] is not None and (r['centile']<5 or r['centile']>95)
        cells=[esc(r['measurement'].replace('_',' ')),fmt(r['observed_mm']),fmt(r['expected_mm']),fmt(r['SD_mm']),fmt(r['o_e'],3),fmt(r['z_score']),fmt(r['centile']),esc(r['status'])]
        table_rows.append('<tr'+(' class="flag"' if flag else '')+'>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    low=min(m['pma_min'] for m in models);high=max(m['pma_max'] for m in models);banner=''
    if not low<=scan_pma<=high:banner='<p class="warning">Scan PMA is outside the fitted reference range. '+('Extrapolated scores are explicitly enabled.' if allow_extrapolation else 'Reference scores are withheld.')+'</p>'
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Brain biometry: '+esc(subject_id)+'</title><style>'
    document+='body{font:15px Arial,sans-serif;background:white;color:#17202a;margin:24px}h1{font-size:28px}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:0;min-width:1250px}figure{margin:0;min-width:0;padding:0;background:white}figcaption{height:80px;padding:6px;box-sizing:border-box;font-size:19px;font-weight:600;line-height:1.15;display:flex;align-items:center;justify-content:center;text-align:center;overflow-wrap:anywhere}img{display:block;width:100%;height:auto}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ddd;padding:8px;text-align:right}th:first-child,td:first-child,td:last-child{text-align:left}.flag{background:#fff0ed}.warning{background:#fff0d0;padding:12px}.unavailable{aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#eee}.disclaimer{margin-top:30px;padding-top:16px;border-top:1px solid #ddd;font-size:13px;line-height:1.5;color:#555}</style></head><body>'
    document+='<h1>Brain MRI biometry report</h1><p>Subject ID: '+esc(subject_id)+' | Scan PMA: '+fmt(scan_pma)+' weeks | Scan date: '+esc(scan_date)+'</p>'+banner
    document+='<h2>Biometry measurements</h2><div class="scroll"><div class="grid">'+''.join(tiles)+'</div></div>'
    document+='<h2>Measurement table</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['Measurement','Observed [mm]','Expected [mm]','SD [mm]','O/E','Z-score','Centile','QC'])+'</tr></thead><tbody>'+''.join(table_rows)+'</tbody></table></div>'
    document+='<h2>Reference curves</h2><div class="scroll">'+centile_graphs(models,records,scan_pma)+'</div>'
    document+='<footer class="disclaimer">'+esc(DISCLAIMER)+'</footer></body></html>'
    output=Path(output_html);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(document,encoding='utf-8')
    with output.with_suffix('.measurements.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t2w',type=Path,required=True);p.add_argument('--landmarks',type=Path,required=True)
    p.add_argument('--scan-pma',type=float,required=True);p.add_argument('--subject-id',required=True)
    p.add_argument('--date',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coefficients',type=Path);p.add_argument('--assume-mm',action='store_true')
    p.add_argument('--allow-extrapolation',action='store_true');a=p.parse_args()
    records=generate_report(a.t2w,a.landmarks,a.scan_pma,a.subject_id,a.date,a.output,a.coefficients,a.assume_mm,a.allow_extrapolation)
    print(f'Saved {a.output}: {len(records)} measurements')

if __name__=='__main__':main()
