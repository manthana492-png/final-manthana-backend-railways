"""TotalSegmentator mask stem → canonical pathology_scores keys (per task)."""

from __future__ import annotations

# task=total — common organs (stem without .nii.gz)
TOTALSEG_TOTAL_VOLUME_KEYS: dict[str, str] = {
    "liver": "liver_cm3",
    "spleen": "spleen_cm3",
    "kidney_right": "kidney_right_cm3",
    "kidney_left": "kidney_left_cm3",
    "gallbladder": "gallbladder_cm3",
    "pancreas": "pancreas_cm3",
    "stomach": "stomach_cm3",
    "aorta": "aorta_cm3",
    "urinary_bladder": "bladder_cm3",
    "lung_right": "lung_right_cm3",
    "lung_left": "lung_left_cm3",
}

# task=total_mr — whole-body MRI (50 stems, StanfordMIMI TotalSegmentatorV2 class_map)
TOTALSEG_TOTAL_MR_VOLUME_KEYS: dict[str, str] = {
    "spleen": "spleen_cm3",
    "kidney_right": "kidney_right_cm3",
    "kidney_left": "kidney_left_cm3",
    "gallbladder": "gallbladder_cm3",
    "liver": "liver_cm3",
    "stomach": "stomach_cm3",
    "pancreas": "pancreas_cm3",
    "adrenal_gland_right": "adrenal_gland_right_cm3",
    "adrenal_gland_left": "adrenal_gland_left_cm3",
    "lung_left": "lung_left_cm3",
    "lung_right": "lung_right_cm3",
    "esophagus": "esophagus_cm3",
    "small_bowel": "small_bowel_cm3",
    "duodenum": "duodenum_cm3",
    "colon": "colon_cm3",
    "urinary_bladder": "urinary_bladder_cm3",
    "prostate": "prostate_cm3",
    "sacrum": "sacrum_cm3",
    "vertebrae": "vertebrae_total_cm3",
    "intervertebral_discs": "intervertebral_discs_cm3",
    "spinal_cord": "spinal_cord_cm3",
    "heart": "heart_cm3",
    "aorta": "aorta_cm3",
    "inferior_vena_cava": "ivc_cm3",
    "portal_vein_and_splenic_vein": "portal_splenic_vein_cm3",
    "iliac_artery_left": "iliac_artery_left_cm3",
    "iliac_artery_right": "iliac_artery_right_cm3",
    "iliac_vena_left": "iliac_vena_left_cm3",
    "iliac_vena_right": "iliac_vena_right_cm3",
    "humerus_left": "humerus_left_cm3",
    "humerus_right": "humerus_right_cm3",
    "scapula_left": "scapula_left_cm3",
    "scapula_right": "scapula_right_cm3",
    "clavicula_left": "clavicula_left_cm3",
    "clavicula_right": "clavicula_right_cm3",
    "femur_left": "femur_left_cm3",
    "femur_right": "femur_right_cm3",
    "hip_left": "hip_left_cm3",
    "hip_right": "hip_right_cm3",
    "gluteus_maximus_left": "gluteus_maximus_left_cm3",
    "gluteus_maximus_right": "gluteus_maximus_right_cm3",
    "gluteus_medius_left": "gluteus_medius_left_cm3",
    "gluteus_medius_right": "gluteus_medius_right_cm3",
    "gluteus_minimus_left": "gluteus_minimus_left_cm3",
    "gluteus_minimus_right": "gluteus_minimus_right_cm3",
    "autochthon_left": "autochthon_left_cm3",
    "autochthon_right": "autochthon_right_cm3",
    "iliopsoas_left": "iliopsoas_left_cm3",
    "iliopsoas_right": "iliopsoas_right_cm3",
    "brain": "brain_cm3",
}

TOTALSEG_HEARTCHAMBERS_VOLUME_KEYS: dict[str, str] = {
    "heart_ventricle_left": "lv_cm3",
    "heart_ventricle_right": "rv_cm3",
    "heart_atrium_left": "la_cm3",
    "heart_atrium_right": "ra_cm3",
    "aorta": "aorta_cm3",
}


def map_organ_key(stem: str, task: str) -> str | None:
    """Map TotalSeg mask filename stem to canonical pathology_scores key."""
    if task == "total":
        return TOTALSEG_TOTAL_VOLUME_KEYS.get(stem)
    if task == "total_mr":
        return TOTALSEG_TOTAL_MR_VOLUME_KEYS.get(stem)
    if task == "heartchambers":
        return TOTALSEG_HEARTCHAMBERS_VOLUME_KEYS.get(stem)
    if task in ("vertebrae_body", "vertebrae_mr"):
        if stem.startswith("vertebrae_") or stem == "sacrum":
            return f"{stem}_cm3"
        return None
    return None
