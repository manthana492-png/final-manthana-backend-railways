"""Shared VISTA-3D label id → name map (reporting subset)."""

# Canonical subset used for reporting; segmentation mask still carries raw class ids.
LABEL_DICT: dict[int, str] = {
    1: "liver",
    2: "right_lung",
    3: "spleen",
    4: "pancreas",
    5: "right_kidney",
    6: "aorta",
    7: "inferior_vena_cava",
    8: "portal_vein",
    9: "left_lung",
    10: "left_kidney",
    22: "brain",
    30: "urinary_bladder",
    35: "prostate_or_uterus",
    40: "heart",
    48: "spinal_canal",
    58: "colon",
    66: "small_bowel",
    77: "thoracic_aorta",
    88: "femur_right",
    89: "femur_left",
    96: "vertebra_l5",
    110: "vertebra_t12",
    127: "miscellaneous_target",
}
