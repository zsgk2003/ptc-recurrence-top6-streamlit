"""
Feature definitions and the label-encoding scheme of the Top-6 LightGBM model.

Encoding follows Table 1 of the source study (see Experimental_Methods.md); the
same integer codes are used in `modeldata_335_PTC.csv`.
"""

FEATURE_NAMES = ["Age", "Physical Examination", "Adenopathy", "T", "N", "Response"]

FEATURE_LABELS = {
    "Age": "Age (years)",
    "Physical Examination": "Physical Examination",
    "Adenopathy": "Adenopathy",
    "T": "T stage (primary tumour)",
    "N": "N stage (regional nodes)",
    "Response": "Treatment Response",
}

FEATURE_HELP = {
    "Age": "Age at diagnosis, in years. Continuous variable, used unscaled.",
    "Physical Examination": "Thyroid morphology on neck palpation / physical examination.",
    "Adenopathy": "Site of cervical lymph-node enlargement on clinical or imaging assessment.",
    "T": "AJCC / TNM stage of the primary tumour.",
    "N": "AJCC / TNM stage of the regional lymph nodes.",
    "Response": "ATA response-to-therapy assessment - the strongest predictor in this model.",
}

# Ordered option label -> integer code
CATEGORICAL_OPTIONS = {
    "Physical Examination": {
        "Diffuse goiter": 0,
        "Multinodular goiter": 1,
        "Normal": 2,
        "Single nodular goiter - left": 3,
        "Single nodular goiter - right": 4,
    },
    "Adenopathy": {
        "Bilateral": 0,
        "Extensive": 1,
        "Left": 2,
        "No": 3,
        "Posterior": 4,
        "Right": 5,
    },
    "T": {
        "T1a": 0,
        "T1b": 1,
        "T2": 2,
        "T3a": 3,
        "T3b": 4,
        "T4a": 5,
        "T4b": 6,
    },
    "N": {
        "N0": 0,
        "N1a": 1,
        "N1b": 2,
    },
    "Response": {
        "Excellent": 0,
        "Indeterminate": 1,
        "Structural Incomplete": 2,
        "Biochemical Incomplete": 3,
    },
}

# Default selection per categorical feature (most common category in the cohort)
DEFAULT_OPTION_INDEX = {
    "Physical Examination": 2,
    "Adenopathy": 3,
    "T": 0,
    "N": 0,
    "Response": 0,
}

CODE_TO_LABEL = {
    feature: {code: label for label, code in options.items()}
    for feature, options in CATEGORICAL_OPTIONS.items()
}

AGE_RANGE = (15, 90)
AGE_DEFAULT = 45


def display_value(feature: str, code) -> str:
    """Human-readable value for a single encoded feature value."""
    if feature == "Age":
        return str(code)
    return CODE_TO_LABEL[feature].get(code, str(code))


def encoding_table() -> list[dict]:
    """Flat description of the encoding scheme, for the About page."""
    rows = [{
        "Feature": FEATURE_LABELS["Age"],
        "Type": "Continuous",
        "Encoding": f"{AGE_RANGE[0]}-{AGE_RANGE[1]} years (raw value)",
    }]
    for feature in FEATURE_NAMES[1:]:
        options = CATEGORICAL_OPTIONS[feature]
        rows.append({
            "Feature": FEATURE_LABELS[feature],
            "Type": f"Categorical ({len(options)} levels)",
            "Encoding": ", ".join(f"{code}={label}" for label, code in options.items()),
        })
    return rows
