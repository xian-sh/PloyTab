"""Shared plotting style for PolyTab figures."""

SOFT_FRAME_COLOR = "#6B7280"

MODEL_COLORS = {
    "BLR": "#4E79A7",
    "Bayesian": "#4E79A7",
    "ERT": "#59A14F",
    "ETR": "#59A14F",
    "TabPFN": "#F28E2B",
    "PolyTab": "#E15759",
    "TransTab": "#E15759",
    "TransTab (multi)": "#B07AA1",
}

DISPLAY_NAMES = {
    "BLR": "BLR",
    "Bayesian": "BLR",
    "ERT": "ERT",
    "ETR": "ERT",
    "TabPFN": "TabPFN",
    "PolyTab": "PolyTab",
    "TransTab": "PolyTab",
    "TransTab (multi)": "PolyTab (multi)",
}


def get_model_color(name: str) -> str:
    return MODEL_COLORS.get(name, "#9CA3AF")


def get_display_model_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)
