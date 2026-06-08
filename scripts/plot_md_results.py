from pathlib import Path
import sys

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from polytab.visual_style import SOFT_FRAME_COLOR, get_display_model_name, get_model_color


def get_arial_font() -> FontProperties:
    local_font = PROJECT_ROOT / "assets" / "fonts" / "Arial.ttf"
    if local_font.exists():
        font_manager.fontManager.addfont(str(local_font))
        return FontProperties(fname=str(local_font))

    arial_candidates = []
    for font_path in font_manager.findSystemFonts(fontpaths=None, fontext="ttf"):
        if "arial" in Path(font_path).name.lower():
            arial_candidates.append(font_path)

    if arial_candidates:
        chosen_font = arial_candidates[0]
        font_manager.fontManager.addfont(chosen_font)
        return FontProperties(fname=chosen_font)

    return FontProperties(family="DejaVu Sans")


arial_prop = get_arial_font()
arial_name = arial_prop.get_name()

plt.rcParams["font.family"] = arial_name
plt.rcParams["font.sans-serif"] = [arial_name]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"


BASE_DIR = PROJECT_ROOT
FIGURE_DIR = PROJECT_ROOT / "figures"

colors = {
    "BLR": get_model_color("BLR"),
    "ETR": get_model_color("ERT"),
    "TabPFN": get_model_color("TabPFN"),
    "PolyTab": get_model_color("PolyTab"),
}
display_name_map = {name: get_display_model_name(name) for name in colors}

ordered_mapping = [
    ("density", "ρ"),
    ("Rg", "Rg"),
    ("Scaled Rg", "sRg"),
    ("r2", "r2"),
    ("Cp", "Cp"),
    ("Cv", "Cv"),
    ("bulk_modulus", "K"),
    ("isentropic_bulk_modulus", "Ks"),
    ("static_dielectric_const", "εs"),
    ("dielectric_const_dc", "εdc"),
    ("nematic_order_parameter", "S"),
    ("refractive_index", "n"),
]
x_labels = [item[1] for item in ordered_mapping]

# Use portable ASCII labels for repository figures.
ordered_mapping = [
    ("density", "rho"),
    ("Rg", "Rg"),
    ("Scaled Rg", "sRg"),
    ("r2", "r2"),
    ("Cp", "Cp"),
    ("Cv", "Cv"),
    ("bulk_modulus", "K"),
    ("isentropic_bulk_modulus", "Ks"),
    ("static_dielectric_const", "eps_s"),
    ("dielectric_const_dc", "eps_dc"),
    ("nematic_order_parameter", "S"),
    ("refractive_index", "n"),
]
x_labels = [item[1] for item in ordered_mapping]


def get_aligned_r2_data(file_path: Path, mapping_list):
    target_len = len(mapping_list)

    if not file_path or not Path(file_path).exists():
        return [0.0] * target_len

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        feature_col = None
        r2_col = None

        for col in df.columns:
            if df[col].astype(str).str.contains("density").any():
                feature_col = col
                break

        possible_headers = ["R²", "R2", "r2", "R^2", "Correlation"]
        for header in possible_headers:
            if header in df.columns:
                r2_col = header
                break

        if not r2_col:
            for col in df.columns:
                if "R2" in col.upper() or "R²" in col:
                    r2_col = col
                    break

        if not feature_col or not r2_col:
            return [0.0] * target_len

        data_dict = dict(zip(df[feature_col].astype(str).str.strip(), df[r2_col]))
        return [data_dict.get(csv_key, 0.0) for csv_key, _ in mapping_list]

    except Exception as exc:
        print(f"  [Error] Failed to read {file_path}: {exc}")
        return [0.0] * target_len


def plot_results(keep_mode: str):
    print(f"\n>>> Processing: {keep_mode} ...")

    paths = {
        "BLR": BASE_DIR / "results" / f"newmd_Bayesian_{keep_mode}_260107_SAFE" / "test_set_evaluation.csv",
        "ETR": BASE_DIR / "results" / f"newmd_ETR_{keep_mode}_260107_SAFE" / "test_set_evaluation.csv",
        "TabPFN": BASE_DIR / "results" / f"newmd_TabPFN_{keep_mode}_260107_SAFE" / "test_set_evaluation.csv",
        "PolyTab": BASE_DIR / "results" / f"newmd_polyBERT_cascade_{keep_mode}_260107_SAFE" / "test_set_evaluation.csv",
    }

    data_map = {model: get_aligned_r2_data(path, ordered_mapping) for model, path in paths.items()}

    fig, ax = plt.subplots(figsize=(20, 7), dpi=60)

    group_spacing = 1.30
    x = np.arange(len(x_labels)) * group_spacing
    width = 0.24
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, (model_name, model_data) in enumerate(data_map.items()):
        ax.bar(
            x + offsets[i] * width,
            model_data,
            width,
            label=display_name_map.get(model_name, model_name),
            color=colors[model_name],
            edgecolor="white",
            linewidth=0.5,
        )

    legend_font_size = 24

    ax.set_ylabel(
        "R²",
        fontsize=legend_font_size,
        fontweight="normal",
        labelpad=10,
        fontproperties=arial_prop,
    )
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel(
        "R2",
        fontsize=legend_font_size,
        fontweight="normal",
        labelpad=10,
        fontproperties=arial_prop,
    )
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=legend_font_size)

    ax.set_xticks(x)
    ax.set_xticklabels(
        x_labels,
        fontsize=legend_font_size,
        fontweight="normal",
        fontname=arial_name,
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.tick_params(axis="x", length=5, pad=8)

    for label in ax.get_yticklabels():
        label.set_fontname(arial_name)
        label.set_fontsize(legend_font_size)
        label.set_fontweight("normal")

    ax.axhline(0, color=SOFT_FRAME_COLOR, linewidth=1.2, zorder=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    legend_prop = FontProperties(
        fname=arial_prop.get_file(),
        size=legend_font_size,
        weight="bold",
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(data_map),
        frameon=False,
        columnspacing=2.5,
        handletextpad=0.8,
        handlelength=2.8,
        prop=legend_prop,
    )

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIGURE_DIR / f"md_r2_comparison_{keep_mode}.png"
    plt.savefig(out_file, dpi=60, bbox_inches="tight")
    print(f"[{keep_mode}] Saved to {out_file.name}")
    plt.close()


def main():
    target_modes = ["keep1", "keep2", "keep3", "keep4"]
    for mode in target_modes:
        plot_results(mode)


if __name__ == "__main__":
    main()
