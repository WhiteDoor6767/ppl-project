"""
coffee_utils.py
---------------
Modular cleaning, visualization, and statistical analysis module for the
Arabica Coffee Quality project (CSBS Python Programming Laboratory).
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.graphics.gofplots import qqplot


# =============================================================================
# PART 1: DATA CLEANING & PREPROCESSING
# =============================================================================

def load_raw(path: str) -> pd.DataFrame:
    """Load raw Arabica coffee ratings CSV."""
    # Fallback to latin-1 if utf-8 decoding fails on scraped text
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty scraper artifacts and uninformative columns."""
    df = df.copy()
    # Drop scraper artifacts, 100% empty columns, and table header leftovers
    junk_cols = [
        "Unnamed: 0", "Unnamed: 51", "view_certificate_1", "view_certificate_2",
        "Cupping Protocol and Descriptors", "View Green Analysis Details",
        "Request a Sample", "NA", "NA.1", "NA.2", "NA.3", "Notes",
    ]
    return df.drop(columns=[c for c in junk_cols if c in df.columns], errors="ignore")


def clean_cup_points(df: pd.DataFrame, col: str = "Total Cup Points") -> pd.DataFrame:
    """Convert Total Cup Points to numeric, stripping scraper prefix."""
    df = df.copy()
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    # Strip 'Sample' prefix and parse to float
    clean_series = df[col].astype(str).str.replace("Sample", "", case=False, regex=False).str.strip()
    df["total_cup_points"] = pd.to_numeric(clean_series, errors="coerce")
    return df


def _parse_altitude_string(raw_val: Any) -> float:
    """Helper to parse complex text altitude strings into meters."""
    if pd.isna(raw_val):
        return np.nan
    s = str(raw_val).strip()
    if not s:
        return np.nan

    # Exclude non-numeric placeholder text
    junk_placeholders = {"-", "-(average)", "-1", "0", "1", "test", "mmm", "none", "unkown", "huanuco"}
    if s.lower() in junk_placeholders:
        return np.nan

    # Fix OCR typos where letter 'O' was entered instead of digit '0'
    s = re.sub(r"([0-9])OO\b", r"\g<1>00", s)
    s = re.sub(r"([0-9])O\b", r"\g<1>0", s)

    # Resolve confirmed data-entry anomalies
    if s == "190164":        # Lost decimal point in scraping (sister lot 1144 is 1901.64m)
        return 1901.64
    if s == "11000 metros":  # Typo for 1100m (Fazenda Pantano, Cerrado Mineiro)
        return 1100.0
    if s == "16.8":          # Typo for 1680m (Mamsera, Kilimanjaro slope)
        return 1680.0
    if s == "100/2000":
        return 1050.0

    # Detect explicit feet units (pies/psnm in Spanish entries)
    is_feet = bool(re.search(r"(ft|feet|pies|psnm|p\.s\.n\.m|psn|\bf\b)", s, re.IGNORECASE))

    # Clean European thousand separators (e.g. 1.872 -> 1872) and punctuation
    s_clean = re.sub(r"(?<=\b[1-2])\.(\d{3})\b", r"\1", s)
    s_clean = s_clean.replace("'", "").replace(",", "")

    tokens = re.findall(r"\d+\.?\d*", s_clean)
    if not tokens:
        return np.nan
    nums = [float(t) for t in tokens]

    # For ranges (e.g. 1200-1800), take the arithmetic midpoint
    val = (nums[0] + nums[1]) / 2.0 if len(nums) >= 2 else nums[0]

    # Convert km to meters if entered as decimal (e.g. 1.2 km -> 1200m)
    if 0.5 < val < 3.0 and not is_feet:
        val = val * 1000.0
    # Central American farms traditionally grade in feet (3000-8000 ft) -> convert to meters
    elif is_feet or (3000.0 <= val <= 8500.0):
        val = val * 0.3048

    return float(val)


def clean_altitude(
    df: pd.DataFrame,
    col: str = "Altitude",
    min_meters: float = 100.0,
    max_meters: float = 3000.0,
) -> pd.DataFrame:
    """Parse raw Altitude into clean elevation in meters bounded to plausible ranges."""
    df = df.copy()
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    parsed = df[col].apply(_parse_altitude_string)
    # Filter to biological Arabica growing limits (100m - 3000m)
    valid_mask = (parsed >= min_meters) & (parsed <= max_meters)
    df["altitude_m"] = parsed.where(valid_mask, np.nan).round(1)
    return df


def clean_physical_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Strip text suffixes from moisture and defect columns."""
    df = df.copy()
    # Extract numeric values from strings like '12 %' and '0 full defects'
    if "Moisture" in df.columns:
        df["moisture_pct"] = pd.to_numeric(
            df["Moisture"].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
        )
    if "Category One Defects" in df.columns:
        df["cat_one_defects"] = pd.to_numeric(
            df["Category One Defects"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
    if "Category Two Defects" in df.columns:
        df["cat_two_defects"] = pd.to_numeric(
            df["Category Two Defects"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
    if "Quakers" in df.columns:
        df["quakers"] = pd.to_numeric(df["Quakers"], errors="coerce").fillna(0)
    return df


def standardize_categories(
    df: pd.DataFrame, top_varieties: int = 8, min_country_samples: int = 10
) -> pd.DataFrame:
    """Standardize categories and group low-frequency levels into 'Other'."""
    df = df.copy()

    # 1. Clean country names
    if "Country of Origin" in df.columns:
        df["country"] = df["Country of Origin"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
        df["country"] = df["country"].replace({
            "Tanzania, United Republic Of": "Tanzania",
            "United States (Hawaii)": "USA (Hawaii)",
            "Cote d?Ivoire": "Ivory Coast",
        })
        # Group origins with fewer than min_country_samples lots into 'Other Origin'
        counts = df["country"].value_counts(dropna=True)
        rare_origins = counts[counts < min_country_samples].index
        df["country_clean"] = df["country"].replace({c: "Other Origin" for c in rare_origins})

    # 2. Standardize processing method names
    if "Processing Method" in df.columns:
        pm_map = {
            "Washed / Wet": "Washed",
            "Natural / Dry": "Natural",
            "Semi-washed / Semi-pulped": "Semi-Washed",
            "Pulped natural / honey": "Honey / Pulped Natural",
            "Other": "Other Method",
        }
        pm_series = df["Processing Method"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
        df["processing_method_clean"] = pm_series.map(lambda x: pm_map.get(x, x) if pd.notna(x) else np.nan)

    # 3. Standardize variety (top 8 distinct + Other)
    if "Variety" in df.columns:
        var_series = df["Variety"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
        valid_vars = var_series[var_series.notna() & (var_series != "Other")]
        top_names = valid_vars.value_counts().head(top_varieties).index.tolist()
        df["variety_clean"] = var_series.apply(
            lambda x: x if (pd.isna(x) or x in top_names) else "Other Variety"
        )

    return df


def handle_missing(df: pd.DataFrame, categorical_strategy: str = "unknown") -> pd.DataFrame:
    """Handle missing values via imputation or listwise deletion."""
    df = df.copy()
    # Target score is required
    if "total_cup_points" in df.columns:
        df = df.dropna(subset=["total_cup_points"])
    # Drop single record missing country (row 1197 has no data)
    if "country" in df.columns:
        df = df[df["country"].notna() & (df["country"] != "nan")]

    if categorical_strategy == "unknown":
        # Label missing categories as Unknown to preserve sample size for univariate tests
        if "processing_method_clean" in df.columns:
            df["processing_method_clean"] = df["processing_method_clean"].fillna("Unknown")
        if "variety_clean" in df.columns:
            df["variety_clean"] = df["variety_clean"].fillna("Unknown")
    elif categorical_strategy == "drop":
        # Complete-case analysis for regression
        cols = [c for c in ["processing_method_clean", "variety_clean", "altitude_m", "country_clean"] if c in df.columns]
        df = df.dropna(subset=cols)
    else:
        raise ValueError("Strategy must be 'unknown' or 'drop'")
    return df


def flag_outliers(
    df: pd.DataFrame,
    column: str = "total_cup_points",
    drop_corrupted: bool = True,
    min_valid_score: float = 60.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Flag IQR outliers and drop corrupted lots (< 60 points)."""
    df = df.copy()
    series = df[column].dropna()
    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    stats_dict = {
        "q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2), "upper_bound": round(upper_bound, 2),
        "total_count": len(series),
        "low_outlier_count": int((series < lower_bound).sum()),
        "high_outlier_count": int((series > upper_bound).sum()),
    }
    df["is_outlier"] = (df[column] < lower_bound) | (df[column] > upper_bound)

    # Drop corrupted entries (e.g. 0.00 and 43.13) to prevent distortion
    if drop_corrupted:
        corrupt = df[column] < min_valid_score
        stats_dict["corrupted_count"] = int(corrupt.sum())
        df = df[~corrupt]

    return df, stats_dict


def get_clean_dataset(
    path: str,
    categorical_strategy: str = "unknown",
    save_clean_path: Optional[str] = None,
) -> pd.DataFrame:
    """Orchestrate the full end-to-end cleaning pipeline."""
    df = load_raw(path)
    # Filter strictly to Arabica species
    if "Species" in df.columns:
        df = df[df["Species"].astype(str).str.strip().str.lower() == "arabica"]

    df = drop_junk_columns(df)
    df = clean_cup_points(df, col="Total Cup Points")
    df = clean_altitude(df, col="Altitude")
    df = clean_physical_attributes(df)
    df = standardize_categories(df, top_varieties=8, min_country_samples=10)
    df, _ = flag_outliers(df, column="total_cup_points", drop_corrupted=True, min_valid_score=60.0)
    df = handle_missing(df, categorical_strategy=categorical_strategy)

    # Standardize column naming to snake_case
    rename_map = {
        "Country of Origin": "country_raw", "Farm Name": "farm_name",
        "Lot Number": "lot_number", "Mill": "mill", "ICO Number": "ico_number",
        "Company": "company", "Region": "region", "Producer": "producer",
        "Number of Bags": "number_of_bags", "Bag Weight": "bag_weight",
        "In-Country Partner": "in_country_partner", "Harvest Year": "harvest_year",
        "Grading Date": "grading_date", "Variety": "variety_raw",
        "Processing Method": "processing_method_raw", "Aroma": "aroma",
        "Flavor": "flavor", "Aftertaste": "aftertaste", "Acidity": "acidity",
        "Body": "body", "Balance": "balance", "Uniformity": "uniformity",
        "Clean Cup": "clean_cup", "Sweetness": "sweetness",
        "Cupper Points": "cupper_points", "Moisture": "moisture_raw",
        "Category One Defects": "cat_one_defects_raw",
        "Category Two Defects": "cat_two_defects_raw", "Color": "color",
        "Expiration": "expiration", "Certification Body": "certification_body",
    }
    df = df.rename(columns=rename_map)

    # Reorder primary columns to front
    lead_cols = [
        "total_cup_points", "altitude_m", "processing_method_clean",
        "variety_clean", "country_clean", "moisture_pct",
        "cat_one_defects", "cat_two_defects", "is_outlier"
    ]
    df = df[lead_cols + [c for c in df.columns if c not in lead_cols]]

    if save_clean_path:
        df.to_csv(save_clean_path, index=False)
    return df


# =============================================================================
# PART 2: EXPLORATORY DATA VISUALIZATION
# =============================================================================

def set_plot_style():
    """Configure plot aesthetics."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#eeeeee"
    plt.rcParams["grid.linestyle"] = "--"


def plot_cup_points_distribution(df: pd.DataFrame, save_path: str):
    """Plot 1: Distribution of Total Cup Points (Histogram + KDE)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    pts = df["total_cup_points"].dropna()
    mean_val, median_val = pts.mean(), pts.median()
    q25, q75 = pts.quantile(0.25), pts.quantile(0.75)

    # Histogram and KDE
    sns.histplot(pts, kde=True, bins=35, color="#8B5A2B", edgecolor="#4A2E12", alpha=0.6, ax=ax, stat="density")

    # SCA specialty thresholds
    ax.axvline(80.0, color="#D9534F", linestyle="--", linewidth=2, label="SCA Specialty Threshold (80.0)")
    ax.axvline(85.0, color="#2E8B57", linestyle="--", linewidth=2, label="SCA Excellent Specialty (85.0)")
    ax.axvline(mean_val, color="#2B6CB0", linestyle="-", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="#4A5568", linestyle=":", linewidth=2, label=f"Median: {median_val:.2f}")
    ax.axvspan(80.0, pts.max() + 1, alpha=0.08, color="#2E8B57")

    stats_text = (
        f"Sample Size (n): {len(pts):,}\n"
        f"Mean: {mean_val:.2f} | Std: {pts.std():.2f}\n"
        f"Median: {median_val:.2f} | IQR: [{q25:.2f}, {q75:.2f}]\n"
        f"Skewness: {pts.skew():.2f} (Left-Skewed)"
    )
    ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E0", alpha=0.9))

    ax.set_title("Figure 1: Distribution of Total Cup Points for Arabica Coffee", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Total Cup Points (SCA Scale: 0–100)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax.set_xlim(60, 95)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_processing_methods(df: pd.DataFrame, save_path: str):
    """Plot 2: Box & Strip Plot of Total Cup Points by Processing Method."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    subset = df[df["processing_method_clean"].notna() & (df["processing_method_clean"] != "Unknown")].copy()
    order = subset.groupby("processing_method_clean")["total_cup_points"].median().sort_values(ascending=False).index
    palette = sns.color_palette("YlOrBr_r", n_colors=len(order))

    # Boxplot with overlaid jittered points
    sns.boxplot(data=subset, x="processing_method_clean", y="total_cup_points", hue="processing_method_clean",
                order=order, palette=palette, legend=False, boxprops=dict(alpha=0.75), showmeans=True,
                meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6), ax=ax)
    sns.stripplot(data=subset, x="processing_method_clean", y="total_cup_points", order=order,
                  color="#2D3748", alpha=0.35, jitter=0.2, size=4, ax=ax)

    counts = subset["processing_method_clean"].value_counts()
    medians = subset.groupby("processing_method_clean")["total_cup_points"].median()
    labels = [f"{cat}\n(n={counts[cat]}, med={medians[cat]:.2f})" for cat in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_title("Figure 2: Arabica Cup Quality Across Post-Harvest Processing Methods", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Processing Method", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Cup Points", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_varieties(df: pd.DataFrame, save_path: str):
    """Plot 3: Boxplot of Total Cup Points by Arabica Variety."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    subset = df[df["variety_clean"].notna() & (df["variety_clean"] != "Unknown")].copy()
    order = subset.groupby("variety_clean")["total_cup_points"].median().sort_values(ascending=False).index
    palette = sns.color_palette("mako", n_colors=len(order))

    sns.boxplot(data=subset, x="variety_clean", y="total_cup_points", hue="variety_clean",
                order=order, palette=palette, legend=False, boxprops=dict(alpha=0.75), showmeans=True,
                meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=6), ax=ax)
    sns.stripplot(data=subset, x="variety_clean", y="total_cup_points", order=order,
                  color="#1A202C", alpha=0.3, jitter=0.2, size=3.5, ax=ax)

    counts = subset["variety_clean"].value_counts()
    labels = [f"{cat}\n(n={counts[cat]})" for cat in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=9.5, rotation=15)

    ax.set_title("Figure 3: Cup Points Distribution Across Major Arabica Cultivars", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Botanical Variety", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Cup Points", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_countries(df: pd.DataFrame, save_path: str, top_n: int = 12):
    """Plot 4: Horizontal Bar Chart of Average Cup Points by Country of Origin."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # Filter to origins with >= 15 lots
    counts = df["country"].value_counts()
    valid_countries = counts[counts >= 15].head(top_n).index
    subset = df[df["country"].isin(valid_countries)].copy()

    means = subset.groupby("country")["total_cup_points"].mean().sort_values(ascending=True)
    order = means.index

    # Horizontal bar plot with 95% bootstrap confidence intervals
    sns.barplot(data=subset, y="country", x="total_cup_points", hue="country",
                order=order, palette="copper", legend=False, errorbar=("ci", 95),
                capsize=0.2, err_kws={"linewidth": 1.2, "color": "#2D3748"}, ax=ax)

    for i, country in enumerate(order):
        ax.text(means[country] - 1.2, i, f"{means[country]:.2f} (n={counts[country]})",
                va="center", ha="right", fontsize=9.5, fontweight="bold", color="white")

    global_mean = df["total_cup_points"].mean()
    ax.axvline(global_mean, color="#C53030", linestyle="--", linewidth=1.8, label=f"Dataset Mean: {global_mean:.2f}")

    ax.set_title("Figure 4: Mean Cup Quality by Country of Origin (Origins with ≥15 Lots)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Mean Total Cup Points (with 95% CI Error Bars)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Country of Origin", fontsize=11, fontweight="bold")
    ax.set_xlim(78, 86.5)
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_altitude_regression(df: pd.DataFrame, save_path: str):
    """Plot 5: Scatter Plot of Altitude vs. Total Cup Points with OLS Trend Line."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    subset = df[df["altitude_m"].notna() & df["total_cup_points"].notna()].copy()
    x, y = subset["altitude_m"], subset["total_cup_points"]
    r_val = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)

    # OLS regression scatter and confidence band
    sns.regplot(data=subset, x="altitude_m", y="total_cup_points",
                scatter_kws={"alpha": 0.4, "color": "#795548", "s": 25},
                line_kws={"color": "#C53030", "linewidth": 2.2, "label": f"OLS Fit: y = {slope:.4f}x + {intercept:.2f}"},
                ax=ax)

    ann_text = (
        f"Valid Lots (n): {len(subset):,}\n"
        f"Pearson r: +{r_val:.3f} (p < 0.001)\n"
        f"Slope: +{slope*1000:.2f} pts / 1,000m\n"
        f"Elevation Range: 100m – 2,560m"
    )
    ax.text(0.04, 0.94, ann_text, transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E0", alpha=0.9))

    ax.set_title("Figure 5: Relationship Between Growing Altitude and Arabica Cup Quality", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Growing Altitude (meters above sea level)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Cup Points (SCA Scale)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    """Plot 6: Triangular Correlation Heatmap of Numeric Variables."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    feature_cols = [
        "total_cup_points", "altitude_m", "moisture_pct", "cat_one_defects",
        "cat_two_defects", "acidity", "flavor", "body", "balance"
    ]
    labels_map = {
        "total_cup_points": "Total Cup Points", "altitude_m": "Altitude (m)",
        "moisture_pct": "Moisture (%)", "cat_one_defects": "Cat 1 Defects",
        "cat_two_defects": "Cat 2 Defects", "acidity": "Acidity Score",
        "flavor": "Flavor Score", "body": "Body Score", "balance": "Balance Score",
    }

    existing = [c for c in feature_cols if c in df.columns]
    corr_df = df[existing].rename(columns=labels_map).corr()

    # Mask upper triangle to avoid redundant mirrored values
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-0.5, vmax=1.0, square=True, linewidths=0.6,
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"}, ax=ax)

    ax.set_title("Figure 6: Correlation Heatmap Across Growing Conditions, Defects & Quality", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_all_visualizations(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """Generate and save all 6 project visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "cup_points_distribution": os.path.join(output_dir, "01_cup_points_distribution.png"),
        "processing_methods": os.path.join(output_dir, "02_cup_points_by_processing_method.png"),
        "varieties": os.path.join(output_dir, "03_cup_points_by_variety.png"),
        "countries": os.path.join(output_dir, "04_cup_points_by_country.png"),
        "altitude_regression": os.path.join(output_dir, "05_altitude_vs_cup_points.png"),
        "correlation_heatmap": os.path.join(output_dir, "06_correlation_heatmap.png"),
    }
    plot_cup_points_distribution(df, paths["cup_points_distribution"])
    plot_processing_methods(df, paths["processing_methods"])
    plot_varieties(df, paths["varieties"])
    plot_countries(df, paths["countries"])
    plot_altitude_regression(df, paths["altitude_regression"])
    plot_correlation_heatmap(df, paths["correlation_heatmap"])
    return paths


# =============================================================================
# PART 3: STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# =============================================================================

def run_normality_and_variance_tests(
    df: pd.DataFrame, group_col: str, target_col: str = "total_cup_points"
) -> Dict[str, Any]:
    """Test group normality (Shapiro-Wilk) and equal variances (Levene's test)."""
    subset = df[df[group_col].notna() & (df[group_col] != "Unknown") & df[target_col].notna()].copy()
    groups = {name: grp[target_col].values for name, grp in subset.groupby(group_col) if len(grp) >= 3}

    shapiro_results = {}
    any_non_normal = False
    for name, vals in groups.items():
        w_stat, p_val = stats.shapiro(vals)
        is_normal = p_val >= 0.05
        if not is_normal:
            any_non_normal = True
        shapiro_results[name] = {
            "n": len(vals),
            "shapiro_w": round(float(w_stat), 4),
            "p_value": float(p_val),
            "is_normal": is_normal,
        }

    # Test equal variance across all groups
    levene_stat, levene_p = stats.levene(*groups.values())
    equal_variances = levene_p >= 0.05

    # Recommend test based on assumption checks
    if any_non_normal or not equal_variances:
        recommended = "Kruskal-Wallis"
        justification = "Assumptions violated: groups reject normality (p < 0.05) or equal variances (Levene p < 0.05)."
    else:
        recommended = "One-Way ANOVA"
        justification = "Groups meet normality (p >= 0.05) and equal variances (Levene p >= 0.05)."

    return {
        "group_col": group_col,
        "shapiro_by_group": shapiro_results,
        "levene_stat": round(float(levene_stat), 4),
        "levene_p": float(levene_p),
        "equal_variances": equal_variances,
        "recommended_test": recommended,
        "justification": justification,
    }


def run_group_hypothesis_tests(
    df: pd.DataFrame, group_col: str, target_col: str = "total_cup_points"
) -> Dict[str, Any]:
    """Execute both One-Way ANOVA and Kruskal-Wallis tests."""
    subset = df[df[group_col].notna() & (df[group_col] != "Unknown") & df[target_col].notna()].copy()
    groups = [grp[target_col].values for _, grp in subset.groupby(group_col) if len(grp) >= 3]

    f_stat, anova_p = stats.f_oneway(*groups)
    h_stat, kw_p = stats.kruskal(*groups)

    return {
        "group_col": group_col,
        "num_groups": len(groups),
        "total_observations": sum(len(g) for g in groups),
        "anova_f": round(float(f_stat), 4),
        "anova_p": float(anova_p),
        "anova_significant": anova_p < 0.05,
        "kruskal_h": round(float(h_stat), 4),
        "kruskal_p": float(kw_p),
        "kruskal_significant": kw_p < 0.05,
    }


def run_posthoc_tests(
    df: pd.DataFrame, group_col: str, target_col: str = "total_cup_points"
) -> Dict[str, Any]:
    """Perform Tukey HSD and Pairwise Mann-Whitney with Holm correction."""
    subset = df[df[group_col].notna() & (df[group_col] != "Unknown") & df[target_col].notna()].copy()
    valid_groups = [name for name, grp in subset.groupby(group_col) if len(grp) >= 3]
    subset = subset[subset[group_col].isin(valid_groups)].copy()

    # 1. Parametric Tukey HSD
    tukey = pairwise_tukeyhsd(endog=subset[target_col], groups=subset[group_col], alpha=0.05)
    tukey_table = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    tukey_sig = tukey_table[tukey_table["reject"] == True]

    # 2. Non-parametric Pairwise Mann-Whitney with Holm correction
    unique_groups = sorted(subset[group_col].unique())
    pairs, p_values = [], []
    for i in range(len(unique_groups)):
        for j in range(i + 1, len(unique_groups)):
            g1, g2 = unique_groups[i], unique_groups[j]
            s1 = subset[subset[group_col] == g1][target_col]
            s2 = subset[subset[group_col] == g2][target_col]
            _, p_val = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            pairs.append((g1, g2))
            p_values.append(p_val)

    reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="holm")

    mann_whitney_results = []
    for (g1, g2), p_raw, p_adj, rej in zip(pairs, p_values, p_adjusted, reject):
        mann_whitney_results.append({
            "group1": g1, "group2": g2,
            "p_raw": float(p_raw), "p_adj": float(p_adj),
            "significant": bool(rej),
        })

    mwu_df = pd.DataFrame(mann_whitney_results)
    mwu_sig = mwu_df[mwu_df["significant"] == True]

    return {
        "group_col": group_col,
        "tukey_summary": tukey_table,
        "tukey_significant": tukey_sig,
        "mann_whitney_summary": mwu_df,
        "mann_whitney_significant": mwu_sig,
    }


def fit_multiple_regression(
    df: pd.DataFrame,
    formula: Optional[str] = None,
    cov_type: str = "HC3",
) -> Dict[str, Any]:
    """Fit multiple linear regression with Type II ANOVA decomposition and HC3 robust SE."""
    reg_df = df.dropna(
        subset=["total_cup_points", "altitude_m", "processing_method_clean", "variety_clean", "country_clean"]
    ).copy()

    if formula is None:
        # Washed, Bourbon, and Colombia as baseline reference levels
        formula = (
            "total_cup_points ~ altitude_m + "
            'C(processing_method_clean, Treatment(reference="Washed")) + '
            'C(variety_clean, Treatment(reference="Bourbon")) + '
            'C(country_clean, Treatment(reference="Colombia"))'
        )

    # Fit standard and HC3 robust models
    ols_model = smf.ols(formula, data=reg_df).fit()
    robust_model = smf.ols(formula, data=reg_df).fit(cov_type=cov_type)

    # Type II ANOVA to decompose unique variance explained by factor
    anova_table = anova_lm(ols_model, typ=2)
    residual_ss = anova_table.loc["Residual", "sum_sq"]
    anova_table["partial_eta_sq"] = anova_table["sum_sq"] / (anova_table["sum_sq"] + residual_ss)
    anova_table["pct_model_variance"] = (anova_table["sum_sq"] / (ols_model.ess)) * 100

    # Residual diagnostic tests
    bp_stat, bp_pvalue, _, _ = sms.het_breuschpagan(ols_model.resid, ols_model.model.exog)
    jb_stat, jb_pvalue, skew, kurtosis = sms.jarque_bera(ols_model.resid)

    coef_df = pd.DataFrame({
        "coef": ols_model.params,
        "std_err": robust_model.bse,
        "t_stat": robust_model.tvalues,
        "p_val_robust": robust_model.pvalues,
        "ci_lower": robust_model.conf_int()[0],
        "ci_upper": robust_model.conf_int()[1],
    })
    sig_coefs = coef_df[coef_df["p_val_robust"] < 0.05].copy()

    return {
        "n_obs": int(ols_model.nobs),
        "r_squared": round(float(ols_model.rsquared), 4),
        "adj_r_squared": round(float(ols_model.rsquared_adj), 4),
        "f_stat": round(float(ols_model.fvalue), 2),
        "f_pvalue": float(ols_model.f_pvalue),
        "ols_model": ols_model,
        "robust_model": robust_model,
        "anova_table": anova_table,
        "coefficients_all": coef_df,
        "coefficients_significant": sig_coefs,
        "diagnostics": {
            "breusch_pagan_stat": round(float(bp_stat), 4),
            "breusch_pagan_p": float(bp_pvalue),
            "is_heteroscedastic": bp_pvalue < 0.05,
            "jarque_bera_stat": round(float(jb_stat), 4),
            "jarque_bera_p": float(jb_pvalue),
            "residual_skew": round(float(skew), 4),
            "residual_kurtosis": round(float(kurtosis), 4),
        },
    }


def plot_regression_diagnostics(model_result: Any, save_path: str):
    """Generate 2-panel regression diagnostic plot (Residuals vs Fitted and Normal Q-Q)."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    fitted = model_result.fittedvalues
    resid = model_result.resid

    # Panel A: Residuals vs Fitted values
    sns.residplot(x=fitted, y=resid, lowess=True,
                  scatter_kws={"alpha": 0.4, "color": "#4A5568", "s": 25},
                  line_kws={"color": "#C53030", "linewidth": 2}, ax=ax1)
    ax1.axhline(0, color="#CBD5E0", linestyle="--", linewidth=1.5)
    ax1.set_title("Panel A: Residuals vs. Fitted Values", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fitted Total Cup Points", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Residuals", fontsize=10, fontweight="bold")

    # Panel B: Normal Q-Q Plot
    qqplot(resid, line="45", fit=True, ax=ax2, markerfacecolor="#4A5568", markeredgecolor="#2D3748", alpha=0.4)
    ax2.set_title("Panel B: Normal Q-Q Plot of Residuals", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Theoretical Quantiles", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Standardized Residuals", fontsize=10, fontweight="bold")

    plt.suptitle("Figure 7: Multiple Linear Regression Diagnostic Plots", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
