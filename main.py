"""
main.py
-------
Pipeline execution and verification script for the Arabica Coffee Quality project.
Executes data cleaning, generates all 7 publication visualizations, and runs
hypothesis testing and multiple regression modeling.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coffee_utils as cu

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "arabica_ratings_raw.csv")
CLEAN_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "arabica_ratings_cleaned.csv")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

def run_pipeline():
    print("=" * 80)
    print("ARABICA COFFEE QUALITY PROJECT — FULL PIPELINE EXECUTION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # PART 1: DATA CLEANING (STEPS 1 & 2)
    # -------------------------------------------------------------------------
    print("\n[PART 1] Running Data Cleaning Pipeline...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}")
    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Run orchestrator cleaning pipeline
    df_clean = cu.get_clean_dataset(
        RAW_DATA_PATH,
        categorical_strategy="unknown",
        save_clean_path=CLEAN_DATA_PATH
    )
    
    print(f"  -> Cleaned dataset shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
    print(f"  -> Saved cleaned file to: {CLEAN_DATA_PATH}")
    
    # -------------------------------------------------------------------------
    # PART 2: EXPLORATORY VISUALIZATION (STEP 3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PART 2] Generating Visualizations (Step 3)...")
    print("=" * 80)
    
    # Generate Figures 1 through 6
    plot_paths = cu.generate_all_visualizations(df_clean, OUTPUTS_DIR)
    for key, path in plot_paths.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  -> Generated: {os.path.basename(path)} ({size_kb:.1f} KB)")
        
    # -------------------------------------------------------------------------
    # PART 3: STATISTICAL ANALYSIS & HYPOTHESIS TESTING (STEP 4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PART 3] Statistical Hypothesis Testing & Regression (Step 4)")
    print("=" * 80)
    
    # 1. Processing Method Tests (Research Question 2)
    print("\n>>> [1. Processing Method: Normality & Homoscedasticity]")
    norm_pm = cu.run_normality_and_variance_tests(df_clean, "processing_method_clean")
    for grp, res in norm_pm["shapiro_by_group"].items():
        print(f"  * {grp:22s} (n={res['n']}): Shapiro W = {res['shapiro_w']:.4f}, p = {res['p_value']:.4e} -> Normal: {res['is_normal']}")
    print(f"  * Levene's Test for Equal Variances: Stat = {norm_pm['levene_stat']:.4f}, p = {norm_pm['levene_p']:.4e} -> Equal: {norm_pm['equal_variances']}")
    print(f"  * Recommended Test: {norm_pm['recommended_test']}")
    
    print("\n>>> [1b. Processing Method: Hypothesis Test Execution]")
    tests_pm = cu.run_group_hypothesis_tests(df_clean, "processing_method_clean")
    print(f"  * One-Way ANOVA F-test: F = {tests_pm['anova_f']:.4f}, p = {tests_pm['anova_p']:.4e} (Significant: {tests_pm['anova_significant']})")
    print(f"  * Kruskal-Wallis H-test: H = {tests_pm['kruskal_h']:.4f}, p = {tests_pm['kruskal_p']:.4e} (Significant: {tests_pm['kruskal_significant']})")
    
    print("\n>>> [1c. Processing Method: Post-Hoc Pairwise Comparisons]")
    posthoc_pm = cu.run_posthoc_tests(df_clean, "processing_method_clean")
    print("  * Non-Parametric Pairwise Mann-Whitney (Holm-Bonferroni Adjusted):")
    for _, row in posthoc_pm["mann_whitney_summary"].iterrows():
        sig_marker = "*** SIGNIFICANT ***" if row["significant"] else "Not Significant"
        print(f"    - {row['group1']:22s} vs {row['group2']:22s}: p_raw = {row['p_raw']:.4f}, p_adj = {row['p_adj']:.4f} | {sig_marker}")
        
    # 2. Botanical Varieties Tests (Research Question 3)
    print("\n" + "-" * 80)
    print(">>> [2. Botanical Varieties: Normality & Homoscedasticity]")
    norm_var = cu.run_normality_and_variance_tests(df_clean, "variety_clean")
    for grp, res in norm_var["shapiro_by_group"].items():
        print(f"  * {grp:16s} (n={res['n']}): Shapiro W = {res['shapiro_w']:.4f}, p = {res['p_value']:.4e} -> Normal: {res['is_normal']}")
    print(f"  * Levene's Test for Equal Variances: Stat = {norm_var['levene_stat']:.4f}, p = {norm_var['levene_p']:.4e} -> Equal: {norm_var['equal_variances']}")
    print(f"  * Recommended Test: {norm_var['recommended_test']}")
    
    print("\n>>> [2b. Botanical Varieties: Hypothesis Test Execution]")
    tests_var = cu.run_group_hypothesis_tests(df_clean, "variety_clean")
    print(f"  * One-Way ANOVA F-test: F = {tests_var['anova_f']:.4f}, p = {tests_var['anova_p']:.4e}")
    print(f"  * Kruskal-Wallis H-test: H = {tests_var['kruskal_h']:.4f}, p = {tests_var['kruskal_p']:.4e}")
    
    print("\n>>> [2c. Botanical Varieties: Significant Post-Hoc Differences (Holm-Adjusted)]")
    posthoc_var = cu.run_posthoc_tests(df_clean, "variety_clean")
    for _, row in posthoc_var["mann_whitney_significant"].iterrows():
        print(f"  * {row['group1']:16s} vs {row['group2']:16s}: p_raw = {row['p_raw']:.4e}, p_adj = {row['p_adj']:.4e}")
        
    # 3. Multiple Linear Regression & Variance Decomposition (Research Question 4)
    print("\n" + "-" * 80)
    print(">>> [3. Multiple Linear Regression: Total Cup Points ~ Altitude + Method + Variety + Country]")
    reg_results = cu.fit_multiple_regression(df_clean)
    
    print(f"  * Model Sample Size (n): {reg_results['n_obs']:,} lots")
    print(f"  * R-Squared: {reg_results['r_squared']:.4f} (Explains {reg_results['r_squared']*100:.2f}% of total variance)")
    print(f"  * Adjusted R-Squared: {reg_results['adj_r_squared']:.4f}")
    print(f"  * Overall F-Stat: F = {reg_results['f_stat']:.2f}, p = {reg_results['f_pvalue']:.4e}")
    
    print("\n>>> [3b. Type II ANOVA Table: Factor Variance Explained (RQ4)]")
    anova_tab = reg_results["anova_table"].copy()
    anova_display = anova_tab[["sum_sq", "df", "F", "PR(>F)", "partial_eta_sq", "pct_model_variance"]].dropna()
    anova_display = anova_display.rename(columns={
        "sum_sq": "Sum of Sq", "df": "DF", "PR(>F)": "p-value",
        "partial_eta_sq": "Partial Eta²", "pct_model_variance": "% Model Variance"
    })
    print(anova_display.round(4).to_string())
    
    print("\n>>> [3c. Significant Robust Coefficients (HC3 Standard Errors)]")
    sig_coefs = reg_results["coefficients_significant"].copy()
    print(sig_coefs[["coef", "std_err", "t_stat", "p_val_robust", "ci_lower", "ci_upper"]].round(4).to_string())
    
    # 4. Generate Figure 7: Regression Diagnostics
    diag_plot_path = os.path.join(OUTPUTS_DIR, "07_regression_diagnostics.png")
    cu.plot_regression_diagnostics(reg_results["ols_model"], diag_plot_path)
    print(f"\n  -> Generated: {os.path.basename(diag_plot_path)} ({os.path.getsize(diag_plot_path)/1024:.1f} KB)")
    
    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
