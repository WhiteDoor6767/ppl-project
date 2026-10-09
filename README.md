# Determinants of Arabica Coffee Cup Quality: An Empirical Agronomic Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain](https://img.shields.io/badge/Theme-Agriculture%20%7C%20CSBS-brightgreen.svg)]()

> **Course**: Python Programming Laboratory (PPL) Mini-Project  
> **Department**: Computer Science & Business Systems (CSBS)  
> **Institution**: K. J. Somaiya School of Engineering  
> **Topic**: What factors affect Arabica coffee cup quality?

---

## Project Overview

This project analyzes the empirical impact of **growing altitude**, **post-harvest processing methods**, **botanical varieties**, and **countries of origin** on the cup quality of 1,312 Arabica coffee lots certified by the Coffee Quality Institute (CQI, 2018).

Through programmatic data cleaning, non-parametric hypothesis testing (Kruskal-Wallis with Holm post-hoc tests), and multiple linear regression with Type II ANOVA decomposition, this project identifies which factors drive cup quality to provide actionable investment insights for coffee producers.

---

## Key Research Findings

1. **Altitude Effect (RQ1)**: A statistically significant positive linear relationship ($r = +0.243, p < 0.001$). After controlling for country, variety, and method, every **$1,000\text{ meters}$ elevation gain yields $+1.20\text{ cup points}$** ($p = 1.36 \times 10^{-6}$).
2. **Processing Method (RQ2)**: Processing methods differ significantly (Kruskal-Wallis $H = 10.37, p = 0.0347$). Post-hoc testing confirms **Natural processing achieves significantly higher rank quality than Washed** ($p_{adj} = 0.0488$).
3. **Varieties & Origins (RQ3)**: Both differ significantly ($p < 10^{-15}$). Modern cultivars (**Caturra** and **Bourbon**) significantly outperform traditional **Typica** ($p < 0.001$). **Ethiopia** ($+2.72$ pts, $p < 10^{-7}$) and **Kenya** ($+1.26$ pts) lead global origins.
4. **Dominant Factor (RQ4)**: **Country of Origin explains the most unique variation** in cup quality ($46.1\%$ of model sum of squares, $\text{Partial } \eta^2 = 0.1364$), followed by Altitude ($6.7\%$), Variety ($6.4\%$), and Processing Method ($3.9\%$).

---

## Project Structure

```
├── data/
│   ├── arabica_ratings_raw.csv        # Raw scraped CQI dataset (1,312 rows, 53 cols)
│   └── arabica_ratings_cleaned.csv    # Cleaned dataset (1,308 rows, 52 cols)
├── outputs/                           # 7 publication-quality charts (300 DPI)
│   ├── 01_cup_points_distribution.png
│   ├── 02_cup_points_by_processing_method.png
│   ├── 03_cup_points_by_variety.png
│   ├── 04_cup_points_by_country.png
│   ├── 05_altitude_vs_cup_points.png
│   ├── 06_correlation_heatmap.png
│   └── 07_regression_diagnostics.png
├── coffee_utils.py                    # Modular library (cleaning, plots, stats)
├── main.py                            # End-to-end pipeline execution script
├── main.ipynb                         # Interactive Jupyter Notebook
├── REPORT.md                          # Full academic project report & viva cheat sheet
├── requirements.txt                   # Dependency list
└── README.md                          # Repository overview
```

---

## Quickstart & Execution

### 1. Clone the repository
```bash
git clone https://github.com/WhiteDoor6767/coffee-quality-analysis.git
cd coffee-quality-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the complete pipeline
```bash
python3 main.py
```

### 4. Or launch the Jupyter Notebook
```bash
jupyter notebook main.ipynb
```

---

## Detailed Academic Report & Viva Guide
For full methodological explanations, mathematical formulas, hypothesis test rationale, and rapid viva defense questions, refer to [**`REPORT.md`**](REPORT.md).
