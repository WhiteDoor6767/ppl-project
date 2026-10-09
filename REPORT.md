# Arabica Coffee Cup Quality Analysis: Determinants of Specialty Rating
**Course**: Python Programming Laboratory (PPL) Mini-Project  
**Department**: Computer Science & Business Systems (CSBS)  
**Institution**: K. J. Somaiya School of Engineering  
**Theme**: Agriculture | **Domain**: Post-Harvest Quality & Agro-Climatology  

---

## 1. Problem Statement & Research Questions

### 1.1 Problem Statement
The Specialty Coffee Association (SCA) cupping protocol grades green Arabica coffee on an objective 100-point scale, where lots scoring $\ge 80.0$ command significant market price premiums over commercial commodity coffee. However, coffee producers face high resource constraints when deciding whether to invest in high-elevation land acquisition, premium botanical varietals, or labor-intensive processing methods (e.g., Natural vs. Washed).

The objective of this project is to analyze the empirical effect of **growing altitude**, **post-harvest processing method**, **botanical variety**, and **country of origin** on the cup quality of 1,312 Arabica coffee lots certified by the Coffee Quality Institute (CQI). Through robust programmatic data cleaning, hypothesis testing (Kruskal-Wallis with post-hoc tests), and multiple linear regression, this project quantifies which agronomic and post-harvest factors drive cup quality and explains where producers should focus their investments.

### 1.2 Research Questions
1. **RQ1**: Does altitude have a statistically significant relationship with total cup points, and is the effect linear?
2. **RQ2**: Do post-harvest processing methods (Washed, Natural, Semi-Washed, Pulped Natural) differ significantly in cup quality?
3. **RQ3**: Do botanical varieties and countries of origin differ significantly, and which specific pairs diverge?
4. **RQ4**: After controlling for the other factors in a multivariable framework, which factor explains the most unique variation in cup quality?

---

## 2. Dataset Description & Source

* **Dataset**: `arabica_ratings_raw.csv` (1,312 raw observations, 53 initial columns).
* **Source**: Scraped in January 2018 by James LeDoux from the Coffee Quality Institute (CQI) database ([database.coffeeinstitute.org](https://database.coffeeinstitute.org/)).
* **Target Variable**: `Total Cup Points` (Specialty coffee rating, continuous scale: 0–100).
* **Predictor Variables**:
  * `Altitude`: Raw text elevation string with heterogeneous units, ranges, and formatting errors.
  * `Processing Method`: Categorical post-harvest processing method (Washed, Natural, Semi-Washed, Honey, Other).
  * `Variety`: Botanical Arabica cultivar (Bourbon, Caturra, Typica, Catuai, Catimor, Kona, etc.).
  * `Country of Origin`: Geographic producing nation (20 countries with $\ge 10$ lots).
  * `Moisture`: Bean moisture percentage.
  * `Defects`: Category 1 (primary) and Category 2 (secondary) physical bean defect counts.

---

## 3. Data Cleaning & Preprocessing Summary

The raw dataset contained extensive web-scraping artifacts, text formatting errors, and extreme anomalies that required systematic programmatic cleaning:

| Step | Identified Anomaly | Implemented Solution | Viva Defense / Methodological Justification |
| :--- | :--- | :--- | :--- |
| **3.1 Junk Removal** | 12 uninformative columns (`view_certificate_1/2`, `Request a Sample`, `NA`, `NA.1`, `Unnamed: 51`) were 100% empty or scraping header leftovers. | Dropped using vectorized `df.drop()`. | *"Eliminated zero-variance noise and scraping table artifacts to clean the feature space without discarding valid observational data."* |
| **3.2 Cup Points Parsing** | Scraped HTML prepended `'Sample  '` to every rating (e.g., `'Sample  83.17'`). | Stripped via regex `str.replace()` and vectorized cast to `float64`. | *"Restored numeric integrity so cupping ratings could be analyzed using statistical estimators."* |
| **3.3 Altitude Normalization** | Mixed units (msnm, masl, ft, PSNM), European decimal formats (`1.872 masl`), midpoint ranges (`1200-1800`), OCR errors (`12OO`), dropped decimals (`190164`), and implicit feet (`5,000` / `5440-5760 METROS` in Guatemala). | Built a custom parser extracting midpoints for ranges, converting feet/pies to meters ($\times 0.3048$), fixing typos, and bounding valid elevations between $100\text{m}$ and $3,000\text{m}$. | *"In Guatemala, coffee is graded in feet under the Anacafé system (Strictly Hard Bean: 4,500–5,500 ft). Blindly treating them as meters implies farming above Mont Blanc. Converting feet to meters restored botanical realism."* |
| **3.4 Physical Attributes** | Moisture had `' %'` signs, and defects had `' full defects'` suffixes. | Extracted numeric floats and integers using regular expression digit tokens. | *"Transformed string fields into continuous physical predictors for correlation and regression analysis."* |
| **3.5 Category Standardization** | High cardinality with single-lot varieties and countries. | Preserved top 8 varieties and countries with $\ge 10$ lots; collapsed low-frequency categories into `'Other Variety'` and `'Other Origin'`. | *"Collapsed micro-categories to avoid dummy variable explosion and preserve statistical degrees of freedom in regression."* |
| **3.6 Outlier Handling** | $1.5 \times \text{IQR}$ flagged 16 high outliers ($> 87.42$) and 59 low outliers ($< 77.42$). Two lots had corrupted entries: Row 1311 ($0.00$) and Row 1310 ($43.13$). | Dropped corrupted lots $< 60.0$ points; retained genuine specialty lots ($88.0\text{–}90.58$) and low commercial lots ($60.0\text{–}77.42$). | *"Blindly trimming IQR outliers introduces sample truncation bias. High scores reflect authentic specialty coffee; we strictly removed uncupped/corrupted lots scoring $< 60$ per SCA standards."* |
| **3.7 Missing Handling** | 202 rows missing Variety (15.4%), 153 missing Processing Method (11.7%), 1 row missing Country. | Dropped row 1197 (no country or farm data); imputed `'Unknown'` category for Variety/Method to preserve sample size ($n=1,308$) for univariate tests; applied complete-case analysis ($n=1,069$) for regression. | *"Imputing 'Unknown' prevents losing 17% of observational power during univariate hypothesis testing while maintaining clean complete cases in multivariable regression."* |

---

## 4. Key Exploratory Visual Findings

All charts were rendered at 300 DPI into `outputs/`:

### 4.1 Figure 1: Distribution of Total Cup Points
* **Visual**: Histogram with overlaid KDE curve and annotated SCA specialty benchmarks ($80.0$ and $85.0$).
* **Finding**: The distribution is **moderately left-skewed** ($\text{skewness} = -1.54$) with mean $82.20$, median $82.50$, and IQR $[81.17, 83.67]$. Over $85\%$ of lots fall between 80.0 and 85.0, reflecting the selective nature of CQI certification.
* **Viva Note**: *"The negative skewness highlights that commercial specialty coffees cluster tightly above 80 points, but lots below 75 represent defective commercial samples."*

### 4.2 Figure 2: Processing Method Comparison (RQ2)
* **Visual**: Sorted boxplots with mean markers overlaid with jittered strip points ($n$ counts in labels).
* **Finding**: **Honey / Pulped Natural** ($\mu = 82.81, \text{med} = 82.66$) and **Natural** ($\mu = 82.35, \text{med} = 82.75$) achieve higher median quality than **Washed** ($\mu = 81.99, \text{med} = 82.42$). Washed coffee accounts for $71\%$ of evaluated volume ($n=811$) and exhibits greater downward dispersion.

### 4.3 Figure 3: Botanical Cultivar Comparison (RQ3)
* **Visual**: Boxplots of top 8 cultivars ordered by median cup points.
* **Finding**: Modern hybrid cultivars like **Catimor** ($\text{med} = 83.17, n=20$) and **Caturra** ($\text{med} = 83.17, n=255$) show high median quality, while traditional heirlooms like **Bourbon** ($\text{med} = 82.38, n=226$) and **Typica** ($\text{med} = 81.54, n=211$) show wider variance with isolated ultra-high scores ($> 88$).

### 4.4 Figure 4: Country of Origin Comparison (RQ3)
* **Visual**: Horizontal bar chart with 95% bootstrap confidence intervals for origins with $\ge 15$ lots.
* **Finding**: **Ethiopia** significantly leads global quality with an average score of **$85.48$** ($n=44$), followed by **Colombia** ($83.11$) and **Costa Rica** ($82.79$). Central American bulk producers (**Mexico** $80.89$, **Honduras** $80.88$, **Nicaragua** $80.46$) average below the global mean ($82.20$).

### 4.5 Figure 5: Altitude vs. Total Cup Points Regression (RQ1)
* **Visual**: Scatter plot of $1,069$ lots with OLS regression fit ($y = 0.0017x + 79.99$) and 95% confidence band.
* **Finding**: Statistically significant positive correlation ($r = +0.243, p < 0.001$). Every $1,000\text{ meters}$ of elevation increase is associated with a unadjusted gain of **$+1.68\text{ cup points}$**.

### 4.6 Figure 6: Correlation Heatmap Across Features
* **Visual**: Triangular Pearson correlation matrix.
* **Finding**: Cup points correlate most strongly with sensory **Flavor** ($r = 0.84$) and **Acidity** ($r = 0.72$). Altitude's strongest sensory pathway is **Acidity** ($r = 0.27$), confirming that elevation enhances cup points by promoting organic acid synthesis. **Category 2 defects** penalize quality ($r = -0.29$).

---

## 5. Statistical Findings & Hypothesis Testing

### 5.1 Assumption Testing (Normality & Homoscedasticity)
* **Shapiro-Wilk Test**: Both Washed ($p = 2.47 \times 10^{-22}$) and Natural ($p = 1.46 \times 10^{-12}$) groups strongly reject normality. Across varieties, 6 out of 8 cultivars reject normality ($p < 0.001$).
* **Levene's Test for Homoscedasticity**:
  * Across Processing Methods: $W = 0.8540, p = 0.491$ (Variances are homogeneous).
  * Across Varieties: $W = 2.7883, p = 0.0047 < 0.05$ (**Variances are unequal**).
* **Methodological Justification**: Because normality is violated across processing methods and both normality and homoscedasticity are violated across varieties, standard parametric One-Way ANOVA is invalid. The non-parametric **Kruskal-Wallis $H$-test** was selected and justified.

### 5.2 Hypothesis Testing Results

#### Processing Methods (RQ2)
* **One-Way ANOVA**: $F = 2.43, p = 0.0461$.
* **Kruskal-Wallis**: $H = 10.37, \mathbf{p = 0.0347 < 0.05}$ (**Statistically Significant**).
* **Post-Hoc Analysis**: Non-parametric pairwise Mann-Whitney U test with Holm-Bonferroni adjustment confirms that **Natural coffee achieves a significantly higher rank quality than Washed coffee** ($p_{raw} = 0.0081, p_{adj} = 0.0488 < 0.05$). Other pairwise differences are non-significant due to smaller sample sizes in Honey and Semi-Washed lots.

#### Botanical Varieties (RQ3 - Part 1)
* **Kruskal-Wallis**: $H = 101.85, \mathbf{p = 1.78 \times 10^{-18} < 0.001}$ (**Highly Significant**).
* **Post-Hoc Analysis (Holm-Adjusted)**:
  * **Caturra vs. Typica**: $p_{adj} = 4.71 \times 10^{-17}$ (Caturra significantly higher).
  * **Bourbon vs. Typica**: $p_{adj} = 3.41 \times 10^{-4}$ (Bourbon significantly higher).
  * **Catimor vs. Typica**: $p_{adj} = 5.83 \times 10^{-4}$ (Catimor significantly higher).
  * **Bourbon vs. Caturra**: $p_{adj} = 1.90 \times 10^{-4}$.

### 5.3 Multiple Linear Regression & Variance Partitioning (RQ4)

$$\text{Total Cup Points} = \beta_0 + \beta_1(\text{Altitude}) + \sum \beta_i(\text{Method}) + \sum \beta_j(\text{Variety}) + \sum \beta_k(\text{Country}) + \varepsilon$$

* **Sample Size**: $n = 1,069$ complete lots.
* **Model Statistics**: $R^2 = 0.2550$, $\text{Adjusted } R^2 = 0.2313$, $F = 10.74, p = 1.37 \times 10^{-46}$.

#### Type II ANOVA Table (Unique Variance Partitioning)

| Factor | Sum of Squares | DF | $F$-Stat | $p$-value | Partial $\eta^2$ | % Model Variance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Country of Origin** | **827.83** | **19** | **8.60** | **$< 10^{-20}$** | **0.1364** | **46.12%** |
| **Altitude ($m$)** | **119.60** | **1** | **23.61** | **$1.36 \times 10^{-6}$** | **0.0223** | **6.66%** |
| **Botanical Variety** | **115.01** | **8** | **2.84** | **0.0040** | **0.0215** | **6.41%** |
| **Processing Method** | **69.20** | **5** | **2.73** | **0.0184** | **0.0130** | **3.86%** |
| *Residual* | *5,243.29* | *1,035* | — | — | — | — |

* **Research Question 4 Conclusion**: **Country of Origin explains the most variation** in Arabica cup quality ($\text{Partial } \eta^2 = 0.1364$, contributing $46.1\%$ of explained model variance), followed by Altitude ($6.7\%$), Variety ($6.4\%$), and Processing Method ($3.9\%$).

#### Robust Regression Coefficients (HC3 Standard Errors)
*(Baseline Reference Group: Washed, Bourbon, Colombia)*
* **Altitude**: $\beta = +0.0012, t = 4.32, p = 1.36 \times 10^{-6}$ (**$+1.20\text{ cup points per } 1,000\text{m}$** elevation gain after controlling for all other factors).
* **Processing**: Natural scores **$+0.47\text{ points}$** higher ($p = 0.0237$) and Semi-Washed scores **$+0.80\text{ points}$** higher ($p = 0.0055$) than Washed.
* **Origins**: Ethiopia scores **$+2.72\text{ points}$** higher ($p < 10^{-7}$) and Kenya scores **$+1.26\text{ points}$** higher ($p = 0.0040$) than Colombia. Conversely, Honduras ($-2.00$, $p < 10^{-6}$) and Mexico ($-1.10$, $p = 0.0007$) score significantly lower.

#### Residual Diagnostics (Figure 7)
* **Breusch-Pagan Test**: $LM = 71.75, p = 1.09 \times 10^{-4}$ (Detects mild heteroscedasticity).
* **Correction**: Implemented **HC3 robust standard errors**; all primary predictors retain statistical significance.

---

## 6. Practical Takeaways for Coffee Producers

1. **Prioritize Terroir & Microclimate (Altitude)**: Altitude provides a guaranteed, linear rate of return of $+1.20$ points per $1,000\text{m}$. For farms situated above $1,400\text{m}$, cooler diurnal cycles naturally foster high cup quality.
2. **Transitioning to Natural / Honey Processing**: Producers at lower or moderate elevations ($800\text{–}1,200\text{m}$) who cannot alter their altitude can gain an estimated $+0.47$ to $+0.80$ points by shifting from Washed to Natural or Semi-Washed processing, helping them surpass the critical $80.0$ SCA specialty price threshold.
3. **Upgrade Away from Typica**: Typica consistently underperforms modern selections (Caturra and Bourbon outscore Typica by $> 0.8$ to $1.7$ points). Replanting aging Typica plots with Caturra or Bourbon elevates quality while maintaining vigor.
4. **Defect Management Over Variety**: In the correlation matrix, physical secondary defects penalize final scores ($r = -0.29$) more than varietal choice can overcome. Investing in flotation sorting and optical sorting at the dry mill offers higher immediate ROI than buying exotic seeds.

---

## 7. Limitations

1. **Cross-Sectional 2018 Snapshot**: The data represents a static snapshot scraped in 2018, precluding longitudinal analysis of climate variability across successive harvest years.
2. **Self-Reported Farm Elevations**: Altitudes were self-reported on cupping submission forms without GPS verification, requiring algorithmic filtering to resolve reporting unit ambiguities.
3. **Selection Bias**: The dataset comprises coffees voluntarily submitted for CQI certification, creating a survivorship bias toward commercial and specialty producers (scores $< 60$ are largely absent).
4. **Exclusion of Robusta**: Only 28 Robusta samples were present in the database, requiring their exclusion to maintain valid group sample sizes and prevent confounding species-level biological differences.

---

## 8. Viva Voce Defense Cheat Sheet (Rapid Q&A)

* **Q1: Why did you choose Kruskal-Wallis instead of ANOVA for Processing Method and Variety?**
  * *Answer*: "Both the Shapiro-Wilk test ($p < 10^{-12}$) and Levene's test ($p = 0.0047$) demonstrated that the parametric assumptions of normality and equal variances were violated. One-Way ANOVA is sensitive to variance inflation from skewed tails; Kruskal-Wallis evaluates mean ranks, making it mathematically valid and robust."

* **Q2: Why did you not drop all IQR outliers below 77.42 and above 87.42?**
  * *Answer*: "Blindly dropping statistical outliers causes sample truncation bias. Scores between 88 and 90.58 are genuine specialty coffees (like Ethiopian heirlooms), which represent the exact high-quality phenomenon we seek to model. We only dropped non-commercial corrupted entries scoring $< 60$ (like 0.00 and 43.13) based on SCA cupping guidelines."

* **Q3: How did you handle the messy Altitude column?**
  * *Answer*: "We developed a multi-stage parser that calculated midpoints for ranges, converted explicit feet and Central American Anacafé values (3,000–8,000) to meters using $\times 0.3048$, resolved OCR errors (like '12OO' to 1200), and bounded plausible coffee cultivation between 100m and 3,000m."

* **Q4: Why did you exclude the individual cupping attributes (Aroma, Flavor, etc.) from the regression?**
  * *Answer*: "Total Cup Points is defined mathematically as the exact sum of the 10 cupping sub-scores. Including them as predictors would introduce perfect circular collinearity ($R^2 = 1.0$), violating regression assumptions and failing to answer what agricultural growing factors actually drive quality."

* **Q5: What factor explains the most variation in Arabica cup quality?**
  * *Answer*: "Based on our Type II ANOVA Sum of Squares decomposition, Country of Origin explains the most unique variation ($46.1\%$ of model sum of squares, Partial $\eta^2 = 0.1364$), reflecting macro-terroir, regional genetics, and national harvesting infrastructure, followed by Altitude ($6.7\%$), Variety ($6.4\%$), and Processing Method ($3.9\%$). In our model, Altitude provides a steady linear gain of $+1.20$ points per $1,000\text{ meters}$."
