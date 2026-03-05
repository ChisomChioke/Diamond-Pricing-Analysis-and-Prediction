[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://github.com/ChisomChioke/Diamond-Pricing-Analysis-and-Prediction/blob/main/Diamond_Pricing_Analysis_and_Prediction.ipynb)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue.svg)](https://www.linkedin.com/in/chisom-chioke/)
[![Dependencies](https://img.shields.io/badge/Dependencies-requirements.txt-green.svg)](https://github.com/ChisomChioke/Diamond-Pricing-Analysis-and-Prediction/blob/main/requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ChisomChioke/Diamond-Pricing-Analysis-and-Prediction/blob/main/LICENSE)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

# Diamond Pricing Optimization Model

## Automated Baseline Pricing for Jewelry Retail Using Multiple Linear Regression| R² = 0.928, MAE = $696

#### Major Finding: Simpson's Paradox
![Simpson's Paradox](images/diamond_simpsons_paradox_portfolio.png)

_Simpson's Paradox in diamond color grading: univariate analysis (left) incorrectly suggests worse color (more yellowed) commands premium prices, but 
controlling for carat weight (right) reveals the expected pattern — less yellowed diamonds command highest prices within each size category._

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Data Overview](#data-overview)
- [Analytical Approach](#analytical-approach)
- [Executive Summary of Results](#executive-summary-of-results)
    - [Key Findings](#key-findings)
    - [Edge Case Handling](#edge-case-handling)
- [Why This Project Matters](#why-this-project-matters)
- [Key Takeaways](#key-takeaways)
- [Limitations & Future Work](#limitations--future-work)
- [Technologies Used](#technologies-used)

## Project Overview

Jewelry retailers face thousands of daily pricing decisions, relying on expert judgment that doesn't scale efficiently. Using **53,921 diamonds**, I built a multiple linear regression model achieving **R² = 0.928 and MAE = $696** (18% of mean price) that automates baseline pricing for **40% of inventory** while flagging edge cases for expert review—reducing manual workload without sacrificing interpretability.

All code, data processing steps, and detailed methodology can be found here: [GitHub Repository Link](https://github.com/ChisomChioke/Diamond-Pricing-Analysis-and-Prediction/blob/main/Diamond_Pricing_Analysis_and_Prediction.ipynb), and a one-page summary is provided in this document: [One-Page Summary](https://drive.google.com/file/d/1f_iZ4K2Dx05scYHZ5-SS6CZlK0jjwA4q/view)

## Business Problem

Manual diamond pricing is time-intensive, inconsistent across specialists, and vulnerable to human error. Raw quality metrics exhibit counterintuitive patterns (worse quality diamonds appearing more expensive), making naive pricing rules unreliable.

#### Challenges:

1. How can we scale pricing decisions from expert judgment to thousands of diamonds per day?
2. Why do lower-quality diamonds sometimes appear more expensive than premium stones?
3. Which attributes truly drive diamond value when confounding variables are controlled?

Retailers need scalable automation that maintains interpretability for stakeholder trust while handling routine diamonds efficiently.

## Data Overview

**Dataset:** [Kaggle Diamonds Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds)

**Coverage:** 53,921 diamonds | 10 features (carat, cut, color, clarity, dimensions, price)

**After cleaning:** 53,921 valid records (removed 20 with impossible zero dimensions)

#### Key Features:

- **Carat:** Weight in metric carats (1 carat = 0.2 grams)
- **Cut:** Quality grade (Fair → Good → Very Good → Premium → Ideal)
- **Color:** D (colorless, best) → J (yellow, worst)
- **Clarity:** IF (flawless) → I1 (heavily included)
- **Dimensions:** Length (x), width (y), depth (z), depth (%), table (%)
- **Price:** US dollars (range: $326 - $18,823)

#### Data Preparation:

- Ordinal encoding for categorical quality features (maintains natural ordering)
- Train-test split: 80/20 stratified by price distribution
- No transformations applied (Linear relationships preserved. Log transformation degraded performance)

## Analytical Approach

The analysis follows a four-phase methodology designed to mirror professional model development:

#### Phase 1: Exploratory Data Analysis
- Distribution analysis and outlier detection
- Correlation study identifying carat as dominant driver (r = 0.92)
- **Simpson's Paradox discovery:** All quality features showed counterintuitive univariate patterns (worse quality → higher prices) due to carat weight confounding

#### Phase 2: Baseline Model Development
- **Carat-only model:** R² = 0.850, MAE = $996
- Established performance floor and validated carat dominance
- Identified model limitation: negative intercept causing invalid predictions for small diamonds

#### Phase 3: Full Model Development
- **20-feature OLS regression:** Carat + quality dimensions (cut, color, clarity) + physical dimensions
- R² = 0.928 (+7.7%), MAE = $696 (-30.1% error reduction)
- Coefficient interpretation aligned with gemological standards after controlling for carat

#### Phase 4: Production Readiness
- **Problem:** 9.5% of predictions negative (physically invalid)
- **Evaluated:** Log transformation (elegant) → **R² = -8.3** (catastrophic failure)
- **Implemented:** Predictive clipping to $326 minimum → **Improved MAE by 13%**
- Performance segmentation across price ranges to identify automation vs. review zones

## Executive Summary of Results

### Key Findings

#### 1. Simpson's Paradox Across All Quality Dimensions

Raw aggregate analysis suggested worse quality diamonds commanded premium prices:

- **Fair cuts:** $4,359 avg vs. Ideal: $3,457 avg
- **J-color:** $5,324 avg vs. D-color: $3,168 avg
- **SI2 clarity:** $5,060 avg vs. IF: $2,865 avg

**Root cause:** Lower-quality diamonds are substantially larger on average:

- **Fair cuts:** 1.05 carats vs. Ideal: 0.70 carats (50% larger)
- **J-color:** 1.16 carats vs. D-color: 0.66 carats (77% larger)
- **I1 clarity:** 1.28 carats vs. IF: 0.51 carats (151% larger)

**Resolution:** Multivariate regression controlling for carat reversed all relationships, aligning coefficients with gemological standards.

![Simpson's Paradox](images/diamond_simpsons_paradox_portfolio.png)
_Figure 1: Simpson's Paradox resolved. Univariate analysis (left) incorrectly suggests worse color commands premium prices, but controlling for carat weight (right) reveals colorless (D) diamonds command highest prices within each size category. Similar patterns observed across cut and clarity._

---

#### 2. Carat Dominates Pricing

| **Feature**     | **Impact**     | **Variance Interpretation** |
|---------    |--------    |----------------    |
| **Carat alone**   | +$7,787 per carat | 85% of variance (baseline model) |
| **Carat (full model)** | +$8,923 per carat | Coefficient increased after controlling for quality |
| **All quality features** | Combined | +7.8% (incremental over carat-only) |

**Business insight:** Size is the primary driver of price. Quality features provide premiums but are secondary to weight in determining base price.

---

#### 3. Quality Premiums Quantified

Once carat is controlled, quality attributes show expected gemological relationships:

| **Feature**     | **Impact**     | **Interpretation** |
|---------    |--------    |----------------|
| **Clarity (IF vs I1)**   | +$5,424 | Flawless commands largest quality premium |
| **Color (D vs J)** | +$2,308 | Colorless premium over yellow |
| **Color (E vs D)** | -$220 | Progressive penalty structure |
| **Cut (Ideal vs Fair)** | +$911 | Premium for ideal cut quality |
| **Cut (Good vs Fair)** | +$643 | Quality ladder consistent |

---

#### 4. Model Performance & Validation

| **Metric**     | **Value**   |
|---------   |-------- |
| **R² (Test)**  | 0.928 (93% of variance explained) |
| **MAE**        | $696 (18% of mean price) |
| **RMSE**       | $1,052 |
| **Overfitting Check** | Train R² (0.916) ≈ Test R² (0.915) |

![Model Performance](images/predicted_vs_actual_prices.png)
_Figure 2: Tight clustering around perfect prediction line (R² = 0.928, MAE = $696) with no systematic bias confirms model reliability across price ranges. Points distribute evenly above/below diagonal, validating production readiness._

#### Residual Diagnostics (4-Panel Analysis)
![Residual Diagnostics](images/residual_diagnostics.png)
_Figure 3: Residual diagnostics confirm model validity: (1) Random scatter around zero = linearity satisfied, (2) Q-Q plot shows approximately normal residuals, (3) Mild heteroscedasticity at higher prices—acceptable for production, (4) Residuals centered at $0 with no systematic bias._

---

#### 5. Performance Varies by Price Segment

Segmentation analysis reveals optimal automation zones:

| **Price Range** | **Test Count** | **MAE ($)** | **MAPE ($)**| **Recommendation**|
|---------    |--------    |------------- |---------| --------- |
| <$1K        |   2,943 (27%)    |    278      |   38.3%     |   Expert review (Clipping effects) |
| $1K – 2.5K  |  2,539 (24%)     |   532       |  33.5%      |  Acceptable for baseline pricing (Moderation automation) |
| $2.5K – 5K  |   2,405 (22%)    |   752       |  21.1%      |  Good performance (Automate) |
| $5K – 10K   |   1,906 (18%)    |   824       |  12.2%      |  Automate (Optimal zone) |
| >$10K       |   992 (9%)       |   1,979     |  14.0%      |  Expert review (High absolute error) |

**Model's optimal range:** $2.5K–10K range (4,311 diamonds, 40% of test set) achieves MAPE consistently below 22%—ideal for automated pricing with minimal expert intervention.

### Edge Case Handling
#### Negative Prediction Problem:

- Original model: 1,022 negative predictions (9.5% of test set)
- Minimum prediction: -$3,910 (impossible)
- **Solution:** Applied predictive clipping at $326 floor (minimum training price)

#### Results After Clipping:

- All predictions physically valid (≥ $326)
- Performance **improved:** R² increased 0.915 → 0.928, MAE decreased $801 → $696
- 1,557 predictions clipped (14.4%)—exclusively small, low-quality diamonds requiring expert review regardless

## Why This Project Matters

This project demonstrates how **rigorous exploratory analysis uncovers hidden patterns** that would mislead naive automation attempts. The discovery of Simpson's Paradox across all quality dimensions—where worse diamonds appeared more expensive until carat weight was controlled—prevented a critically flawed pricing rule that would have systematically mispriced inventory.

The analysis shows how **statistical models create measurable operational value** when:

**1. Diagnostic rigor prevents flawed assumptions:** Questioning counterintuitive patterns revealed confounding that would have undermined any automated rule

**2. Transparent models maintain stakeholder trust:** Interpretable coefficients enable pricing justification and regulatory compliance

**3. Performance segmentation guides deployment:** Understanding where models work (40% automation zone) vs. where expert judgment adds value (premium stones, outliers)

**4. Pragmatic problem-solving prioritizes business outcomes:** Clipping strategy outperformed elegant alternatives by focusing on valid predictions over mathematical purity

For jewelry retailers, the difference between naive automation and diagnostic-driven modeling represents the difference between operational chaos (mispriced inventory, eroded trust) and sustainable efficiency (expert time focused on high-value decisions, transparent pricing at scale).

## Key Takeaways

This project demonstrates:

**1. Domain knowledge is critical** — Discovering Simpson's Paradox required understanding gemological standards and questioning counterintuitive patterns

**2. EDA reveals fundamental insights** — The confounding relationship between carat and quality was evident before modeling, highlighting value of thorough investigation

**3. Simple solutions can outperform complex ones** — Predictive clipping proved more effective than log transformation

**4. Production readiness requires more than accuracy** — Handling edge cases, ensuring valid outputs, and clear documentation are essential

**5. Interpretability has business value** — Linear regression coefficients directly show stakeholders how each quality attribute impacts price, building trust in automated decisions

The model successfully balances **92.8% accuracy** with **interpretable coefficients**, handling **85.6% of diamonds automatically** while escalating edge cases to human experts—demonstrating a complete ML workflow from EDA through production deployment.

## Limitations & Future Work
#### Current Limitations:

**1. Linearity Assumptions:** Model assumes constant marginal value per carat. Residuals suggest nonlinear relationships at extremes. Log transformation degraded performance (R² = -8.3) due to error amplification.

**2. No Interaction Effects:** Model treats quality features independently. Diamond buyers likely value combinations differently (e.g., flawless + colorless may command disproportionate premium).

**3. Limited Coverage at Extremes:** Only 5 diamonds >4 carats in training (0.01%). Prediction confidence lower for very large stones.

**4. Static Pricing:** Doesn't capture temporal trends or seasonal market fluctuations. Periodic retraining required as markets evolve.

**5. Clipping Artifacts:** 14.4% of predictions required clipping, particularly affecting <$1K segment (MAPE = 38.3%). Manual review appropriate for these edge cases.

#### Future Enhancements:
**1. Advanced Modeling**
- Polynomial features (carat²) to capture nonlinear size-value relationship
- Interaction terms (carat × clarity, cut × color) for premium combinations
- Benchmark against tree-based methods (Random Forest, XGBoost) to quantify accuracy-interpretability trade-off

**2. Additional Features**
- Polish and symmetry ratings (subtle quality factors affecting premium stones)
- Temporal features to capture seasonal fluctuations
- More training samples for rare combinations (large + high-quality)

**3. Production Infrastructure**
- Web-based pricing application (Streamlit/Flask)
- Confidence intervals to flag uncertain predictions
- Monitoring dashboards to track accuracy and trigger retraining
- A/B testing framework to validate pricing recommendations

**4. Model Segmentation**
- Separate models for different size ranges to optimize accuracy at extremes
- Expert override system with feedback loop for continuous improvement

## Technologies Used
- **Python** — Data analysis and modeling
    - **Pandas/NumPy** — Data manipulation and aggregation
    - **Statsmodels** — OLS regression and statistical testing
    - **Scikit-learn** — Train-test split, preprocessing, metrics
    - **Matplotlib/Seaborn** — Visualization and diagnostics
- Jupyter Notebook — End-to-end reproducible analysis
- Statistical Methods — Simpson's Paradox diagnosis, hypothesis testing, residual diagnostics
