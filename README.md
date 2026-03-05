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
    - [Business Impact](#business-impact)
- [Strategic Recommendations](#strategic-recommendations)
- [Implementation Considerations](#implementation-considerations)
- [Assumptions & Limitations](#assumptions--limitations)
- [Future Work](#future-work)
- [Technologies Used](#technologies-used)

## Project Overview

Jewelry retailers face thousands of daily pricing decisions, relying on expert judgment that doesn't scale efficiently. Using **53,921 diamonds**, I built a multiple linear regression model achieving **R² = 0.928 and MAE = $696** (18% of mean price) that automates baseline pricing for **40% of inventory** while flagging edge cases for expert review—reducing manual workload without sacrificing interpretability.

All code, data processing steps, and detailed methodology can be found here: [GitHub Repository Link](https://github.com/ChisomChioke/Diamond-Pricing-Analysis-and-Prediction/blob/main/Diamond_Pricing_Analysis_and_Prediction.ipynb), and a one-page summary is provided in this document: [One-Page Summary](https://drive.google.com/file/d/1f_iZ4K2Dx05scYHZ5-SS6CZlK0jjwA4q/view)

## Business Problem

Manual diamond pricing is time-intensive, inconsistent across specialists, and vulnerable to human error. Raw quality metrics exhibit counterintuitive patterns (worse quality diamonds appearing more expensive), making naive pricing rules unreliable.

**Challenges:**

1. How can we scale pricing decisions from expert judgment to thousands of diamonds per day?
2. Why do lower-quality diamonds sometimes appear more expensive than premium stones?
3. Which attributes truly drive diamond value when confounding variables are controlled?

Retailers need scalable automation that maintains interpretability for stakeholder trust while handling routine diamonds efficiently.

## Data Overview

**Dataset:** [Kaggle Diamonds Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds)

**Coverage:** 53,921 diamonds | 10 features (carat, cut, color, clarity, dimensions, price)

**After cleaning:** 53,921 valid records (removed 20 with impossible zero dimensions)

### Key Features:

- **Carat:** Weight in metric carats (1 carat = 0.2 grams)
- **Cut:** Quality grade (Fair → Good → Very Good → Premium → Ideal)
- **Color:** D (colorless, best) → J (yellow, worst)
- **Clarity:** IF (flawless) → I1 (heavily included)
- **Dimensions:** Length (x), width (y), depth (z), depth (%), table (%)
- **Price:** US dollars (range: $326 - $18,823)

### Data Preparation:

- Ordinal encoding for categorical quality features (maintains natural ordering)
- Train-test split: 80/20 stratified by price distribution
- No transformations applied (linear relationships preserved)

## Analytical Approach

The analysis follows a four-phase methodology designed to mirror professional model development:

### Phase 1: Exploratory Data Analysis
- Distribution analysis and outlier detection
- Correlation study identifying carat as dominant driver (r = 0.92)
- **Simpson's Paradox discovery:** All quality features showed counterintuitive univariate patterns (worse quality → higher prices) due to carat weight confounding

### Phase 2: Baseline Model Development
- **Carat-only model:** R² = 0.850, MAE = $996
- Established performance floor and validated carat dominance
- Identified model limitation: negative intercept causing invalid predictions for small diamonds

### Phase 3: Full Model Development
- **20-feature OLS regression:** Carat + quality dimensions (cut, color, clarity) + physical dimensions
- R² = 0.928 (+7.7%), MAE = $696 (-30.1% error reduction)
- Coefficient interpretation aligned with gemological standards after controlling for carat

### Phase 4: Production Readiness
- **Problem:** 9.5% of predictions negative (physically invalid)
- **Evaluated:** Log transformation (elegant) → **R² = -8.3** (catastrophic failure)
- **Implemented:** Predictive clipping to $326 minimum → **Improved MAE by 13%**
- Performance segmentation across price ranges to identify automation vs. review zones

## Executive Summary of Results

### Key Findings

### 1. Simpson's Paradox Across All Quality Dimensions

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

### 2. Carat Dominates Pricing

| **Feature**     | **Impact**     | **Variance Interpretation** |
|---------    |--------    |----------------    |
| **Carat alone**   | +$7,787 per carat | 85% of variance (baseline model) |
| **Carat (full model)** | +$8,923 per carat | Coefficient increased after controlling for quality |
| **All quality features** | Combined | +7.8% (incremental over carat-only) |

**Business insight:** Size is the primary driver of price. Quality features provide premiums but are secondary to weight in determining base price.

---

### 3. Quality Premiums Quantified

Once carat is controlled, quality attributes show expected gemological relationships:

| **Feature**     | **Impact**     | **Interpretation** |
|---------    |--------    |----------------|
| **Clarity (IF vs I1)**   | +$5,424 | Flawless commands largest quality premium |
| **Color (D vs J)** | +$2,308 | Colorless premium over yellow |
| **Color (E vs D)** | -$220 | Progressive penalty structure |
| **Cut (Ideal vs Fair)** | +$911 | Premium for ideal cut quality |
| **Cut (Good vs Fair)** | +$643 | Quality ladder consistent |

---

## 4. Model Performance & Validation

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

## 5. Performance Varies by Price Segment

Segmentation analysis reveals optimal automation zones:

| **Price Range** | **Test Count** | **MAE ($)** | **MAPE ($)**| **Recommendation**|
|---------    |--------    |------------- |---------| --------- |
| <$1K        |   2,943 (27%)    |    278      |   38.3%     |   Expert review (Clipping effects) |
| $1K — 2.5K  |  2,539 (24%)     |   532       |  33.5%      |  Acceptable for baseline pricing (Moderation automation) |
| $2.5K — 5K  |   2,405 (22%)    |   752       |  21.1%      |  Good performance (Automate) |
| $5K — 10K   |   1,906 (18%)    |   824       |  12.2%      |  Automate (optimal zone) |
| >$10K       |   992 (9%)       |   1,979     |  14.0%      |  Expert review (High absolute error) |

```
Overall Metrics:
├── R² (Test): 0.928
├── MAE (Test): $696
├── RMSE (Test): $1,139
└── MAPE (Test): 17.9%

Performance by Price Range:
├── <$1K:     MAPE = 38.3% (clipping effects)
├── $1K-2.5K: MAPE = 33.5%
├── $2.5K-5K: MAPE = 21.1%
├── $5K-10K:  MAPE = 12.2% ← Optimal range
└── >$10K:    MAPE = 14.0% (sparse training data)
```


---

## 🔧 Technical Approach

### 1. Exploratory Data Analysis
- **Dataset**: 53,940 diamonds with 10 features (carat, cut, color, clarity, dimensions, price)
- **Data Quality**: Removed 19 duplicates → 53,921 final observations
- **Feature Analysis**: Discovered Simpson's Paradox across all three quality dimensions
- **Correlation Study**: Identified carat as primary confounding variable (r = 0.92 with price)

### 2. Feature Engineering
- **Ordinal Encoding**: Cut (Fair=0 → Ideal=4), Color (J=0 → D=6), Clarity (I1=0 → IF=7)
- **Categorical Treatment**: Maintained ordinal relationships in quality features
- **No Transformations**: Linear relationships preserved (log transformation degraded performance)

### 3. Model Development
```
Baseline Model (Carat Only):
├── R² = 0.850
├── MAE = $996
└── Purpose: Establish performance floor

Full Model (All Features):
├── R² = 0.928 (+7.7% improvement)
├── MAE = $696 (-30.1% error reduction)
└── Features: Carat + Cut + Color + Clarity + Dimensions
```

### 4. Problem-Solving: Negative Predictions
**Issue**: Model's negative intercept caused 9.5% of predictions to be negative (physically invalid)

**Approach Evaluated**:
1. **Log Transformation**: Transform price → log(price) → exponential back
   - Result: R² = -8.3 (catastrophic failure)
   - Reason: Exponential amplification of errors for large diamonds
   
2. **Predictive Clipping**: Set minimum prediction = $326 (5th percentile of training data)
   - Result: R² = 0.928, MAE = $696 (improved 13% vs no clipping)
   - Advantage: Simple, maintains accuracy, ensures valid outputs

**Decision**: Implemented clipping strategy—demonstrating practical judgment over mathematical elegance

### 5. Model Validation



- **Residuals vs Fitted**: Slight heteroscedasticity at extremes (expected for price data)
- **Q-Q Plot**: Approximately normal distribution with minor heavy tails
- **Scale-Location**: Confirms heteroscedasticity (variance increases with fitted values)
- **Residuals Histogram**: Near-normal distribution centered at zero

#### Performance Segmentation
```python
# Evaluation by price range reveals optimal performance zones
$2.5K-10K segment: 40% of test set, MAPE consistently below 22%
→ Ideal range for automated pricing
```

---

## 📊 Methodology

### Statistical Analysis
- **Multiple Linear Regression**: OLS regression with 20 features (4 quality + 5 dimensions + interactions)
- **Hypothesis Testing**: t-tests for coefficient significance (p < 0.01 threshold)
- **Correlation Analysis**: Pearson correlation to identify confounding relationships
- **VIF Assessment**: Variance Inflation Factor to check multicollinearity (all VIF < 10)

### Model Evaluation
- **Train-Test Split**: 80/20 stratified split maintaining price distribution
- **Cross-Validation**: Confirmed no overfitting (train R² = 0.916, test R² = 0.915)
- **Multiple Metrics**: R², MAE, RMSE, MAPE for comprehensive evaluation
- **Segmentation Analysis**: Performance breakdown by price range to identify optimal zones

### Validation Checks
- **Residual Analysis**: 4-panel diagnostics (residuals vs fitted, Q-Q, scale-location, histogram)
- **Heteroscedasticity Testing**: Confirmed variance increases with price (expected behavior)
- **Normality Assessment**: Q-Q plot shows approximately normal distribution of residuals
- **Outlier Detection**: Identified but retained (represent legitimate premium stones)

---

## 🔍 Key Insights

### 1. Simpson's Paradox Across All Quality Features

**Color Paradox**:
- Univariate: J-color ($5,324) > D-color ($3,170) [+$2,154 apparent premium]
- Controlled: D-color commands +$2,308 vs J-color within same carat bin
- Explanation: J-color stones average 1.16 carats vs D-color 0.66 carats (77% larger)

**Clarity Paradox**:
- Univariate: SI2 ($5,060) > IF ($2,865) [+$2,195 apparent premium]
- Controlled: IF commands +$5,424 vs I1 within same carat bin
- Explanation: I1 stones average 1.28 carats vs IF 0.51 carats (151% larger)

**Cut Paradox**:
- Univariate: Fair ($4,359) > Ideal ($3,458) [+$901 apparent premium]
- Controlled: Ideal commands +$911 vs Fair within same carat bin
- Explanation: Fair cuts average 1.05 carats vs Ideal 0.70 carats (50% larger)

**Lesson**: Always check for confounding variables before drawing conclusions from bivariate relationships

### 2. Carat Dominates Diamond Valuation

**Statistical Evidence**:
- Univariate R² (carat alone): 0.850 (explains 85% of variance)
- Full model R² (all features): 0.928 (quality adds only 7.8% explanatory power)
- Coefficient: +$8,923 per carat (standardized β = 0.92)

**Business Implication**: Size is the primary value driver; quality features provide premiums 
but are secondary to weight in determining base price.

### 3. Performance Varies by Price Segment

**Optimal Zone ($2.5K-10K)**:
- 4,311 diamonds (40% of test set)
- MAPE = 12-21% (within business tolerance)
- High training density enables accurate predictions
- **Recommendation**: Automate pricing for this segment

**Edge Cases Requiring Manual Review**:
- Budget diamonds (<$1K): MAPE = 38.3% due to clipping distortion
- Premium stones (>$10K): MAPE = 14.0% but absolute errors ($1,979 MAE) may exceed tolerance
- Very large diamonds (>4 carats): Only 5 training samples (0.01%), low confidence

### 4. Simple Solutions Can Outperform Complex Ones

**Problem**: 9.5% of predictions negative (physically impossible)

**Complex Solution Attempted**: Log transformation
- Theoretically elegant (ensures positive predictions mathematically)
- Result: R² = -8.3 (worse than random guessing!)
- Failure reason: Exponential back-transformation amplifies errors for large diamonds

**Simple Solution Implemented**: Clipping to minimum observed price
- Straightforward approach (if prediction < $326, set to $326)
- Result: R² = 0.928, MAE improved 13%
- Success reason: Preserves model accuracy while ensuring validity

**Lesson**: Mathematical elegance doesn't guarantee practical effectiveness; prioritize business outcomes over theoretical purity

---

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+**: Primary programming language
- **Pandas 1.5.3**: Data manipulation and analysis
- **NumPy 1.24.2**: Numerical computations
- **Statsmodels 0.14.0**: Statistical modeling (OLS regression)
- **Scikit-learn 1.2.2**: Machine learning utilities (train-test split, metrics)

### Visualization
- **Matplotlib 3.7.1**: Static plots and multi-panel visualizations
- **Seaborn 0.12.2**: Statistical visualizations and theme styling

### Statistical Analysis
- **SciPy 1.10.1**: Statistical tests (t-tests, correlation)

### Development Environment
- **Jupyter Notebook**: Interactive development and documentation
- **Git**: Version control

---

## 📚 Future Enhancements

### Model Improvements
1. **Polynomial Features**: Test carat² term to capture non-linear carat-price relationship
2. **Interaction Terms**: Evaluate carat × clarity, carat × color to model premium scaling
3. **Segmented Models**: Separate models for budget (<$2.5K), mid-range, and premium (>$10K) diamonds
4. **Advanced Methods**: Benchmark Random Forest/XGBoost to quantify accuracy-interpretability trade-off

### Additional Features
5. **Certification Lab**: GIA vs other labs affects pricing (data not available in current dataset)
6. **Fluorescence**: Impacts value for high-color diamonds (data not available)
7. **Polish & Symmetry**: Subtle quality factors affecting premium stones (data not available)
8. **Temporal Features**: Capture seasonal price fluctuations and market trends

### Deployment Considerations
9. **Web Interface**: Streamlit/Flask app for retailer self-service pricing
10. **Confidence Intervals**: Provide prediction ranges (e.g., ±10%) alongside point estimates
11. **Monitoring Dashboard**: Track accuracy over time, detect model drift
12. **Automated Retraining**: Quarterly updates when performance degrades or market shifts
