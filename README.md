# Home Credit Default Risk Modeling

**Group 4 – Corinn Childs, Gaby Rodriguez, Joel Jorgensen, Josh McAlister**

Home Credit is a global lender that serves many applicants without traditional credit histories.  
These “unbanked” customers represent both:

- **Growth opportunity** – new borrowers in underserved markets  
- **Credit risk** – higher uncertainty around ability to repay

Our goal in this project is to build a **probability-of-default (PD) model** that helps Home Credit:

- Assess default risk at the **individual applicant** level  
- Make better **approve / decline / manual review** decisions  
- Balance **profit** from good loans against **losses** from bad ones  

---

## Business Problem

Home Credit needs a reliable way to estimate:  
> *“What is the probability this applicant will default if we approve their loan?”*

The challenge is that defaults are **rare events (~8%)**, and information about applicants is spread across:

- Application data
- Credit bureau records
- Installment loan history
- Credit card behavior
- POS & cash loan balances

A successful solution must **combine these data sources**, handle missingness and imbalance, and produce **calibrated risk scores** that can be plugged into real credit policy.

---

## Our Solution: Gradient Boosted Trees (XGBoost)

We framed the problem as a **binary classification** task (default vs non-default) and built a **gradient boosted decision tree model (XGBoost)** on a unified engineered dataset.

### Data Preparation

Using R and `tidymodels`, we:

- Joined application data with four auxiliary tables:
  - **Bureau** (external credit history)
  - **Installments** (loan repayment behavior)
  - **Credit card** balances and payments
  - **POS & cash loans**  
- Dropped features with **>50% missing values**, except `EXT_SOURCE_1` (an important external credit score)
- Aggregated transactional tables to the **customer level** (mins, maxes, means, sums, ratios)
- Created **missing-value indicator flags**, then:
  - Imputed numeric variables with **medians**
  - Imputed categorical variables with **modes**
- Applied:
  - **Log transform** to heavily skewed fields (e.g., income)  
  - **Normalization** of numeric predictors  
  - **One-hot encoding** for categorical features  
  - Removal of **zero-variance** predictors  

All of this was implemented as a **reusable recipe**, so the exact same steps are applied to training, holdout, and Kaggle test data.

### Handling Class Imbalance

Because only ~8% of applicants default, a naive model can look “accurate” by predicting everyone as safe. To avoid this:

- We **down-sampled the majority class** so the training data had roughly a **2:1** ratio of non-default to default
- We evaluated models using metrics suited to imbalance:
  - ROC-AUC
  - PR-AUC
  - Precision, recall, specificity
  - F1 and Brier score  
  - Threshold-based confusion matrices

### Model Training & Evaluation

We trained an XGBoost model on the down-sampled dataset using:

- Moderate tree depth and number of trees
- A conservative learning rate
- Row and column subsampling to improve generalization

Performance was assessed via **stratified cross-validation** and a **strict holdout set**, then validated externally through **Kaggle submissions**.

---

## Model Performance & Business Impact

### Holdout Performance (Threshold = 0.5)

On the internal holdout set, our chosen XGBoost model achieved approximately:

- **ROC-AUC:** 0.77  
- **PR-AUC:** 0.26  
- **Accuracy:** 0.84  
- **Precision:** 0.95  
- **Recall:** 0.87  
- **Specificity:** 0.48  
- **F1 Score:** 0.91  

**Interpretation:**

- **High precision (0.95)**  
  When the model flags someone as high risk, it is usually correct. This reduces unnecessary declines and manual reviews.

- **Strong recall (0.87)**  
  The model catches most true defaulters, helping to avoid costly bad loans.

- **Improved specificity (~0.48)**  
  Compared to more aggressive configurations, this preserves more **good customers** in the approval pipeline.

### Kaggle Benchmark

We submitted the model to the **Home Credit Default Risk** competition:

- **Public AUC:** 0.7597  
- **Private AUC:** 0.7669  

These results are closely aligned with our internal ROC-AUC, which:

- Confirms that the model **generalizes well**  
- Suggests limited **overfitting** to our local validation data  

### Business Value

The gradient boosting solution creates value by:

1. **Risk Stratification**

   - Every applicant receives a **probability of default**  
   - Credit policy can map ranges of PD to:
     - Auto-approve
     - Auto-decline
     - Manual review

2. **Economic Optimization via Thresholds**

   - We applied **Platt scaling** (logistic calibration) to better align predicted PDs with observed default rates.
   - Using these calibrated scores, we built a **synthetic cost table** that estimates, per 1,000 applications:
     - Approvals
     - Defaults among approved loans
     - Expected **loss** from bad loans
     - Expected **profit** from good loans
     - **Net economic value**  

   This allows Risk and Product teams to choose a **decision threshold** that maximizes net value, instead of using an arbitrary 0.5 cutoff.

3. **Better Use of Behavioral Data**

   - An ablation comparison showed that including transactional and bureau data **improves ROC-AUC and Brier score** over using application data alone.
   - This supports further investment in integrating **behavioral and credit history** signals into the decisioning process.

---

## My Contribution – Josh McAlister

In addition to contributing to overall project discussions and documentation, my main responsibilities were:

### 1. Modeling R&D Around Complex Learners

- Implemented and tuned **neural network models** using `brulee` on the engineered feature set.
- Designed and ran grid searches over:
  - Hidden units
  - Dropout
  - Learning rate  
- Evaluated neural network performance against gradient boosting to understand:
  - Where deep learning might add value
  - When it becomes **computationally expensive** without clear gains

### 2. Ensemble (Stacking) Scaffold

- Built a **stacked ensemble** using `caret` that combined:
  - The XGBoost model  
  - The neural network  
  - A logistic regression meta-model on top (`caretStack`)
- Addressed compute constraints by:
  - Sub-sampling the training data
  - Reducing the number of CV folds
  - Simplifying neural net architecture  
- Showed empirically that, under our hardware and time constraints, the ensemble **did not materially outperform** the standalone XGBoost model in ROC-AUC or PR-AUC.
---
