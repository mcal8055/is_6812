title: "Home Credit Default Risk Modeling"

business_problem_and_objective: |
  Home Credit is a global lender serving many applicants without traditional credit histories.
  These "unbanked" customers represent both a growth opportunity and a source of credit risk.

  Business problem:
  Home Credit needs a reliable way to estimate the probability that a loan applicant will default,
  using all available application and transaction data.

  Project objective:
  Build and deploy a probability-of-default (PD) model that:
    - assigns each applicant a PD score,
    - supports data-driven approve/decline/review decisions, and
    - quantifies the trade-off between profit and expected credit losses.

solution_overview: |
  We framed the task as a binary classification problem (default vs non-default, ~8% default rate)
  and built a gradient boosted decision tree model (XGBoost) on a unified, engineered dataset.

  Data preparation (R + tidymodels):
    - Joined the main application data with four auxiliary sources:
        * Bureau data (external credit history)
        * Installment payments
        * Credit card balances
        * POS & cash loan balances
    - Dropped features with >50% missing values, except EXT_SOURCE_1 (external credit score).
    - Aggregated transactional tables to the customer level (mins, maxes, averages, sums, ratios).
    - Created missing-value indicator flags, then used median/mode imputation.
    - Log-transformed heavily skewed variables (e.g., income) and normalized numeric features.
    - One-hot encoded categorical variables and removed zero-variance predictors via a reusable recipe.

  Modeling (Gradient Boosting / XGBoost):
    - Handled class imbalance by down-sampling the majority class to a 2:1 ratio.
    - Fit an XGBoost model with reasonable tree depth, number of trees, learning rate, and subsampling.
    - Used stratified cross-validation to evaluate performance with ROC-AUC, PR-AUC, accuracy,
      precision, recall, specificity, F1, and Brier score.
    - Trained the final model on the down-sampled training data and scored:
        * a holdout set for internal validation, and
        * the Kaggle test set to generate competition submissions.

gbm_results: |
  Internal holdout performance (threshold = 0.5):
    - ROC-AUC: ~0.77
    - PR-AUC: ~0.26
    - Accuracy: ~0.84
    - Precision: ~0.95
    - Recall: ~0.87
    - Specificity: ~0.48
    - F1 Score: ~0.91

  Confusion-matrix interpretation:
    - High precision: most applicants flagged as high risk truly are high risk, minimizing
      unnecessary declines and manual reviews.
    - Strong recall: the model captures the majority of true defaulters, reducing credit losses.
    - Improved specificity vs more aggressive settings, preserving more low-risk approvals.

  Kaggle Home Credit competition:
    - Public AUC: 0.7597
    - Private AUC: 0.7669

  These scores closely match the holdout ROC-AUC, indicating that the gradient boosting model
  generalizes well and is not overfit to the training data.

business_value: |
  The gradient boosting solution delivers business value by:

    - Risk stratification:
        * Each applicant receives a calibrated probability of default.
        * Credit policy can define PD thresholds for auto-approve, auto-decline, or manual review.

    - Economic optimization:
        * Using Platt-scaled probabilities, we built a synthetic cost table that simulates
          approvals, defaults, expected loss, expected profit, and net value per 1,000 applicants
          across different PD thresholds.
        * Risk and Product teams can choose the threshold that maximizes net economic value
          instead of relying on arbitrary cutoffs.

    - Better use of behavioral data:
        * An ablation experiment showed that including transaction-level features along with
          application data improves ROC-AUC and Brier score.
        * This supports further investment in data integration and behavioral signals.

    - Operational readiness:
        * A single, scripted pipeline (recipes + XGBoost) makes the model reproducible and
          easier to monitor, document, and deploy in a production scoring environment.

my_contribution_josh_mcalister: |
  My primary contributions to the project were:

    - Modeling R&D around complex learners:
        * Implemented and tuned neural network models (using brulee) on the cleaned feature set.
        * Experimented with hyperparameters (hidden units, dropout, learning rate) to understand
          when deep models provide incremental value over tree-based approaches.
        * Built and evaluated an ensemble stack that combined XGBoost and neural nets, using a
          logistic regression meta-learner (caretStack).

    - Evidence for selecting GBM as the winning approach:
        * Showed that, under our compute constraints, the stacked ensemble and neural networks
          did not materially outperform the gradient boosting model in ROC-AUC or PR-AUC.
        * Helped position XGBoost as the best balance of performance, stability, and
          deployability for this business use case.

    - Communication and interpretability:
        * Contributed to the interpretation of GBM metrics (precision, recall, specificity, PR-AUC)
          in business terms—how many defaulters we catch vs how many good applicants we might flag.
        * Helped translate technical performance into interview-ready talking points about
          threshold selection, calibration, and portfolio impact.

difficulties_encountered: |
  Key challenges in the GBM-focused workflow:

    - Class imbalance:
        * With only ~8% defaults, naive models could achieve high accuracy by predicting
          "non-default" for most applicants.
        * We mitigated this by down-sampling, focusing on PR-AUC and recall, and analyzing
          confusion matrices rather than accuracy alone.

    - Data complexity:
        * Joining five datasets, aggregating transactions, and one-hot encoding produced a
          large, sparse feature space.
        * We addressed this with a tidymodels recipe that standardized imputation, scaling,
          and feature selection, and ensured the same steps applied to training, holdout,
          and Kaggle test data.

    - Threshold and calibration:
        * A fixed 0.5 threshold is not automatically optimal for profit.
        * Platt scaling plus the synthetic cost table helped connect model scores to business
          outcomes and guided a more principled threshold choice.

lessons_learned: |
  From this project we learned:

    - How to design an end-to-end, production-style credit risk pipeline:
        * ingestion, feature engineering, modeling, evaluation, and deployment artifacts
          (Kaggle submissions).

    - The importance of class imbalance handling and proper metrics:
        * ROC-AUC alone is not enough; PR-AUC, precision/recall, and cost-based analysis are
          critical in rare-event settings like default prediction.

    - Why gradient boosting is often a strong baseline for tabular data:
        * It captures non-linear interactions and handles mixed feature types while remaining
          more stable and easier to deploy than heavier deep learning or stacked ensembles.

    - How to frame model performance for business stakeholders:
        * Turning confusion matrices and thresholds into clear statements about expected
          defaults avoided, good customers approved, and net economic impact.

repo_structure_and_usage: |
  Suggested repository layout:

    - data/              : Kaggle Home Credit CSV files
    - notebooks_or_qmd/  : main Quarto or R Markdown file with the full pipeline
    - models/            : saved GBM model objects and calibration artifacts
    - submission.csv     : final Kaggle submission generated by the GBM model

  To reproduce the winning GBM approach:

    1) Download the Home Credit Default Risk competition data from Kaggle and place the files in data/.
    2) Open the main Quarto/R Markdown analysis in RStudio.
    3) Install required R packages (tidyverse, tidymodels, xgboost, probably, etc.).
    4) Knit/render the document to:
         - load and join the datasets,
         - run the preprocessing recipe,
         - train the down-sampled XGBoost model,
         - apply Platt scaling and generate the cost table, and
         - write submission.csv with predicted default probabilities.
