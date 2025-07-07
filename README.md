# PP4 – Delivery Delay Prediction Using Supervised Learning

---

## What Was Done

1. **Data Preprocessing**:
   - Categorical variables were one-hot encoded.
   - Numerical features were standardized using `StandardScaler`.
   - Data was split into **training (60%)**, **validation (20%)**, and **test (20%)** sets using stratified sampling.

2. **EDA**:
   - Extensive visualizations with `histplot`, `boxplot`, and class-specific mean overlays.
   - Correlation heatmap with encoded and scaled variables.

3. **Model Training & Tuning**:
   - Built a modular pipeline with `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`.
   - Tuned five models using `GridSearchCV` with `StratifiedKFold` cross-validation:
     - Logistic Regression
     - XGBoost
     - Support Vector Classifier
     - CatBoost
     - K-Nearest Neighbors
   - Evaluation on the **test set** using:
     - Accuracy, Recall, ROC AUC
     - Classification reports
     - Confusion matrices
     - ROC curves
     - Bar chart comparison of accuracy

4. **Model Selection**:
   - XGBoost was selected based on validation and test performance.
   - Final evaluation on the validation set confirmed strong generalization.

---

## Key Features of the Dataset

- The dataset contains **10,999 shipment records** with both **numerical** and **categorical** features.
- **Target variable**: `Reached.on.Time_Y.N` (1 = Delayed, 0 = On Time)
- **Numerical features**:
  - `Customer_care_calls`
  - `Customer_rating`
  - `Cost_of_the_Product`
  - `Prior_purchases`
  - `Discount_offered`
  - `Weight_in_gms`
- **Categorical features**:
  - `Warehouse_block`
  - `Mode_of_Shipment`
  - `Product_importance`
  - `Gender`

---

## Key Findings

- The dataset showed **moderate noise**, confirmed by:
  - Low feature-target correlation (~0.4 max)
  - Some class imbalance (60% delayed, 40% on time)
  - Overlapping distributions in feature plots (histograms, boxplots)
- `Discount_offered` and `Weight_in_gms` were among the **most predictive features**.
- Tree-based models (e.g., **XGBoost**, **CatBoost**) significantly outperformed linear ones (e.g., Logistic Regression) due to the data's nonlinear nature and mixed feature types.
- The best model, **XGBoost**, achieved:
  - High **F1 score**, **ROC AUC**, and **Recall**
  - Good separation of delayed vs. on-time predictions in ROC curve and confusion matrix

---

## Why These Parameters Were Chosen and Performed Best

The parameter ranges used during tuning were selected based on the nature of the dataset, the type of models, and prior knowledge of their behavior. For example, for tree-based models like XGBoost and Random Forest, `n_estimators` was varied between 50 and 200 to balance model complexity with training time. `max_depth` and `min_child_weight` were included to control overfitting, especially important given that some features (e.g., `Discount_offered`) showed strong class influence while others had weak predictive power. For models like Logistic Regression and SVC, hyperparameters such as `C`, `penalty`, and `kernel` were tested to control regularization and decision boundary flexibility.

The tuning grids were deliberately broad to allow the search algorithm to explore both conservative and expressive configurations. This was crucial due to the dataset’s moderately noisy nature and the overlapping class distributions observed in the EDA. The best-performing model, XGBoost, benefited from a combination of moderate depth and learning rate, which helped it generalize well without overfitting to noise. Additionally, the tree-based models were well-suited to the mix of categorical and numerical features, and to the nonlinear relationships in the data.

Overall, the parameters that worked best allowed the model to focus on the most important features while ignoring irrelevant variations, leading to superior performance in F1 score and ROC AUC. These results validate the tuning approach and demonstrate that the model architecture and hyperparameters aligned well with the dataset’s structure and challenges.

---

### Requirements

**requirements.txt** :
```
python == 3.13.2
pandas == 2.2.3
matplotlib == 3.10.0
seaborn == 0.13.2
numpy == 2.2.5
scikit-learn == 1.7.0
xgboost == 3.0.2
catboost == 1.2.8
```