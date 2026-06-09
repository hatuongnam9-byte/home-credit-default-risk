# Home Credit Default Risk - ML Project

The goal of this project is to develop a machine learning pipeline that can predict whether a loan applicant will default on their loan or experience payment difficulties. Many individuals, especially those with little or no credit history, struggle to obtain loans. Home Credit Group uses alternative data (like telecom and transactional data) to better assess repayment capabilities.

Challenges include:
- Handling highly imbalanced datasets where default cases are only ~8% of total applications.
- Processing and aggregating data from multiple relational tables (`bureau`, `previous_application`, `installments_payments`).
- Optimizing memory usage when handling large datasets (several gigabytes in size).

---

## Step 1: Importing necessary Libraries

We begin by importing the required data manipulation, visualization, and modeling libraries:

```python
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Step 2: Loading and Cleaning the Data

We load the training and testing datasets and clean anomalous features. For instance, in `DAYS_EMPLOYED`, the value `365243` represents an anomaly (approximately 1000 years of employment) and is replaced with `NaN`:

```python
train_path = 'data/application_train.csv'
test_path = 'data/application_test.csv'

df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Concatenate train and test sets for unified preprocessing
df = pd.concat([df, test_df], ignore_index=True)

# Replace anomalous DAYS_EMPLOYED values
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
```

---

## Step 3: Domain-Specific Feature Engineering

We design key financial and demographic ratios from the application data:

```python
# Create custom credit scoring ratios
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['INCOME_CREDIT_PERCENT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

# Aggregate external scoring sources
ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
df['EXT_SOURCES_PROD'] = df[ext_sources].prod(axis=1)
df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)
df['EXT_SOURCES_STD'] = df[ext_sources].std(axis=1).fillna(df['EXT_SOURCES_STD'].mean())
```

---

## Step 4: Aggregating Data from Relational Tables

To capture historical borrowing behavior, we group and aggregate records from secondary tables:

### 1. Bureau Data (CIC History)
```python
# Aggregate bureau details grouped by client ID
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum']
})
bureau_agg.columns = pd.Index(["BUREAU_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
df = df.join(bureau_agg, how='left', on='SK_ID_CURR')
```

### 2. Previous Applications
```python
# Aggregate past applications with Home Credit
prev_agg = prev.groupby('SK_ID_CURR').agg({
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_APPLICATION': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['mean', 'sum']
})
prev_agg.columns = pd.Index(["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
df = df.join(prev_agg, how='left', on='SK_ID_CURR')
```

---

## Step 5: Building and Training the Model

We implement a **Stratified 5-Fold Cross-Validation** loop and train a **LightGBM Classifier** using hyperparameters optimized for tabular credit data:

```python
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])

features = [col for col in train_df.columns if col not in ['TARGET', 'SK_ID_CURR']]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[trn_idx], y[trn_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        max_depth=8,
        subsample=0.87156,
        colsample_bytree=0.949703,
        reg_alpha=0.0415454,
        reg_lambda=0.0735294,
        n_jobs=-1,
        verbosity=-1
    )
    
    # Fit model with early stopping
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)]
    )
    
    oof_preds[val_idx] = clf.predict_proba(X_val, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
```

---

## Step 6: Evaluating the Model

The main evaluation metric is the **Area Under the ROC Curve (ROC-AUC)**. After cross-validation, we evaluate the overall out-of-fold validation score:

```python
overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall Out-of-Fold ROC-AUC: {overall_auc:.6f}")
```

We also visualize the relative importance of features to understand what factors drive credit default prediction:

```python
# Get feature importance dataframe
importance_df = pd.DataFrame()
importance_df["feature"] = features
importance_df["importance"] = clf.feature_importances_

# Plot top 40 features
cols = importance_df.groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
sns.barplot(x="importance", y="feature", data=importance_df[importance_df.feature.isin(cols)])
plt.title('LightGBM Features (avg over folds)')
plt.show()
```

---

## 🏃 Getting Started & How to Run

1. Clone the repository and install required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Put the Kaggle dataset files inside the `data/` directory.
3. Run the entire pipeline:
   ```bash
   python run.py --mode all
   ```
4. Find the training log output, saved LightGBM models, and the feature importance plots in the `models/` directory. The final test predictions will be saved as `submission.csv`.
