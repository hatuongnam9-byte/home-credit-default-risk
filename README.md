# Home Credit Default Risk - Improved Machine Learning Pipeline

This repository contains an end-to-end, optimized machine learning pipeline for the Kaggle **Home Credit Default Risk** competition. 

It improves upon baseline notebook approaches (which typically score ~0.69 Validation ROC-AUC) by incorporating multi-source relational aggregations, domain-specific feature engineering, stratified cross-validation, and state-of-the-art LightGBM models.

---

## 🚀 Key Improvements & Features

1. **Multi-Source Aggregations**: Instead of using only the main table (`application_train/test.csv`), this pipeline aggregates historical, transactional, and installment records from:
   - `bureau.csv` & `bureau_balance.csv` (Credit Bureau history)
   - `previous_application.csv` (Past applications at Home Credit)
   - `installments_payments.csv` (Repayment behaviors)
2. **Domain Feature Engineering**: Engineered professional credit scoring metrics:
   - `CREDIT_INCOME_PERCENT`: Debt-to-income ratio.
   - `ANNUITY_INCOME_PERCENT`: Monthly repayment burden relative to income.
   - `DAYS_EMPLOYED_PERCENT`: Employment duration relative to age.
   - `EXT_SOURCES_PROD` / `EXT_SOURCES_MEAN` / `EXT_SOURCES_STD`: Interactions between external credit sources.
3. **Robust Stratified 5-Fold Cross-Validation**: Prevents overfitting and ensures reliable local validation on highly imbalanced targets (~8% default rate).
4. **LightGBM Modeling**: Utilizes highly tuned hyperparameters for LightGBM, which handles missing values natively and trains 10x faster than Random Forests or deep learning architectures.
5. **Memory Usage Optimization**: Includes a data downcasting utility (`reduce_mem_usage`) that reduces RAM footprint by up to 60%, allowing the pipeline to execute on standard consumer laptops.

---

## 📁 Repository Structure

```
├── data/                       # Dataset directory (place Kaggle CSVs here)
│   └── processed_data.csv      # Generated feature matrix
├── src/
│   ├── preprocessing.py        # Feature engineering and aggregation logic
│   └── train.py                # Stratified 5-Fold cross-validation and LightGBM training
├── models/                     # Saved fold models, OOF predictions & plots
│   ├── lgb_fold_1.model        # LightGBM booster checkpoint for Fold 1
│   ├── oof_predictions.csv     # Out-of-fold validation predictions
│   ├── feature_importance.png  # Feature importance plot
│   └── feature_importance.csv  # Mean importance score per feature
├── run.py                      # Main pipeline runner (CLI interface)
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/home-credit-default-risk.git
   cd home-credit-default-risk
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Download the CSV files from the [Kaggle Competition Page](https://www.kaggle.com/competitions/home-credit-default-risk/data) and extract them into a `data/` folder in the root directory.

---

## 🏃 How to Run the Pipeline

The pipeline is orchestrated via `run.py` and supports three modes of execution:

### 1. Run Preprocessing & Feature Engineering
Build the processed feature matrix from all raw CSV files:
```bash
python run.py --mode preprocess
```

### 2. Run Model Training & Validation
Train the Stratified 5-Fold LightGBM model on the processed dataset:
```bash
python run.py --mode train
```

### 3. Run the Entire Pipeline
Run both feature engineering and model training sequentially:
```bash
python run.py --mode all
```

### 💡 Extra CLI Options:
- `--num-rows <int>`: Read only the first N rows of raw CSVs (highly recommended for fast testing/debugging).
  ```bash
  python run.py --mode all --num-rows 10000
  ```
- `--data-dir <path>`: Specify a custom raw data directory (default is `data`).
- `--model-dir <path>`: Specify a custom directory to save models (default is `models`).
- `--submission <path>`: Specify a custom path for the submission file (default is `submission.csv`).

---

## 📊 Results & Artifacts

After running the pipeline, the following output files will be generated:
- `models/lgb_fold_*.model`: Saved binary LightGBM models for prediction.
- `models/oof_predictions.csv`: Out-of-Fold validation predictions on the training set.
- `models/feature_importance.png`: Bar chart visual of the top 40 most predictive features.
- `submission.csv`: Ready-to-submit predictions for the Kaggle test set.
