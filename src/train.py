import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def train_lgb_model(data_path, output_dir='models', submission_path='submission.csv'):
    print(f"Loading preprocessed dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate train and test sets
    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    if len(train_df) == 0:
        print("Error: Train dataset is empty. Ensure 'TARGET' is not null in training data.")
        return
        
    # Extract IDs and target
    train_ids = train_df['SK_ID_CURR'].values
    test_ids = test_df['SK_ID_CURR'].values
    y = train_df['TARGET'].values
    
    # Drop identifier columns
    features = [col for col in train_df.columns if col not in ['TARGET', 'SK_ID_CURR']]
    
    X = train_df[features].copy()
    X_test = test_df[features].copy()
    
    # Chuẩn hóa tên cột để loại bỏ các ký tự đặc biệt gây lỗi LightGBM
    import re
    cleaned_features = [re.sub(r'[ :,{}="\'\[\]\(\)]+', '_', col) for col in features]
    X.columns = cleaned_features
    X_test.columns = cleaned_features
    
    print(f"Number of features used for training: {len(features)}")
    
    # Initialize Stratified K-Fold
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Arrays to store out-of-fold and test predictions
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0]) if len(test_df) > 0 else None
    
    feature_importance_df = pd.DataFrame()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Train LGBM model for each fold
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"\n--- Fold {fold_ + 1} ---")
        
        X_train, y_train = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # Configure LightGBM Classifier
        # Tuned hyperparameters for Home Credit Default Risk
        min_child_weight = 39.3259775
        min_split_gain = 0.0222415
        min_child_samples = 20
        
        if len(X_train) < 1000:
            min_child_weight = 0.1
            min_split_gain = 0.0
            min_child_samples = 5
            print(f"  [Info] Small dataset detected ({len(X_train)} samples). Adjusting hyperparameters (min_child_weight=0.1, min_split_gain=0.0, min_child_samples=5) to allow splitting.")

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
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            random_state=42 + fold_,
            n_jobs=-1,
            verbosity=-1
        )
        
        # Train with early stopping
        # Using newer callbacks style compatible with lightgbm >= 3.3.0
        callbacks = [
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=200)
        ]
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'valid'],
            callbacks=callbacks
        )
        # Predict on validation set
        oof_preds[val_idx] = clf.predict_proba(X_val, num_iteration=clf.best_iteration_)[:, 1]
        
        # Predict on test set
        if sub_preds is not None:
            sub_preds += clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
            
        # Record feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        # Save model checkpoint
        model_save_path = os.path.join(output_dir, f'lgb_fold_{fold_ + 1}.model')
        clf.booster_.save_model(model_save_path)
        print(f"Saved model for fold {fold_ + 1} to {model_save_path}")
        
        # Compute fold ROC-AUC and PR-AUC scores
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        from sklearn.metrics import precision_recall_curve, auc
        prec_fold, rec_fold, _ = precision_recall_curve(y_val, oof_preds[val_idx])
        fold_pr_auc = auc(rec_fold, prec_fold)
        print(f"Fold {fold_ + 1} Validation ROC-AUC: {fold_auc:.6f} | PR-AUC (CPR): {fold_pr_auc:.6f}")
        
        # Clean up memory
        del X_train, y_train, X_val, y_val, clf
        gc.collect()
        
    # Calculate global OOF ROC-AUC and PR-AUC
    oof_auc = roc_auc_score(y, oof_preds)
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y, oof_preds)
    oof_pr_auc = auc(recall, precision)
    print(f"\n==========================================")
    print(f"Overall Out-of-Fold ROC-AUC: {oof_auc:.6f}")
    print(f"Overall Out-of-Fold PR-AUC (CPR): {oof_pr_auc:.6f}")
    print(f"==========================================")
    # ===== Lưu kết quả =====
    
 
    # Create submission file if test predictions exist
    if sub_preds is not None:
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': sub_preds})
        submission.to_csv(submission_path, index=False)
        print(f"Created submission file at {submission_path}")
        
    # Display and save feature importance
    display_importances(feature_importance_df, output_dir)
    
    # Plot and save PR-AUC Curve
    plot_pr_curve(y, oof_preds, output_dir)
    
    return oof_pr_auc

def plot_pr_curve(y_true, y_preds, output_dir):
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_true, y_preds)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='dodgerblue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Baseline is the fraction of positive samples (default rate)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='crimson', lw=2, linestyle='--', label=f'Baseline (Rate = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Out-of-Fold Precision-Recall (PR) Curve')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'auc_pr_curve.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved PR curve plot to {plot_path}")

def display_importances(feature_importance_df, output_dir):
    cols = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:40].index)
            
    best_features = feature_importance_df[feature_importance_df.feature.isin(cols)]
    
    plt.figure(figsize=(10, 14))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Also save importances as CSV
    importance_csv_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance_df.groupby("feature").mean().sort_values(by="importance", ascending=False).to_csv(importance_csv_path)
    print(f"Saved feature importance plot to {plot_path} and CSV to {importance_csv_path}")

if __name__ == '__main__':
    # Test path or direct execution
    # Xác định đường dẫn động dựa trên vị trí của file script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'processed_sample.csv'))
    output_dir = os.path.abspath(os.path.join(script_dir, '..', 'models_test'))
    
    train_lgb_model(data_path, output_dir=output_dir)
