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
    
    X = train_df[features]
    X_test = test_df[features]
    
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
        
        # Compute fold ROC-AUC score
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold_ + 1} Validation ROC-AUC: {fold_auc:.6f}")
        
        # Clean up memory
        del X_train, y_train, X_val, y_val, clf
        gc.collect()
        
    # Calculate global OOF ROC-AUC
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n==========================================")
    print(f"Overall Out-of-Fold ROC-AUC: {oof_auc:.6f}")
    print(f"==========================================")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({'SK_ID_CURR': train_ids, 'TARGET': oof_preds})
    oof_df.to_csv(os.path.join(output_dir, 'oof_predictions.csv'), index=False)
    print("Saved OOF predictions to models/oof_predictions.csv")
    
    # Create submission file if test predictions exist
    if sub_preds is not None:
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': sub_preds})
        submission.to_csv(submission_path, index=False)
        print(f"Created submission file at {submission_path}")
        
    # Display and save feature importance
    display_importances(feature_importance_df, output_dir)
    
    return oof_auc

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
    train_lgb_model('data/processed_sample.csv', output_dir='models_test')
