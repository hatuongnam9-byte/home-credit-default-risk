import os
import gc
import numpy as np
import pandas as pd

# Helper function to one-hot encode categorical variables
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Memory usage reduction helper
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage decreased to {end_mem:.2f} MB (reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%)")
    return df

# 1. Process application_train.csv and application_test.csv
def process_application(data_dir, num_rows=None):
    train_path = os.path.join(data_dir, 'application_train.csv')
    test_path = os.path.join(data_dir, 'application_test.csv')
    
    print(f"Processing application train/test... (num_rows={num_rows})")
    df = pd.read_csv(train_path, nrows=num_rows)
    test_df = pd.read_csv(test_path, nrows=num_rows)
    
    print(f"Train shape: {df.shape}, Test shape: {test_df.shape}")
    df = pd.concat([df, test_df], ignore_index=True)
    
    # Remove anomalous data
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # Engineer domain-specific features
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERCENT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # Interactions of External Sources
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_PROD'] = df[ext_sources].prod(axis=1)
    df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)
    df['EXT_SOURCES_STD'] = df[ext_sources].std(axis=1)
    df['EXT_SOURCES_STD'] = df['EXT_SOURCES_STD'].fillna(df['EXT_SOURCES_STD'].mean())
    
    # Categorical variables encoding
    df, cat_cols = one_hot_encoder(df, nan_as_category=True)
    
    del test_df
    gc.collect()
    
    return reduce_mem_usage(df)

# 2. Process bureau.csv and bureau_balance.csv
def process_bureau(data_dir, num_rows=None):
    bureau_path = os.path.join(data_dir, 'bureau.csv')
    bb_path = os.path.join(data_dir, 'bureau_balance.csv')
    
    if not os.path.exists(bureau_path):
        print("bureau.csv not found, skipping bureau features.")
        return None
        
    print("Processing bureau and bureau balance...")
    
    # Process bureau_balance.csv first if it exists
    if os.path.exists(bb_path):
        bb = pd.read_csv(bb_path, nrows=num_rows)
        bb, bb_cat = one_hot_encoder(bb, nan_as_category=True)
        
        # Numeric aggregations
        num_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        # Categorical aggregations
        cat_aggregations = {}
        for col in bb_cat:
            cat_aggregations[col] = ['mean']
            
        bb_agg = bb.groupby('SK_ID_BUREAU').agg({**num_aggregations, **cat_aggregations})
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        
        # Merge back to bureau
        bureau = pd.read_csv(bureau_path, nrows=num_rows)
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        del bb, bb_agg 
        gc.collect()
    else:
        bureau = pd.read_csv(bureau_path, nrows=num_rows)
        
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=True)
    
    # Bureau numeric aggregations
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'DAYS_ENDDATE_FACT': ['mean']
    }
    
    # Bureau categorical aggregations
    cat_aggregations = {}
    for col in bureau_cat:
        cat_aggregations[col] = ['mean']
    
    # Merge column-specific aggregations from bureau_balance if present
    for col in bureau.columns:
        if 'MONTHS_BALANCE' in col or any(c in col for c in ['STATUS_']):
            num_aggregations[col] = ['mean']
            
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(["BUREAU_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # Active Bureau Loans - count & specific features
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg({'DAYS_CREDIT': ['mean', 'max']})
    active_agg.columns = pd.Index(["BUREAU_ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left')
    
    # Closed Bureau Loans - count & specific features
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg({'DAYS_CREDIT': ['mean', 'max']})
    closed_agg.columns = pd.Index(["BUREAU_CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left')
    
    del bureau, active, closed, active_agg, closed_agg
    gc.collect()
    
    return reduce_mem_usage(bureau_agg)

# 3. Process previous_applications.csv
def process_previous_applications(data_dir, num_rows=None):
    prev_path = os.path.join(data_dir, 'previous_application.csv')
    if not os.path.exists(prev_path):
        print("previous_application.csv not found, skipping.")
        return None
        
    print("Processing previous applications...")
    prev = pd.read_csv(prev_path, nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    
    # Replace anomalous days values with NaN
    days_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols:
        prev[col] = prev[col].replace(365243, np.nan)
        
    # Simple feature engineering
    prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    
    cat_aggregations = {}
    for col in cat_cols:
        cat_aggregations[col] = ['mean']
        
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    # Approved Applications - specific aggregations
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg({'AMT_CREDIT': ['mean', 'sum']})
    approved_agg.columns = pd.Index(["APPROVED_" + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left')
    
    # Refused Applications - specific aggregations
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg({'AMT_CREDIT': ['mean', 'sum']})
    refused_agg.columns = pd.Index(["REFUSED_" + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left')
    
    del prev, approved, refused, approved_agg, refused_agg
    gc.collect()
    
    return reduce_mem_usage(prev_agg) 

# 4. Process installments_payments.csv
def process_installments(data_dir, num_rows=None):
    inst_path = os.path.join(data_dir, 'installments_payments.csv')
    if not os.path.exists(inst_path):
        print("installments_payments.csv not found, skipping.")
        return None
        
    print("Processing installment payments...")
    ins = pd.read_csv(inst_path, nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
     
    # Feature engineering: payment differences and delays
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    
    for col in cat_cols:
        aggregations[col] = ['mean']
        
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    del ins
    gc.collect()
    
    return reduce_mem_usage(ins_agg)

# Main preprocessing entry function
def generate_features(data_dir, num_rows=None, output_path='processed_data.csv'):
    print(f"Starting feature engineering on datasets in: {data_dir}")
    
    # 1. Process application
    df = process_application(data_dir, num_rows)
    
    # 2. Process bureau
    bureau_df = process_bureau(data_dir, num_rows)
    if bureau_df is not None:
        print(f"Merging bureau features... Shape: {bureau_df.shape}")
        df = df.join(bureau_df, how='left', on='SK_ID_CURR')
        del bureau_df
        gc.collect()
        
    # 3. Process previous applications
    prev_df = process_previous_applications(data_dir, num_rows)
    if prev_df is not None:
        print(f"Merging previous applications... Shape: {prev_df.shape}")
        df = df.join(prev_df, how='left', on='SK_ID_CURR')
        del prev_df
        gc.collect()
        
    # 4. Process installments
    inst_df = process_installments(data_dir, num_rows)
    if inst_df is not None:
        print(f"Merging installment features... Shape: {inst_df.shape}")
        df = df.join(inst_df, how='left', on='SK_ID_CURR')
        del inst_df
        gc.collect()
        
    print(f"Final preprocessed dataset shape: {df.shape}")
    
    # Save the output
    df.to_csv(output_path, index=False)
    print(f"Successfully saved engineered features to {output_path}")
    
    del df
    gc.collect()

if __name__ == '__main__':
    import sys
    # Xác định đường dẫn động đến thư mục data dựa vào vị trí của file script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    
    # Mặc định chạy thử nghiệm 10000 dòng, thêm tham số '--full' để xử lý toàn bộ dữ liệu
    num_rows = 10000
    output_file = 'processed_sample.csv'
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        num_rows = None
        output_file = 'processed_data.csv'
        print("Running full preprocessing...")
    else:
        print("Running sample preprocessing (10,000 rows). Add '--full' to process all data.")
        
    output_path = os.path.join(data_dir, output_file)
    generate_features(data_dir, num_rows=num_rows, output_path=output_path)
