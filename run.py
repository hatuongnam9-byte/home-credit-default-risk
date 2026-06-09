import os
import argparse
from src.preprocessing import generate_features
from src.train import train_lgb_model

def main():
    parser = argparse.ArgumentParser(description="Home Credit Default Risk ML Pipeline")
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['preprocess', 'train', 'all'],
        help="Pipeline mode: 'preprocess' to build features, 'train' to fit models, 'all' for both."
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help="Directory containing the raw CSV dataset files."
    )
    parser.add_argument(
        '--num-rows',
        type=int,
        default=None,
        help="Optional row limit to load from raw CSVs (for fast testing/debugging)."
    )
    parser.add_argument(
        '--output-data',
        type=str,
        default='data/processed_data.csv',
        help="Path where preprocessed feature matrix will be saved."
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help="Directory where models, log files and OOF predictions will be saved."
    )
    parser.add_argument(
        '--submission',
        type=str,
        default='submission.csv',
        help="Path where the final Kaggle submission CSV file will be written."
    )
    
    args = parser.parse_args()
    
    # Verify raw data directory exists
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
        print(f"Warning: Data directory '{args.data_dir}' did not exist, created it.")
        print("Please extract your Kaggle Home Credit dataset CSV files into this directory.")
        if args.mode in ['preprocess', 'all']:
            print("Aborting preprocessing since data files are missing.")
            return

    # Run Preprocessing Mode
    if args.mode in ['preprocess', 'all']:
        print("\n==========================================")
        print("STEP 1: Run Data Preprocessing & Feature Engineering")
        print("==========================================")
        
        # Ensure the output directory for processed data exists
        output_dir = os.path.dirname(args.output_data)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        generate_features(
            data_dir=args.data_dir,
            num_rows=args.num_rows,
            output_path=args.output_data
        )

    # Run Model Training Mode
    if args.mode in ['train', 'all']:
        print("\n==========================================")
        print("STEP 2: Run Stratified K-Fold LightGBM Model Training")
        print("==========================================")
        
        if not os.path.exists(args.output_data):
            print(f"Error: Processed data file '{args.output_data}' not found. Run preprocessing first.")
            return
            
        train_lgb_model(
            data_path=args.output_data,
            output_dir=args.model_dir,
            submission_path=args.submission
        )
        
    print("\nPipeline execution finished successfully!")

if __name__ == '__main__':
    main()
