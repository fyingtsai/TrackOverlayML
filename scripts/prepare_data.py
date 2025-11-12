import sys
import os
# Add parent directory to path so we can import from data/, network/, utils/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import datetime

from data.dataloader_multi import prepareData
from utils.args import get_args


def main():
    args = get_args()
    
    datetimenow = datetime.datetime.now().strftime("%B %d, %Y: %H:%M:%S")
    
    # Settings
    np.random.seed(args.seed)
    split = args.trainsplit
    use_density = bool(args.usedensity)
    sample = args.sample
    path = 'data'
    
    # Output directory for processed data
    processed_dir = f"data/processed/{sample}"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create log file
    log_file = open(f"{processed_dir}/prepare_data_log.txt", "w")
    print(f"Data preparation started: {datetimenow}", file=log_file, flush=True)
    print("="*60, file=log_file, flush=True)
    
    # Load and process data
    print(f"Loading data from {path}/{sample}.h5", file=log_file, flush=True)
    print(f"Train/test split: {split}/{1-split}", file=log_file, flush=True)
    print(f"Use density features: {use_density}", file=log_file, flush=True)
    
    # Prepare the raw data (load CSVs, compute features, save to HDF5)
    fullData_pos, fullData_neg, feat_list = prepareData(
        path=path,
        logfile=log_file,
        sample=sample,
        use_densities=use_density
    )
    
    print(
        f"\nData preparation complete:",
        file=log_file,
        flush=True,
    )
    print(f"  Matched samples (TargetLabel=1): {fullData_pos.shape}", file=log_file, flush=True)
    print(f"  Unmatched samples (TargetLabel=0): {fullData_neg.shape}", file=log_file, flush=True)
    print(f"  Number of features: {len(feat_list)}", file=log_file, flush=True)
    print(f"  Feature list: {feat_list}", file=log_file, flush=True)
    
    # Save metadata for training/evaluation scripts
    print(f"\nSaving metadata to {processed_dir}", file=log_file, flush=True)
    metadata = {
        "feat_list": feat_list,
        "num_features": len(feat_list),
        "sample": sample,
        "split": split,
        "use_density": use_density,
        "seed": args.seed,
        "preparation_date": datetimenow,
        "pos_samples": fullData_pos.shape[0],
        "neg_samples": fullData_neg.shape[0]
    }
    with open(f"{processed_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Data preparation complete!", file=log_file, flush=True)
    print(f"  - Processed data: {path}/matched_{sample}_data.h5, {path}/unmatched_{sample}_data.h5", file=log_file, flush=True)
    print(f"  - Metadata: {processed_dir}/metadata.pkl", file=log_file, flush=True)
    print("="*60, file=log_file, flush=True)
    
    log_file.close()
    
    # Also print to console
    print(f"✓ Data preparation complete!")
    print(f"  Matched samples (TargetLabel=1): {fullData_pos.shape[0]}")
    print(f"  Unmatched samples (TargetLabel=0): {fullData_neg.shape[0]}")
    print(f"  Features: {len(feat_list)}")
    print(f"  Metadata saved to: {processed_dir}/metadata.pkl")


if __name__ == "__main__":
    main()
