import sys
import os
# Add parent directory to path so we can import from data/, network/, utils/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import datetime
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

from data.dataloader_multi import loadEvaluationData
from utils.evaluator_multi import evaluate_performance, evaluate_event
from utils.args import get_args


def main():
    args = get_args()
    
    datetimenow = datetime.datetime.now().strftime("%B %d, %Y: %H:%M:%S")
    
    # Data/eval config
    sample = args.sample  # Sample used for training (for loading model)
    eval_sample = args.eval_sample if args.eval_sample else args.sample  # Sample to evaluate on
    xscore = args.xscore
    use_density = bool(args.usedensity)
    roul_type = args.rouletter
    temperature = args.temperature
    path = 'data'
    
    sv_dir = sample + "/" + xscore + "/" + roul_type
    if xscore[:7] == "density":
        sv_dir += "/" + args.evtmix
    if (roul_type.casefold() == 'smart') and temperature is not None:
        sv_dir = sv_dir + "/" + str(temperature)
    
    log_sv_dir = f"results/{sv_dir}/logs/"
    os.makedirs(log_sv_dir, exist_ok=True)
    
    # Load evaluation data (no train/test split needed - use entire dataset)
    path = "data"
    
    print(f"Loading evaluation data for sample: {eval_sample}")
    if eval_sample != sample:
        print(f"  Note: Evaluating on different sample than training ({sample} → {eval_sample})")
    
    # You can specify sample sizes to limit evaluation, or None to use all data
    F_eval, L_eval, feat_list, df_eval = loadEvaluationData(
        path=path,
        logfile=None,
        sample=eval_sample,
        use_densities=use_density,
        pos_sample_size=args.matched_size,
        neg_sample_size=args.unmatched_size
    )
    
    print(f"Evaluation data loaded: {F_eval.shape}")
    if args.matched_size or args.unmatched_size:
        print(f"  Using subset: matched={args.matched_size or 'all'}, unmatched={args.unmatched_size or 'all'}")
    
    # Load trained model
    cl_dir = f"results/{sample}/classifier"
    model_path = f"{cl_dir}/classifier.h5"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            f"Please run train_model.py first."
        )
    
    print(f"Loading model from {model_path}")
    trkPredictor = keras.models.load_model(model_path)
    
    # Load training history if available
    history_path = f"{cl_dir}/history.pkl"
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
    else:
        history = {}
        print(f"Warning: Training history not found at {history_path}")
    
    # Logging
    logfile = open(f"{log_sv_dir}/eval_log.txt", "w")
    print("\n\n", file=logfile)
    print("=*" * 60, file=logfile, flush=True)
    print("*=" * 60, file=logfile, flush=True)
    print(f"\nEvaluation started: {datetimenow}", file=logfile, flush=True)
    print(f"Evaluation data shape: {F_eval.shape}", file=logfile, flush=True)
    print(f"Model loaded from: {model_path}", file=logfile, flush=True)
    
    # Track-level evaluation
    print("\n=== Track-level evaluation ===", file=logfile, flush=True)
    evaluate_performance(
        trkPredictor,
        sv_dir,
        feat=F_eval,
        labels=L_eval,
        history=history,
        logfile=logfile,
        isTraining=True
    )
    
    # Event-level evaluation
    print("\n=== Event-level evaluation ===", file=logfile, flush=True)
    
    # Use the dataframe we already loaded
    print(f"Using loaded dataframe with {len(df_eval)} samples", file=logfile, flush=True)
    
    # Create scalers for the evaluation data
    FS = MinMaxScaler()
    CS = MinMaxScaler()  # Dummy conditional scaler if needed
    
    dec_kwargs = {
        "smoothing": temperature,
        "rouletter": roul_type
    }
    
    evaluate_event(
        df_eval,
        trkPredictor,
        use_density,
        (FS, CS),
        sv_dir,
        xscore,
        args.evtmix,
        logfile,
        **dec_kwargs
    )
    
    print("\n✓ Evaluation complete!", file=logfile, flush=True)
    print("="*60, file=logfile, flush=True)
    
    logfile.close()
    
    # Also print to console
    print(f"✓ Evaluation complete!")
    print(f"  Results saved to: results/{sv_dir}")
    print(f"  Logs saved to: {log_sv_dir}")


if __name__ == "__main__":
    main()
