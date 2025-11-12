import sys
import os
# Add parent directory to path so we can import from data/, network/, utils/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import datetime
import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, LearningRateScheduler

from network.classifier import getTrkPredictor
from data.dataloader_multi import loadTrainTestData
from utils.args import get_args
from utils.io import save_settings


def main():
    args = get_args()
    
    datetimenow = datetime.datetime.now().strftime("%B %d, %Y: %H:%M:%S")
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Hyperparameters
    layers = args.layers
    lr = args.lr
    nepochs = args.epochs
    batchsize = args.batchsize
    loss = binary_crossentropy
    optim = Adam(learning_rate=lr)
    
    # Data/eval config
    sample = args.sample
    xscore = args.xscore
    roul_type = args.rouletter
    temperature = args.temperature
    
    sv_dir = sample + "/" + xscore + "/" + roul_type
    if xscore[:7] == "density":
        sv_dir += "/" + args.evtmix
    if (roul_type.casefold() == 'smart') and temperature is not None:
        sv_dir = sv_dir + "/" + str(temperature)
    
    log_sv_dir = f"results/{sv_dir}/logs/"
    os.makedirs(log_sv_dir, exist_ok=True)
    
    # Load processed data and create train/test split
    path = "data"
    sample = args.sample
    split = args.trainsplit
    use_density = bool(args.usedensity)
    
    print(f"Loading processed data for sample: {sample}")
    
    # Optional: Use subset for balanced training
    matched_size = args.matched_size if hasattr(args, 'matched_size') else None
    unmatched_size = args.unmatched_size if hasattr(args, 'unmatched_size') else None
    
    if matched_size or unmatched_size:
        print(f"  Training on SUBSET:")
        if matched_size:
            print(f"    Matched samples: {matched_size}")
        if unmatched_size:
            print(f"    Unmatched samples: {unmatched_size}")
    
    F_train, F_test, L_train, L_test, W_train, W_test, feat_list = loadTrainTestData(
        path=path,
        logfile=None,  # Will log to file below
        sample=sample,
        split=split,
        use_densities=use_density,
        pos_sample_size=matched_size,
        neg_sample_size=unmatched_size
    )
    
    numFeatures = len(feat_list)
    
    # Logging
    logfile = open(f"{log_sv_dir}/training_log.txt", "w")
    print("\n\n", file=logfile)
    print("=*" * 60, file=logfile, flush=True)
    print("*=" * 60, file=logfile, flush=True)
    print(f"\nTraining started: {datetimenow}", file=logfile, flush=True)
    print(f"Training data shape: {F_train.shape}", file=logfile, flush=True)
    print(f"Number of features: {numFeatures}", file=logfile, flush=True)
    print(f"Feature list: {feat_list}", file=logfile, flush=True)
    
    # Save settings
    save_settings(sv_dir, args)
    
    # Build model
    print(
        "\nInitialising and training classifier from scratch.",
        file=logfile,
        flush=True,
    )
    trkPredictor = getTrkPredictor(numFeatures, layers)
    trkPredictor.compile(loss=loss, optimizer=optim, metrics=["accuracy"])
    print(trkPredictor.summary(), file=logfile, flush=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        './results/saved_models/model-{epoch:03d}.ckpt',
        monitor='val_loss',
        verbose=2,
        save_best_only=True,
        mode='min'
    )
    csvLogger = CSVLogger(f"{log_sv_dir}/training_logger.csv", separator=",", append=False)
    earlyStopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        verbose=1,
        restore_best_weights=True
    )
    
    def scheduler(epoch, lr):
        if epoch < 2:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    
    learningrate = LearningRateScheduler(scheduler)
    callback = [csvLogger, earlyStopping, checkpoint, learningrate]
    
    # Train
    print("Training Begins", file=logfile, flush=True)
    train_start = time.time()
    
    history = trkPredictor.fit(
        F_train,
        L_train,
        epochs=nepochs,
        sample_weight=W_train,
        validation_data=(F_test, L_test, W_test),
        batch_size=batchsize,
        callbacks=callback,
    )
    
    train_end = time.time()
    history = history.history
    
    print(
        f"Finished training in {str(datetime.timedelta(seconds=round(train_end-train_start)))}",
        file=logfile,
        flush=True
    )
    
    # Save model and history
    cl_dir = f"results/{sample}/classifier"
    os.makedirs(cl_dir, exist_ok=True)
    
    trkPredictor.save(f"{cl_dir}/classifier.h5")
    with open(f"{cl_dir}/history.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)
    
    print(f"\n✓ Model saved to {cl_dir}/classifier.h5", file=logfile, flush=True)
    print(f"✓ Training history saved to {cl_dir}/history.pkl", file=logfile, flush=True)
    print("="*60, file=logfile, flush=True)
    
    logfile.close()
    
    # Also print to console
    print(f"✓ Training complete!")
    print(f"  Training time: {str(datetime.timedelta(seconds=round(train_end-train_start)))}")
    print(f"  Model saved to: {cl_dir}/classifier.h5")
    print(f"  Logs saved to: {log_sv_dir}")


if __name__ == "__main__":
    main()
