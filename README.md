[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# TrackOverlayML Pipeline

> **Note**: Refactored and documented with GitHub Copilot assistance. Based on ATLAS Collaboration code for ML-driven track overlay routing.

## Overview

Train a neural network to intelligently route ATLAS simulation events:
- **MC-overlay**: Full simulation (accurate but slow)
- **Track-overlay**: Fast simulation (approximation)
- **Goal**: Use Track-overlay when it matches MC-overlay (MatchProb > 0.5), otherwise use MC-overlay

## Data Access

The framework requires ATLAS simulation data in specific formats:

### Option 1: Pre-processed HDF5 files (Ready for training)
```
/eos/user/f/fatsai/TrackOverlayDATA/matched_JZ7W_data.h5
/eos/user/f/fatsai/TrackOverlayDATA/unmatched_JZ7W_data.h5
```

### Option 2: Raw CSV files (For full preprocessing pipeline)
```
/eos/user/f/fatsai/TrackOverlayDATA/MCOverlay_JZ7/*.csv
/eos/user/f/fatsai/TrackOverlayDATA/TrackOverlay_JZ7/*.csv
```

**Access:** These datasets are stored on CERN EOS and require ATLAS collaboration access rights.

## Quick Start

### Using Singularity

```bash
# Pull the container (only needed once)
singularity pull docker://fyingtsai/dsnnr_4gpu:v5
or on Perlmutter
podman-hpc pull docker://fyingtsai/dsnnr_4gpu:v5

```bash
# Full pipeline
singularity exec dsnnr_4gpu_v5.sif python scripts/run_pipeline.py --sample ttbar --epochs 5

# Or run steps individually (Recommended. Run each step individually for easier debugging and better control)
singularity exec dsnnr_4gpu_v5.sif python scripts/prepare_data.py --sample ttbar --path data
singularity exec dsnnr_4gpu_v5.sif python scripts/train_model.py --sample ttbar --path data --epochs 5
singularity exec dsnnr_4gpu_v5.sif python scripts/evaluate_model.py --sample ttbar
```

## More Training examples
# Train on balanced 10k + 10k
python scripts/train_model.py --sample ttbar \
    --matched_size 10000 --unmatched_size 10000

# Train on realistic imbalanced ratio (1:10)
python scripts/train_model.py --sample ttbar \
    --matched_size 5000 --unmatched_size 50000

# Use all matched, but limit unmatched
python scripts/train_model.py --sample ttbar \
    --unmatched_size 20000
#######
# Full pipeline with balanced training
python scripts/run_pipeline.py --stage all --sample ttbar \
    --path data --matched_size 5000 --unmatched_size 5000 --epochs 20

# Full pipeline with cross-sample evaluation
python scripts/run_pipeline.py --stage all --sample ttbar \
    --eval_sample JZ7W

# Just train on subset
python scripts/run_pipeline.py --stage train --sample ttbar \
    --matched_size 10000 --unmatched_size 10000

## Project Structure

```
TrackOverlayML/
├── data/                       # Data directory (--path to customize)
│   ├── MC-overlay_{sample}/    # MC workflow CSVs (required, if not yet have a h5 dataframe)
│   ├── Track-overlay_{sample}/ # Track workflow CSVs (required, if not yet have a h5 dataframe)
│   ├── matched_{sample}_data.h5    # Good matches (pre-created)
│   └── unmatched_{sample}_data.h5  # Poor matches (pre-created)
├── scripts/                    # Main entry points
│   ├── prepare_data.py         # Merge MC/Track, compute features
│   ├── train_model.py          # Train classifier
│   ├── evaluate_model.py       # Evaluate performance
│   └── run_pipeline.py         # Run all steps
├── network/classifier.py       # Model architecture
├── utils/                      # Evaluation & plotting
└── results/                    # Outputs (models, plots, logs)
```

## Usage

### Individual Steps (best for those just getting started)

```bash
# Step 1: Prepare data (merge MC/Track workflows)
python scripts/prepare_data.py --sample ttbar --trainsplit 0.8

# Step 2: Train model
python scripts/train_model.py --sample ttbar --epochs 200

# Step 3: Evaluate (same sample)
python scripts/evaluate_model.py --sample ttbar

# Step 3b: Evaluate on different sample
python scripts/evaluate_model.py --sample ttbar --eval_sample JZ7W
```

### Common Workflows

**Train multiple models on same data:**
```bash
python scripts/prepare_data.py --sample ttbar
python scripts/train_model.py --sample ttbar --layers 32 16 8
python scripts/train_model.py --sample ttbar --layers 64 32 16
```

**Cross-sample evaluation:**
```bash
# Train on ttbar, test on JZ7W
python scripts/train_model.py --sample ttbar
python scripts/prepare_data.py --sample JZ7W
python scripts/evaluate_model.py --sample ttbar --eval_sample JZ7W
```

**Quick evaluation on subset:**
```bash
python scripts/evaluate_model.py --sample ttbar --matched_size 5000 --unmatched_size 50000
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--path` | `data` | Data directory path |
| `--sample` | `JZ7W` | Sample name (ttbar, JZ7W, etc.) |
| `--eval_sample` | None | Different sample for evaluation |
| `--trainsplit` | 0.8 | Train/test split ratio |
| `--epochs` | 200 | Training epochs |
| `--lr` | 0.001 | Learning rate |
| `--layers` | 16 10 8 | Hidden layer sizes |
| `--matched_size` | None | Limit good match samples for eval |
| `--unmatched_size` | None | Limit poor match samples for eval |

Run `python scripts/run_pipeline.py --help` for full list.

## Data Flow

```
MC-overlay_{sample}/        Track-overlay_{sample}/
└── *.csv                   └── *.csv
         ↓                           ↓
         └──── Merge on EventNumber ─┘
                      ↓
          Create labels (MatchProb > 0.5)
                      ↓
    matched_*.h5 (good) & unmatched_*.h5 (poor)
                      ↓
              Train/Test split
                      ↓
            Train classifier
                      ↓
          Evaluate performance
```

## Output

```
results/{sample}/
├── classifier/
│   ├── classifier.h5           # Trained model
│   └── history.pkl             # Training history
├── logs/                       # Logs for each step
└── {xscore}/{rouletter}/
    └── plots/                  # ROC, efficiency, fraction plots
```

## Notes

- **Matched** (TargetLabel=1): MatchProb > 0.5 (Track-overlay accurate)
- **Unmatched** (TargetLabel=0): MatchProb ≤ 0.5 (needs MC-overlay)
- Preprocessed HDF5 files are cached for faster reruns

## Contributing

When making changes:
- Keep function docstrings updated
- Add inline comments for complex physics calculations
- Update this README if workflow changes
