#Imports
import os
import joblib
from keras.models import load_model

import preprocessing
import model_module
import whatif_lstm

#================================================================
#  CONFIG
#================================================================

DATASETS = {
    "1": {
        "path":  "data/augmented_combined.csv",
        "label": "Augmented Combined Dataset",
    },
    "2": {
        "path":  "data/cleaned_sequential_ev_dataset.csv",
        "label": "Cleaned Sequential EV Dataset",
    },
}

CONFIG = {
    "sequence_length": 5,
    "test_size":       0.2,
    "epochs":          100,
    "batch_size":      256,
    "learning_rate":   0.001,
}

PATHS = {
    "deg_model": "saved_models/lstm_degradation.keras",
    "beh_model": "saved_models/lstm_behaviour.keras",
    "deg_pre":   "saved_models/deg_preprocessor.pkl",
}

#================================================================
#  TERMINAL INTERFACE
#================================================================

def select_dataset():
    """
    Prompts the user to pick a dataset from the list.
    Returns the file path and label of the chosen dataset.
    """
    print("\n  Select a dataset to use:")
    print("  ─────────────────────────────────────────────")
    for key, info in DATASETS.items():
        print(f"  {key}. {info['label']}")
        print(f"     {info['path']}")
    print("  ─────────────────────────────────────────────")

    while True:
        choice = input("\n  Enter choice (1 or 2): ").strip()
        if choice in DATASETS:
            selected = DATASETS[choice]
            print(f"\n  Selected: {selected['label']}")
            return selected["path"], selected["label"]
        else:
            print("  Invalid choice. Please enter 1 or 2.")


def select_retrain():
    """
    Prompts the user to choose between retraining or loading saved models.
    Data processing always runs regardless of this choice.
    """
    print("\n  Select run mode:")
    print("  ─────────────────────────────────────────────")
    print("  1. Train from scratch  (runs CV + full training, saves models)")
    print("  2. Load saved models   (skips training, runs what-if only)")
    print("  ─────────────────────────────────────────────")

    while True:
        choice = input("\n  Enter choice (1 or 2): ").strip()
        if choice == "1":
            print("\n  Mode: Training from scratch.")
            return True
        elif choice == "2":
            print("\n  Mode: Loading saved models.")
            return False
        else:
            print("  Invalid choice. Please enter 1 or 2.")

#================================================================
#  LOAD DATA
#================================================================

def load_data(csv_path):
    df = preprocessing.load_dataset(csv_path)
    df = preprocessing.add_engineered_features(df)
    return df

#================================================================
#  TRAIN
#================================================================

def train(df):
    deg_model, deg_pre = model_module.train_degradation(df, CONFIG, PATHS)
    model_module.train_behaviour(df, CONFIG, PATHS)
    return deg_model, deg_pre

#================================================================
#  LOAD SAVED MODELS
#================================================================

def load_saved_models():
    missing = [k for k, p in PATHS.items() if not os.path.exists(p)]

    if missing:
        print(f"\n  ERROR: Missing saved files: {missing}")
        print("  Please run with Train option first to generate them.")
        return None, None

    deg_model = load_model(PATHS["deg_model"])
    deg_pre   = joblib.load(PATHS["deg_pre"])

    print(f"  Loaded: {PATHS['deg_model']}")
    print(f"  Loaded: {PATHS['deg_pre']}")

    return deg_model, deg_pre

#================================================================
#  WHAT-IF ANALYSIS
#================================================================

def run_what_if(df, deg_model, deg_pre):
    whatif_lstm.run_all_what_ifs(
        df, deg_model, deg_pre,
        sequence_length = CONFIG["sequence_length"]
    )

#================================================================
#  MAIN
#================================================================

def main():
    print("\n" + "=" * 65)
    print("  EV BATTERY HEALTH PREDICTION — LSTM PIPELINE")
    print("=" * 65)

    # Step 1 — Choose dataset
    csv_path, dataset_label = select_dataset()

    # Step 2 — Choose run mode
    retrain = select_retrain()

    print(f"\n  Dataset : {dataset_label}")
    print(f"  Mode    : {'Train from scratch' if retrain else 'Load saved models'}")
    print(f"  {'─' * 45}")

    # Step 3 — Load and prepare data (always runs)
    print("\n  Loading and preparing data...")
    df = load_data(csv_path)

    # Step 4 — Train or load
    if retrain:
        deg_model, deg_pre = train(df)
    else:
        deg_model, deg_pre = load_saved_models()
        if deg_model is None:
            return

    # Step 5 — What-if scenario analysis
    run_what_if(df, deg_model, deg_pre)

    print("\n  Pipeline complete.\n")


if __name__ == "__main__":
    main()