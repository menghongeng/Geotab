#Imports
import os
import joblib
import numpy as np
import pandas as pd

import model_module
import evaluation
import preprocessing

from keras.callbacks import EarlyStopping

#================================================================
#  CONFIG
#================================================================

CSV_PATH = "data/cleaned_sequential_ev_dataset.csv"

CONFIG = {
    "sequence_length": 5,
    "test_size":       0.2,
    "epochs":          100,
    "batch_size":      256,
}

LEARNING_RATES  = [0.001, 0.005, 0.01, 0.05, 0.1]
MODEL_SAVE_DIR  = "saved_models/lr_sweep"
COMPARISON_FILE = "saved_models/lr_sweep/lstm_lr_comparison.csv"

#================================================================
#  TERMINAL INTERFACE
#================================================================

def select_random_seed():
    """
    Asks the user whether to use a random seed for reproducibility.
    If yes, prompts for a number and sets numpy's random seed.
    Returns the seed used (or None if not set).
    """
    print("\n  Use a random seed for reproducibility?")
    print("  ─────────────────────────────────────────────")
    print("  1. Yes — enter a seed number")
    print("  2. No  — fully random each run")
    print("  ─────────────────────────────────────────────")

    while True:
        choice = input("\n  Enter choice (1 or 2): ").strip()

        if choice == "1":
            while True:
                seed_input = input("  Enter seed number (e.g. 42): ").strip()
                if seed_input.isdigit():
                    seed = int(seed_input)
                    np.random.seed(seed)
                    import tensorflow as tf
                    tf.random.set_seed(seed)
                    print(f"\n  Random seed set to: {seed}")
                    return seed
                else:
                    print("  Please enter a valid whole number.")

        elif choice == "2":
            print("\n  No seed set — results will vary between runs.")
            return None

        else:
            print("  Invalid choice. Please enter 1 or 2.")

#================================================================
#  SWEEP
#================================================================

def train_single_lr(X_train, X_test, y_train, y_test, learning_rate, config):
    """
    Trains one LSTM regression model at the given learning rate.
    Returns the trained model and its evaluation metrics.
    """
    model = model_module.lstm_regression_model(
        (X_train.shape[1], X_train.shape[2]),
        learning_rate = learning_rate
    )

    model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs          = config["epochs"],
        batch_size      = config["batch_size"],
        callbacks       = [EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
        verbose         = 1,
    )

    preds   = model.predict(X_test, verbose=0).flatten()
    metrics = evaluation.regression_metrics(y_test, preds)

    return model, metrics


def save_model(model, deg_pre, learning_rate):
    """Saves the model and preprocessor for this learning rate."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    safe_lr    = str(learning_rate).replace(".", "_")
    model_path = os.path.join(MODEL_SAVE_DIR, f"lstm_deg_lr_{safe_lr}.keras")
    pre_path   = os.path.join(MODEL_SAVE_DIR, f"lstm_deg_lr_{safe_lr}_pre.pkl")

    model.save(model_path)
    joblib.dump(deg_pre, pre_path)

    return model_path, pre_path


def rename_best_model(best_model_path, best_pre_path, best_lr):
    """
    Renames the best model and preprocessor files to include
    'best' and the winning learning rate in the filename.
    """
    safe_lr         = str(best_lr).replace(".", "_")
    new_model_path  = os.path.join(MODEL_SAVE_DIR, f"best_lstm_deg_lr_{safe_lr}.keras")
    new_pre_path    = os.path.join(MODEL_SAVE_DIR, f"best_lstm_deg_lr_{safe_lr}_pre.pkl")

    os.rename(best_model_path, new_model_path)
    os.rename(best_pre_path,   new_pre_path)

    return new_model_path, new_pre_path


def choose_best_model(results_df):
    """Ranks by MAE ascending, RMSE ascending, R2 descending and returns the top row."""
    ranked = results_df.sort_values(
        by        = ["mae", "rmse", "r2"],
        ascending = [True, True, False]
    ).reset_index(drop=True)

    return ranked.iloc[0]


def discard_weaker_models(results_df, best_model_path):
    """Deletes all saved models except the best one."""
    for _, row in results_df.iterrows():
        if row["model_path"] != best_model_path:
            for path_key in ["model_path", "pre_path"]:
                if os.path.exists(row[path_key]):
                    os.remove(row[path_key])
                    print(f"  Discarded: {row[path_key]}")


def run_lr_sweep(df):
    """
    Trains the LSTM degradation model once per learning rate.
    Saves all models during the sweep, then keeps and renames
    the best one and discards the rest.
    """
    X_seq, y_seq, deg_pre, _ = preprocessing.preprocess_regression(
        df, CONFIG["sequence_length"]
    )

    X_train, X_test, y_train, y_test = preprocessing.split_sequences(
        X_seq, y_seq, CONFIG["test_size"]
    )

    results = []

    print("\n------ LSTM DEGRADATION LEARNING RATE SWEEP ------")

    for lr in LEARNING_RATES:
        print(f"\n{'='*50}")
        print(f"  Learning Rate: {lr}")
        print(f"{'='*50}")

        model, metrics = train_single_lr(X_train, X_test, y_train, y_test, lr, CONFIG)

        evaluation.print_regression_metrics(
            metrics,
            header = f"LSTM Degradation Results (LR={lr})"
        )

        model_path, pre_path = save_model(model, deg_pre, lr)
        print(f"  Saved: {model_path}")

        results.append({
            "learning_rate": lr,
            "mae":           metrics["mae"],
            "mse":           metrics["mse"],
            "rmse":          metrics["rmse"],
            "r2":            metrics["r2"],
            "model_path":    model_path,
            "pre_path":      pre_path,
        })

    results_df = pd.DataFrame(results)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    results_df.to_csv(COMPARISON_FILE, index=False)

    # Print comparison table
    print("\n------ LEARNING RATE COMPARISON ------")
    print(results_df[["learning_rate", "mae", "rmse", "r2"]].to_string(index=False))

    # Pick the best
    best = choose_best_model(results_df)

    # Discard weaker models first
    discard_weaker_models(results_df, best["model_path"])

    # Rename best model to include 'best' and LR in filename
    new_model_path, new_pre_path = rename_best_model(
        best["model_path"], best["pre_path"], best["learning_rate"]
    )

    print("\n------ BEST LSTM DEGRADATION MODEL ------")
    print(f"  Learning Rate : {best['learning_rate']}")
    print(f"  MAE           : {best['mae']:.4f}")
    print(f"  RMSE          : {best['rmse']:.4f}")
    print(f"  R2            : {best['r2']:.4f}")
    print(f"  Saved as      : {new_model_path}")
    print(f"  Preprocessor  : {new_pre_path}")
    print(f"\n  Comparison saved: {COMPARISON_FILE}")
    print(f"\n  Plug this into main_lstm.py CONFIG:")
    print(f"  \"learning_rate\": {best['learning_rate']}")

    return results_df, best

#================================================================
#  MAIN
#================================================================

def main():
    print("\n" + "=" * 65)
    print("  LSTM LEARNING RATE SWEEP — BATTERY DEGRADATION")
    print("=" * 65)
    print(f"  Learning rates to test: {LEARNING_RATES}")

    # Step 1 — Random seed selection
    select_random_seed()

    # Step 2 — Load data
    df = preprocessing.load_dataset(CSV_PATH)
    df = preprocessing.add_engineered_features(df)

    # Step 3 — Run sweep
    run_lr_sweep(df)

    print("\n  Sweep complete.\n")


if __name__ == "__main__":
    main()