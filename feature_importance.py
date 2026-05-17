#Imports
import numpy as np
from sklearn.metrics import mean_absolute_error

#Feature Importance via Permutation Method

def _clean_feature_name(name):
    """Strips sklearn ColumnTransformer prefixes (num__, cat__) for readable output."""
    if "__" in name:
        return name.split("__", 1)[1]
    return name

def compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10, random_state=42):
    """
    Computes permutation importance for an LSTM regression model predicting SoH.

    For each feature, its values are shuffled across samples (breaking the
    relationship with the target). The rise in MAE tells us how much that
    feature was contributing — a bigger rise means the feature matters more.

    Args:
        model:         Trained Keras LSTM model.
        X_test:        Test sequences, shape (n_samples, sequence_length, n_features).
        y_test:        True SoH values, shape (n_samples,).
        feature_names: List of feature names matching the last axis of X_test.
        n_repeats:     Number of shuffle repeats per feature (averaged for stability).
        random_state:  Seed for reproducibility.

    Returns:
        importances_mean: np.ndarray of mean MAE increase per feature.
        importances_std:  np.ndarray of std deviation of MAE increase per feature.
    """
    np.random.seed(random_state)

    baseline_preds = model.predict(X_test, verbose=0).flatten()
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    importances_mean = []
    importances_std = []

    n_features = X_test.shape[2]

    for feat_idx in range(n_features):
        feat_maes = []

        for _ in range(n_repeats):
            X_permuted = X_test.copy()

            # Shuffle this feature across all samples (all timesteps in the window)
            perm = np.random.permutation(X_permuted.shape[0])
            X_permuted[:, :, feat_idx] = X_permuted[perm, :, feat_idx]

            preds = model.predict(X_permuted, verbose=0).flatten()
            feat_maes.append(mean_absolute_error(y_test, preds))

        importances_mean.append(np.mean(feat_maes) - baseline_mae)
        importances_std.append(np.std(feat_maes))

    return np.array(importances_mean), np.array(importances_std)


def print_feature_importance(importances_mean, importances_std, feature_names, top_n=15):
    """
    Prints a ranked table of feature importances with a visual bar.

    A positive value means shuffling that feature hurt the model's MAE —
    i.e. the model was relying on it. Near-zero or negative values mean
    the model barely uses that feature for SoH prediction.
    """
    clean_names = [_clean_feature_name(n) for n in feature_names]

    print("\n------ Feature Importance: What Most Affects SoH? (Permutation Method) ------")
    print("Shuffling a feature breaks its signal. The bigger the MAE rise, the more the model relied on it.\n")

    sorted_idx = np.argsort(importances_mean)[::-1]
    top_idx = sorted_idx[:top_n]

    max_imp = max(importances_mean[top_idx[0]], 1e-6)

    for rank, idx in enumerate(top_idx, 1):
        name = clean_names[idx]
        imp = importances_mean[idx]
        std = importances_std[idx]
        bar_len = max(0, int((imp / max_imp) * 30))
        bar = "█" * bar_len
        direction = "+" if imp >= 0 else ""
        print(f"  {rank:2}. {name:<40} {direction}{imp:.4f} MAE  (±{std:.4f})  {bar}")

    print(f"\n  Baseline MAE shown above is the reference point; values show the increase when each feature is removed.")

    return sorted_idx, importances_mean


def get_top_features(importances_mean, feature_names, top_n=5):
    """Returns a list of (feature_name, importance) for the top N features."""
    clean_names = [_clean_feature_name(n) for n in feature_names]
    sorted_idx = np.argsort(importances_mean)[::-1]
    return [(clean_names[i], importances_mean[i]) for i in sorted_idx[:top_n]]
