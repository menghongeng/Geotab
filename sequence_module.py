#Imports
import numpy as np

#Sequence Builder

def build_sequences_from_timeseries(df, feature_cols, target_col, sequence_length=5, vehicle_col="Vehicle_ID", time_col="Timestamp"):
    """
    Builds LSTM-ready sequences from time-series data grouped by vehicle.

    For each vehicle, slides a window of `sequence_length` timesteps across
    their history. The target is the value at the step immediately after
    the window (predict next state from past N states).

    Returns:
        X: np.ndarray of shape (n_sequences, sequence_length, n_features)
        y: np.ndarray of shape (n_sequences,)
    """
    X_sequences = []
    y_sequences = []

    for _, group in df.groupby(vehicle_col):
        group    = group.sort_values(time_col)
        features = group[feature_cols].values
        target   = group[target_col].values

        if len(group) <= sequence_length:
            continue

        for i in range(sequence_length, len(group)):
            X_sequences.append(features[i - sequence_length:i])
            y_sequences.append(target[i])

    return np.array(X_sequences), np.array(y_sequences)
