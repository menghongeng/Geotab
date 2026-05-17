#Imports
import numpy as np
import pandas as pd
import sequence_module

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

#Dataset Loader

_BATTERY_TYPE_MAP = {
    "BYD Atto 3":          "LFP",
    "Wuling Air EV":       "LFP",
    "Tesla Model 3":       "NMC",
    "Ford Mustang Mach-E": "NMC",
    "Hyundai Ioniq 5":     "NMC",
}

def load_dataset(path):
    """
    Loads and prepares the sequential EV dataset for the LSTM pipeline.

    Fixes applied on load:
      - Time_Step renamed → Timestamp       (pipeline sorts on this column)
      - Battery_Type inferred from Car_Model (column not present in raw data)
      - SoH_Change dropped                  (equals SoH[t]-SoH[t-1], direct data leakage)
      - Cycle_Increment dropped             (21% negative values, unreliable)
      - Distance_Increment dropped          (41% negative values, unreliable)
      - Sequence_ID dropped                 (row identifier, not a feature)
      - Record_Date dropped                 (string date, not usable as numeric feature)
    """
    df = pd.read_csv(path)

    df = df.rename(columns={"Time_Step": "Timestamp"})
    df["Battery_Type"] = df["Car_Model"].map(_BATTERY_TYPE_MAP).fillna("Unknown")

    drop_cols = ["Sequence_ID", "Record_Date", "SoH_Change", "Cycle_Increment", "Distance_Increment"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"\n  Dataset loaded: {len(df):,} rows | "
          f"{df['Vehicle_ID'].nunique():,} vehicles | "
          f"{df['Timestamp'].nunique()} timesteps each")

    return df

#Feature Engineering

def add_engineered_features(df):
    """
    Adds interaction and stress features derived from raw columns.
    Consistent with the dense model pipeline.
    """
    df = df.copy()

    style_map = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}

    df["Driving_Style_Score"]   = df["Driving_Style"].map(style_map)
    df["Thermal_Stress"]        = df["Avg_Temperature_C"] * df["Fast_Charge_Ratio"]
    df["Cycle_Intensity"]       = df["Total_Charging_Cycles"] / (df["Vehicle_Age_Months"] + 1)
    df["Discharge_Stress"]      = df["Avg_Discharge_Rate_C"] * df["Total_Charging_Cycles"]
    df["Usage_Severity"]        = df["Avg_Discharge_Rate_C"] * df["Fast_Charge_Ratio"] * df["Total_Charging_Cycles"]
    df["Age_Cycle_Interaction"] = df["Vehicle_Age_Months"] * df["Total_Charging_Cycles"]
    df["Aggression_Index"]      = df["Avg_Discharge_Rate_C"] * df["Fast_Charge_Ratio"]
    df["Thermal_Load"]          = df["Avg_Temperature_C"] * df["Total_Charging_Cycles"]
    df["Behaviour_Stress"]      = df["Driving_Style_Score"] * df["Avg_Discharge_Rate_C"] * df["Fast_Charge_Ratio"]

    return df

#Helpers

def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

#Sequence Preprocessors

def preprocess_regression(df, sequence_length):
    """
    Scales features and builds sliding-window sequences for the SoH regression model.

    Returns:
        X_seq         : (n_sequences, sequence_length, n_features)
        y_seq         : (n_sequences,)
        preprocessor  : fitted ColumnTransformer — saved for what-if use
        feature_names : list of transformed feature names
    """
    df = df.sort_values(["Vehicle_ID", "Timestamp"]).reset_index(drop=True)

    target_col = "SoH_Percent"
    cat        = ["Car_Model", "Battery_Type", "Driving_Style"]
    drop_cols  = ["Vehicle_ID", "Timestamp", "Battery_Status", target_col]

    feature_df = df.drop(columns=drop_cols)
    num        = [c for c in feature_df.columns if c not in cat]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

    transformed   = _to_dense(pre.fit_transform(feature_df))
    feature_names = list(pre.get_feature_names_out())

    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    transformed_df["Vehicle_ID"] = df["Vehicle_ID"].values
    transformed_df["Timestamp"]  = df["Timestamp"].values
    transformed_df[target_col]   = df[target_col].values

    X_seq, y_seq = sequence_module.build_sequences_from_timeseries(
        transformed_df,
        feature_cols    = feature_names,
        target_col      = target_col,
        sequence_length = sequence_length,
        vehicle_col     = "Vehicle_ID",
        time_col        = "Timestamp",
    )

    print(f"  Sequences built: {X_seq.shape[0]}  |  shape: {X_seq.shape}")
    return X_seq, y_seq, pre, feature_names


def preprocess_classification(df, sequence_length):
    """
    Scales features and builds sliding-window sequences for the driving style classifier.

    Returns:
        X_seq         : (n_sequences, sequence_length, n_features)
        y_seq         : (n_sequences,)
        preprocessor  : fitted ColumnTransformer
        label_encoder : fitted LabelEncoder for Driving_Style
    """
    df = df.sort_values(["Vehicle_ID", "Timestamp"]).reset_index(drop=True)

    target_col = "Driving_Style"
    cat        = ["Car_Model", "Battery_Type"]
    drop_cols  = ["Vehicle_ID", "Timestamp", "Battery_Status",
                  target_col, "Driving_Style_Score", "Behaviour_Stress"]

    feature_df = df.drop(columns=drop_cols)
    num        = [c for c in feature_df.columns if c not in cat]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

    transformed   = _to_dense(pre.fit_transform(feature_df))
    feature_names = list(pre.get_feature_names_out())

    enc       = LabelEncoder()
    y_encoded = enc.fit_transform(df[target_col])

    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    transformed_df["Vehicle_ID"] = df["Vehicle_ID"].values
    transformed_df["Timestamp"]  = df["Timestamp"].values
    transformed_df[target_col]   = y_encoded

    X_seq, y_seq = sequence_module.build_sequences_from_timeseries(
        transformed_df,
        feature_cols    = feature_names,
        target_col      = target_col,
        sequence_length = sequence_length,
        vehicle_col     = "Vehicle_ID",
        time_col        = "Timestamp",
    )

    print(f"  Sequences built: {X_seq.shape[0]}  |  classes: {list(enc.classes_)}")
    return X_seq, y_seq, pre, enc

#Train / Test Split

def split_sequences(X_seq, y_seq, test_size):
    """
    Chronological train/test split — no shuffling so temporal order is preserved.
    Trains on earlier months, tests on later months.
    """
    split_idx = int(len(X_seq) * (1 - test_size))
    return (
        X_seq[:split_idx], X_seq[split_idx:],
        y_seq[:split_idx], y_seq[split_idx:],
    )
