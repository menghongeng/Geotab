#Imports
import numpy as np
import pandas as pd

#Config

STYLE_ORDER              = ["Conservative", "Moderate", "Aggressive"]
FAST_CHARGE_SCENARIOS    = [0.1, 0.3, 0.5, 0.7, 0.9]
DISCHARGE_RATE_SCENARIOS = [0.5, 1.0, 1.5, 2.0, 2.5]

_STYLE_MAP = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}
_REG_DROP  = ["Vehicle_ID", "Timestamp", "Battery_Status", "SoH_Percent"]

#Helpers

def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


def _recompute_engineered(row):
    """
    Recomputes all engineered features for a single modified row.
    Must be called after changing any raw feature so derived features stay consistent.
    """
    row         = row.copy()
    style_score = _STYLE_MAP.get(row["Driving_Style"], 1)

    row["Driving_Style_Score"]   = style_score
    row["Thermal_Stress"]        = row["Avg_Temperature_C"] * row["Fast_Charge_Ratio"]
    row["Cycle_Intensity"]       = row["Total_Charging_Cycles"] / (row["Vehicle_Age_Months"] + 1)
    row["Discharge_Stress"]      = row["Avg_Discharge_Rate_C"] * row["Total_Charging_Cycles"]
    row["Usage_Severity"]        = row["Avg_Discharge_Rate_C"] * row["Fast_Charge_Ratio"] * row["Total_Charging_Cycles"]
    row["Age_Cycle_Interaction"] = row["Vehicle_Age_Months"] * row["Total_Charging_Cycles"]
    row["Aggression_Index"]      = row["Avg_Discharge_Rate_C"] * row["Fast_Charge_Ratio"]
    row["Thermal_Load"]          = row["Avg_Temperature_C"] * row["Total_Charging_Cycles"]
    row["Behaviour_Stress"]      = style_score * row["Avg_Discharge_Rate_C"] * row["Fast_Charge_Ratio"]

    return row


def _sequence_to_lstm_input(sequence_df, pre):
    """
    Drops non-feature columns, preprocesses, and returns (1, seq_len, n_features).
    """
    feature_df  = sequence_df.drop(columns=[c for c in _REG_DROP if c in sequence_df.columns])
    transformed = _to_dense(pre.transform(feature_df))
    return transformed.reshape(1, transformed.shape[0], transformed.shape[1])

#Vehicle Selection

def get_representative_vehicles(df, sequence_length=5):
    """
    Picks one vehicle per driving style (Conservative, Moderate, Aggressive).
    Returns: { driving_style: DataFrame of last sequence_length rows }

    One vehicle per style lets the demo show how the same scenario
    affects different driver profiles differently.
    """
    representatives = {}

    for style in STYLE_ORDER:
        candidates = df[df["Driving_Style"] == style]["Vehicle_ID"].unique()

        for vid in candidates:
            vehicle_df = df[df["Vehicle_ID"] == vid].sort_values("Timestamp")
            if len(vehicle_df) >= sequence_length:
                representatives[style] = vehicle_df.tail(sequence_length).reset_index(drop=True).copy()
                break

    return representatives

#Scenario Runners
#All scenarios vary the feature across ALL timesteps in the sequence.
#Asks: "what would SoH look like if this driver had always behaved this way?"
#This gives meaningful SoH differences vs only changing the last month.

def run_driving_style_what_if(vehicle_df, model, pre):
    results = []

    for style in STYLE_ORDER:
        sequence = vehicle_df.copy()
        for i in range(len(sequence)):
            row = sequence.iloc[i].copy()
            row["Driving_Style"] = style
            row = _recompute_engineered(row)
            sequence.iloc[i] = row

        X        = _sequence_to_lstm_input(sequence, pre)
        pred_soh = float(np.clip(model.predict(X, verbose=0).flatten()[0], 0, 100))
        results.append({"Driving_Style": style, "Predicted_SoH_Percent": pred_soh})

    return pd.DataFrame(results)


def run_fast_charge_what_if(vehicle_df, model, pre):
    results = []

    for ratio in FAST_CHARGE_SCENARIOS:
        sequence = vehicle_df.copy()
        for i in range(len(sequence)):
            row = sequence.iloc[i].copy()
            row["Fast_Charge_Ratio"] = ratio
            row = _recompute_engineered(row)
            sequence.iloc[i] = row

        X        = _sequence_to_lstm_input(sequence, pre)
        pred_soh = float(np.clip(model.predict(X, verbose=0).flatten()[0], 0, 100))
        results.append({"Fast_Charge_Ratio": ratio, "Predicted_SoH_Percent": pred_soh})

    return pd.DataFrame(results)


def run_discharge_rate_what_if(vehicle_df, model, pre):
    results = []

    for rate in DISCHARGE_RATE_SCENARIOS:
        sequence = vehicle_df.copy()
        for i in range(len(sequence)):
            row = sequence.iloc[i].copy()
            row["Avg_Discharge_Rate_C"] = rate
            row = _recompute_engineered(row)
            sequence.iloc[i] = row

        X        = _sequence_to_lstm_input(sequence, pre)
        pred_soh = float(np.clip(model.predict(X, verbose=0).flatten()[0], 0, 100))
        results.append({"Avg_Discharge_Rate_C": rate, "Predicted_SoH_Percent": pred_soh})

    return pd.DataFrame(results)

#Printers

def print_driving_style_what_if(results_df, style_label):
    print(f"\n  [{style_label} driver]")
    for _, row in results_df.iterrows():
        bar = "█" * int(row["Predicted_SoH_Percent"] / 5)
        print(f"    {row['Driving_Style']:<13} | SoH: {row['Predicted_SoH_Percent']:>6.2f}%  {bar}")

    best  = results_df.loc[results_df["Predicted_SoH_Percent"].idxmax()]
    worst = results_df.loc[results_df["Predicted_SoH_Percent"].idxmin()]
    diff  = best["Predicted_SoH_Percent"] - worst["Predicted_SoH_Percent"]
    print(f"    → Switching to best style would recover {diff:.2f}% SoH")


def print_fast_charge_what_if(results_df, style_label):
    print(f"\n  [{style_label} driver]")
    for _, row in results_df.iterrows():
        print(f"    Fast Charge {int(row['Fast_Charge_Ratio'] * 100):>3}%  | SoH: {row['Predicted_SoH_Percent']:>6.2f}%")

    best  = results_df.loc[results_df["Predicted_SoH_Percent"].idxmax()]
    worst = results_df.loc[results_df["Predicted_SoH_Percent"].idxmin()]
    diff  = best["Predicted_SoH_Percent"] - worst["Predicted_SoH_Percent"]
    print(f"    → {diff:.2f}% SoH difference between {int(best['Fast_Charge_Ratio']*100)}% and {int(worst['Fast_Charge_Ratio']*100)}% fast charging")


def print_discharge_rate_what_if(results_df, style_label):
    print(f"\n  [{style_label} driver]")
    for _, row in results_df.iterrows():
        print(f"    Discharge {row['Avg_Discharge_Rate_C']:.1f}C  | SoH: {row['Predicted_SoH_Percent']:>6.2f}%")

    best  = results_df.loc[results_df["Predicted_SoH_Percent"].idxmax()]
    worst = results_df.loc[results_df["Predicted_SoH_Percent"].idxmin()]
    diff  = best["Predicted_SoH_Percent"] - worst["Predicted_SoH_Percent"]
    print(f"    → {diff:.2f}% SoH difference between {best['Avg_Discharge_Rate_C']:.1f}C and {worst['Avg_Discharge_Rate_C']:.1f}C discharge rate")

#Main Entry Point

def run_all_what_ifs(df, model, pre, sequence_length=5):
    """
    Runs all three what-if scenarios for one vehicle per driving style.
    Called from main_lstm.py after training or loading the degradation model.
    """
    representatives = get_representative_vehicles(df, sequence_length)

    print("\n" + "=" * 55)
    print("  WHAT-IF SCENARIO ANALYSIS")
    print("  Varying feature across all 5 months of history")
    print("=" * 55)

    print("\n------ Scenario 1: What If the Driver Always Behaved This Way? ------")
    for style, vehicle_df in representatives.items():
        results = run_driving_style_what_if(vehicle_df, model, pre)
        print_driving_style_what_if(results, style)

    print("\n------ Scenario 2: What If They Always Fast Charged at This Rate? ------")
    for style, vehicle_df in representatives.items():
        results = run_fast_charge_what_if(vehicle_df, model, pre)
        print_fast_charge_what_if(results, style)

    print("\n------ Scenario 3: What If They Always Discharged at This Rate? ------")
    for style, vehicle_df in representatives.items():
        results = run_discharge_rate_what_if(vehicle_df, model, pre)
        print_discharge_rate_what_if(results, style)
