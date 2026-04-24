import os
import joblib
import pandas as pd

from tensorflow.keras.models import load_model
import whatif

MODEL_CACHE = {
    "loaded": False,
    "model": None,
    "preprocessor": None,
    "feature_columns": None,
    "dataframe": None,
}

def add_engineered_features(df):
    df = df.copy()

    style_map = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}

    if "Driving_Style" in df.columns:
        df["Driving_Style_Score"] = df["Driving_Style"].map(style_map)

    if "Avg_Temperature_C" in df.columns and "Fast_Charge_Ratio" in df.columns:
        df["Thermal_Stress"] = df["Avg_Temperature_C"] * df["Fast_Charge_Ratio"]

    if "Total_Charging_Cycles" in df.columns and "Vehicle_Age_Months" in df.columns:
        df["Cycle_Intensity"] = df["Total_Charging_Cycles"] / (df["Vehicle_Age_Months"] + 1)
        df["Age_Cycle_Interaction"] = df["Vehicle_Age_Months"] * df["Total_Charging_Cycles"]

    if "Avg_Discharge_Rate_C" in df.columns and "Total_Charging_Cycles" in df.columns:
        df["Discharge_Stress"] = df["Avg_Discharge_Rate_C"] * df["Total_Charging_Cycles"]

    if all(col in df.columns for col in ["Avg_Discharge_Rate_C", "Fast_Charge_Ratio", "Total_Charging_Cycles"]):
        df["Usage_Severity"] = df["Avg_Discharge_Rate_C"] * df["Fast_Charge_Ratio"] * df["Total_Charging_Cycles"]

    if "Avg_Discharge_Rate_C" in df.columns and "Fast_Charge_Ratio" in df.columns:
        df["Aggression_Index"] = df["Avg_Discharge_Rate_C"] * df["Fast_Charge_Ratio"]

    if "Avg_Temperature_C" in df.columns and "Total_Charging_Cycles" in df.columns:
        df["Thermal_Load"] = df["Avg_Temperature_C"] * df["Total_Charging_Cycles"]

    if all(col in df.columns for col in ["Driving_Style_Score", "Avg_Discharge_Rate_C", "Fast_Charge_Ratio"]):
        df["Behaviour_Stress"] = df["Driving_Style_Score"] * df["Avg_Discharge_Rate_C"] * df["Fast_Charge_Ratio"]

    # Optional compatibility feature from Hiraniya's script
    if "Battery_Age_Years" not in df.columns and "Vehicle_Age_Months" in df.columns:
        df["Battery_Age_Years"] = (df["Vehicle_Age_Months"] / 12).round(2)

    return df

def get_expected_columns_from_preprocessor(preprocessor):
    # Most likely available on fitted sklearn transformers
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    # Fallback if feature_names_in_ is unavailable
    expected = []
    if hasattr(preprocessor, "transformers_"):
        for _, _, cols in preprocessor.transformers_:
            if cols == "drop":
                continue
            if isinstance(cols, list):
                expected.extend(cols)
    return expected

def add_missing_columns(df, expected_columns):
    df = df.copy()

    categorical_like = {
        "Car_Model",
        "Battery_Type",
        "Driving_Style",
        "Battery_Status",
        "Health_Risk_Level",
    }

    for col in expected_columns:
        if col not in df.columns:
            if col in categorical_like:
                df[col] = "Unknown"
            else:
                df[col] = 0

    return df

def load_model_once(csv_path):
    if MODEL_CACHE["loaded"]:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "prediction_model.keras")
    preprocessor_path = os.path.join(base_dir, "Prediction_PreProcessor.joblib")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    df = add_engineered_features(df)

    preprocessor = joblib.load(preprocessor_path)
    model = load_model(model_path)

    feature_columns = get_expected_columns_from_preprocessor(preprocessor)
    df = add_missing_columns(df, feature_columns)

    MODEL_CACHE["loaded"] = True
    MODEL_CACHE["model"] = model
    MODEL_CACHE["preprocessor"] = preprocessor
    MODEL_CACHE["feature_columns"] = feature_columns
    MODEL_CACHE["dataframe"] = df

def predict_vehicle(vehicle_id, csv_path):
    load_model_once(csv_path)

    model = MODEL_CACHE["model"]
    preprocessor = MODEL_CACHE["preprocessor"]
    feature_columns = MODEL_CACHE["feature_columns"]
    df = MODEL_CACHE["dataframe"]

    row = df[df["Vehicle_ID"].astype(str) == str(vehicle_id)]
    if row.empty:
        return {"error": "Vehicle not found"}

    row = row.iloc[[0]].copy()
    row = add_missing_columns(row, feature_columns)

    X_input = row[feature_columns]
    X_processed = preprocessor.transform(X_input)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    predicted_soh = float(model.predict(X_processed, verbose=0).flatten()[0])

    return {
        "Vehicle_ID": row.iloc[0].get("Vehicle_ID", ""),
        "Car_Model": row.iloc[0].get("Car_Model", ""),
        "Current_SoH": float(row.iloc[0].get("SoH_Percent", 0)),
        "Predicted_SoH": round(predicted_soh, 2),
        "Battery_Status": row.iloc[0].get("Battery_Status", "")
    }

def what_if_vehicle(vehicle_id, csv_path):
    load_model_once(csv_path)

    model = MODEL_CACHE["model"]
    preprocessor = MODEL_CACHE["preprocessor"]
    feature_columns = MODEL_CACHE["feature_columns"]
    df = MODEL_CACHE["dataframe"]

    row = df[df["Vehicle_ID"].astype(str) == str(vehicle_id)]
    if row.empty:
        return {"error": "Vehicle not found"}

    base_row = row.iloc[0].copy()

    if "Driving_Style" not in df.columns:
        base_row["Driving_Style"] = "Unknown"

    scenario_df = whatif.prepare_counterfactual_samples(base_row).reset_index(drop=True)
    scenario_df = add_engineered_features(scenario_df)
    scenario_df = add_missing_columns(scenario_df, feature_columns)

    X_input = scenario_df[feature_columns]
    X_processed = preprocessor.transform(X_input)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    predictions = model.predict(X_processed, verbose=0).flatten()

    results = scenario_df[["Driving_Style"]].copy()
    results["Predicted_SoH_Percent"] = [round(float(x), 2) for x in predictions]

    return {
        "Vehicle_ID": base_row.get("Vehicle_ID", ""),
        "Car_Model": base_row.get("Car_Model", ""),
        "scenarios": results.to_dict(orient="records")
    }