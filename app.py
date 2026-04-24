from flask import Flask, jsonify
import pandas as pd
import os
from model_service import predict_vehicle, what_if_vehicle

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "ev_battery_degradation_v1.csv")

def load_data():
    df = pd.read_csv(FILE_PATH, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    return df

@app.route("/")
def home():
    return jsonify({"message": "EV software skeleton is running"})

@app.route("/data")
def get_data():
    df = load_data()
    return jsonify(df.head(20).fillna("").to_dict(orient="records"))

@app.route("/battery-health")
def get_battery_health():
    df = load_data()

    required_cols = [
        "Vehicle_ID",
        "Car_Model",
        "SoH_Percent",
        "Battery_Status",
        "Total_Charging_Cycles",
        "Fast_Charge_Ratio"
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return jsonify({
            "error": "Missing expected columns",
            "missing_columns": missing,
            "columns_found": list(df.columns)
        })

    return jsonify(df[required_cols].head(50).fillna("").to_dict(orient="records"))

@app.route("/summary")
def summary():
    df = load_data()
    return jsonify({
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns)
    })

@app.route("/predict/<vehicle_id>")
def predict_route(vehicle_id):
    result = predict_vehicle(vehicle_id, FILE_PATH)
    return jsonify(result)

@app.route("/what-if/<vehicle_id>")
def what_if_route(vehicle_id):
    result = what_if_vehicle(vehicle_id, FILE_PATH)
    return jsonify(result)

@app.route("/insights")
def insights():
    df = load_data()

    result = {
        "total_records": int(len(df)),
        "average_soh": round(float(df["SoH_Percent"].mean()), 2),
        "average_cycles": round(float(df["Total_Charging_Cycles"].mean()), 2),
        "average_fast_charge_ratio": round(float(df["Fast_Charge_Ratio"].mean()), 2),
        "battery_status_counts": df["Battery_Status"].value_counts().to_dict()
    }

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)