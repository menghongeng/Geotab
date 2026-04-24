import pandas as pd

file_path = "ev_battery_degradation_v1.csv"
df = pd.read_csv(file_path)


df.columns = df.columns.str.strip().str.replace(" ", "_")


df = df.dropna(how="all")


df = df.drop_duplicates()
df = df.drop_duplicates(subset=["Vehicle_ID"], keep="first")


required_columns = [
    "Vehicle_ID",
    "Car_Model",
    "Battery_Capacity_kWh",
    "Vehicle_Age_Months",
    "SoH_Percent"
]

df = df.dropna(subset=required_columns)

numeric_columns = [
    "Battery_Capacity_kWh",
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
    "Internal_Resistance_Ohm",
    "SoH_Percent"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Battery_Capacity_kWh", "Vehicle_Age_Months", "SoH_Percent"])


df["Battery_Age_Years"] = (df["Vehicle_Age_Months"] / 12).round(2)


df["Health_Risk_Level"] = df["SoH_Percent"].apply(
    lambda x: "High Risk" if x < 85 else ("Medium Risk" if x < 92 else "Low Risk")
)


output_file = "cleaned_ev_battery_dataset.xlsx"
df.to_excel(output_file, index=False)

print("Cleaning complete.")
print(f"Cleaned file saved as: {output_file}")
print(df.head())