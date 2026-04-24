#Imports
import pandas as pd

#What-If Analysis

STYLE_ORDER = ["Conservative", "Moderate", "Aggressive"]

def prepare_counterfactual_samples(base_row):
    samples = []

    for style in STYLE_ORDER:
        row = base_row.copy()
        row["Driving_Style"] = style
        samples.append(row)

    return pd.DataFrame(samples)

def run_what_if(raw_input_row, degradation_model, preprocessing_fn):
    scenario_df = prepare_counterfactual_samples(raw_input_row).reset_index(drop=True)

    X_processed = preprocessing_fn(scenario_df)
    predictions = degradation_model.predict(X_processed, verbose=0).flatten()

    results = scenario_df[["Driving_Style"]].copy()
    results["Predicted_SoH_Percent"] = predictions

    return results.reset_index(drop=True)

def print_what_if(results_df):
    print("\n------ What-If Driver Behaviour Impact ------")

    results_df = results_df.reset_index(drop=True)

    for _, row in results_df.iterrows():
        print(f"{row['Driving_Style']:<13} | Predicted SoH: {row['Predicted_SoH_Percent']:.2f}")

    best_pos = results_df["Predicted_SoH_Percent"].values.argmax()
    worst_pos = results_df["Predicted_SoH_Percent"].values.argmin()

    best_row = results_df.iloc[best_pos]
    worst_row = results_df.iloc[worst_pos]

    print("\nBest behaviour:", best_row["Driving_Style"], f"({best_row['Predicted_SoH_Percent']:.2f})")
    print("Worst behaviour:", worst_row["Driving_Style"], f"({worst_row['Predicted_SoH_Percent']:.2f})")