#Imports
import whatif

#Current Demo

def print_demo_header():
    print("\n------ Current Demo: Snapshot Battery Modelling ------")
    print("\nUsing current non-sequential dataset.")

def print_demo_summary(df):
    print("\nAverage SoH by Driving Style")
    print(df.groupby("Driving_Style")["SoH_Percent"].mean())

def run_what_if(sample, deg_model, preprocess_fn):
    results = whatif.run_what_if(sample, deg_model, preprocess_fn)
    whatif.print_what_if(results)
    return results

def print_demo_footer():
    print("\nCurrent demo complete.")
    print("Future demo will utilise LSTM model once access to time sequence dataset is finalised.")
