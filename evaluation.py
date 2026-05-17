#Imports
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix

#Regression Evaluation

def evaluate_regression(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    print("\n------ Regression Evaluation ------")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)

def evaluate_regression_detailed(y_true, y_pred):
    errors = np.abs(y_pred - y_true)

    print("\n------ Detailed Error Breakdown ------")
    print(f"Worst prediction error : {errors.max():.2f}%")
    print(f"Best prediction error  : {errors.min():.2f}%")
    print(f"Within 1% accuracy     : {(errors < 1).mean() * 100:.1f}% of predictions")
    print(f"Within 2% accuracy     : {(errors < 2).mean() * 100:.1f}% of predictions")
    print(f"Within 5% accuracy     : {(errors < 5).mean() * 100:.1f}% of predictions")

def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    return {
        "mae":  float(mae),
        "mse":  float(mse),
        "rmse": float(rmse),
        "r2":   float(r2),
    }

def print_regression_metrics(metrics, header="Regression Evaluation"):
    print(f"\n------ {header} ------")
    print("MAE :", metrics["mae"])
    print("MSE :", metrics["mse"])
    print("RMSE:", metrics["rmse"])
    print("R2  :", metrics["r2"])

#Classification Evaluation

def evaluate_classification(y_true, y_pred):
    print("\n------ Classification Report ------")
    print(classification_report(y_true, y_pred))

def print_confusion(y_true, y_pred):
    print("\n------ Confusion Matrix ------")
    print(confusion_matrix(y_true, y_pred))

#Cross Validation Summaries
#Note: avoiding ± character to prevent encoding errors on Windows terminals

def print_regression_cv_summary(fold_mae, fold_rmse, fold_r2):
    print("\n------ Regression CV Summary ------")
    print(f"Mean MAE : {np.mean(fold_mae):.4f} +/- {np.std(fold_mae):.4f}")
    print(f"Mean RMSE: {np.mean(fold_rmse):.4f} +/- {np.std(fold_rmse):.4f}")
    print(f"Mean R2  : {np.mean(fold_r2):.4f} +/- {np.std(fold_r2):.4f}")

def print_classification_cv_summary(fold_accuracies, fold_macro_f1):
    print("\n------ Classification CV Summary ------")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    print(f"Mean Macro F1: {np.mean(fold_macro_f1):.4f} +/- {np.std(fold_macro_f1):.4f}")
