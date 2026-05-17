#Imports
import numpy as np
import os
import joblib

import evaluation
import feature_importance
import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score

#Config

DEFAULT_LEARNING_RATE = 0.001

#Architectures

def dense_regression_model(input_dim, learning_rate=DEFAULT_LEARNING_RATE):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss      = "mse",
        metrics   = ["mae"]
    )

    return model


def lstm_regression_model(input_shape, learning_rate=DEFAULT_LEARNING_RATE):
    """
    Two stacked LSTM layers with dropout for regularisation.
    Predicts SoH as a continuous percentage value.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss      = "mse",
        metrics   = ["mae"]
    )

    return model


def dense_classification_model(input_dim, num_classes, learning_rate=DEFAULT_LEARNING_RATE):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    return model


def lstm_classification_model(input_shape, num_classes, learning_rate=DEFAULT_LEARNING_RATE):
    """
    Two stacked LSTM layers with dropout for regularisation.
    Classifies driving style into Conservative / Moderate / Aggressive.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    return model

#Cross Validation

def cv_regression(X_seq, y_seq, epochs, batch_size, learning_rate, n_splits=5):
    """
    Time-aware KFold CV for the SoH regression model.
    No shuffle — preserves temporal ordering across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_mae, fold_rmse, fold_r2 = [], [], []

    print("\n------ Cross-Validation: LSTM Battery Degradation Regression ------")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_seq), start=1):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y_seq[train_idx], y_seq[test_idx]

        model = lstm_regression_model((X_train.shape[1], X_train.shape[2]), learning_rate)
        model.fit(
            X_train, y_train,
            validation_data = (X_test, y_test),
            epochs          = epochs,
            batch_size      = batch_size,
            callbacks       = [EarlyStopping(patience=20, restore_best_weights=True)],
            verbose         = 0,
        )

        preds = model.predict(X_test, verbose=0).flatten()
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        print(f"  Fold {fold}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    evaluation.print_regression_cv_summary(fold_mae, fold_rmse, fold_r2)


def cv_classification(X_seq, y_seq, n_classes, epochs, batch_size, learning_rate, n_splits=5):
    """
    Time-aware KFold CV for the driving style classifier.
    Uses class weights to handle imbalance between driving styles.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_accuracies, fold_macro_f1 = [], []

    print("\n------ Cross-Validation: LSTM Driver Behaviour Classification ------")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_seq), start=1):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y_seq[train_idx], y_seq[test_idx]

        cw      = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        cw_dict = dict(enumerate(cw))

        model = lstm_classification_model(
            (X_train.shape[1], X_train.shape[2]), n_classes, learning_rate
        )
        model.fit(
            X_train, y_train,
            validation_data = (X_test, y_test),
            epochs          = epochs,
            batch_size      = batch_size,
            class_weight    = cw_dict,
            callbacks       = [EarlyStopping(patience=20, restore_best_weights=True)],
            verbose         = 0,
        )

        preds  = model.predict(X_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        acc      = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        fold_accuracies.append(acc)
        fold_macro_f1.append(macro_f1)
        print(f"  Fold {fold}: Accuracy={acc:.4f}  Macro F1={macro_f1:.4f}")

    evaluation.print_classification_cv_summary(fold_accuracies, fold_macro_f1)

#Training

def train_degradation(df, config, paths):
    """
    Full training pipeline for the LSTM SoH regression model.

    Steps:
      1. Preprocess and build sequences
      2. Cross-validate
      3. Train final model on full train split
      4. Evaluate on held-out test split
      5. Compute permutation feature importances
      6. Save model and preprocessor

    Returns:
        model   : trained Keras LSTM model
        deg_pre : fitted ColumnTransformer (needed for what-if)
    """
    print(f"\n{'='*55}")
    print(f"  PRIMARY MODEL: LSTM Battery Degradation")
    print(f"{'='*55}")

    X_seq, y_seq, deg_pre, feature_names = preprocessing.preprocess_regression(
        df, config["sequence_length"]
    )

    cv_regression(
        X_seq, y_seq,
        config["epochs"], config["batch_size"], config["learning_rate"]
    )

    X_train, X_test, y_train, y_test = preprocessing.split_sequences(
        X_seq, y_seq, config["test_size"]
    )
    print(f"\n  Final split — Train: {len(X_train)}  Test: {len(X_test)}")

    model = lstm_regression_model(
        (X_train.shape[1], X_train.shape[2]), config["learning_rate"]
    )

    print("\n  Training final model...")
    model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs          = config["epochs"],
        batch_size      = config["batch_size"],
        callbacks       = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ],
    )

    preds = model.predict(X_test, verbose=0).flatten()
    evaluation.evaluate_regression(y_test, preds)
    evaluation.evaluate_regression_detailed(y_test, preds)

    print("\n  Computing feature importances (this may take a moment)...")
    imp_mean, imp_std = feature_importance.compute_permutation_importance(
        model, X_test, y_test, feature_names, n_repeats=10
    )
    feature_importance.print_feature_importance(imp_mean, imp_std, feature_names, top_n=15)

    os.makedirs(os.path.dirname(paths["deg_model"]), exist_ok=True)
    model.save(paths["deg_model"])
    joblib.dump(deg_pre, paths["deg_pre"])
    print(f"\n  Model saved      : {paths['deg_model']}")
    print(f"  Preprocessor saved: {paths['deg_pre']}")

    return model, deg_pre


def train_behaviour(df, config, paths):
    """
    Full training pipeline for the LSTM driving style classification model.

    Steps:
      1. Preprocess and build sequences
      2. Cross-validate
      3. Train final model with class weights to handle imbalance
      4. Evaluate on held-out test split
      5. Save model
    """
    print(f"\n{'='*55}")
    print(f"  SECONDARY MODEL: LSTM Driver Behaviour")
    print(f"{'='*55}")

    X_seq, y_seq, beh_pre, enc = preprocessing.preprocess_classification(
        df, config["sequence_length"]
    )
    n_classes = len(enc.classes_)

    cv_classification(
        X_seq, y_seq, n_classes,
        config["epochs"], config["batch_size"], config["learning_rate"]
    )

    X_train, X_test, y_train, y_test = preprocessing.split_sequences(
        X_seq, y_seq, config["test_size"]
    )
    print(f"\n  Final split — Train: {len(X_train)}  Test: {len(X_test)}")

    model = lstm_classification_model(
        (X_train.shape[1], X_train.shape[2]), n_classes, config["learning_rate"]
    )

    class_weights     = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    print("\n  Training final model...")
    model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs          = config["epochs"],
        batch_size      = config["batch_size"],
        class_weight    = class_weight_dict,
        callbacks       = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ],
    )

    preds  = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    evaluation.evaluate_classification(y_test, y_pred)
    evaluation.print_confusion(y_test, y_pred)

    os.makedirs(os.path.dirname(paths["beh_model"]), exist_ok=True)
    model.save(paths["beh_model"])
    print(f"\n  Model saved: {paths['beh_model']}")
