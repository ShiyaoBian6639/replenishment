import os
import sys
import math
import shutil
import random
import datetime
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import scrapbook as sb
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from fclib.common.utils import git_repo_path, module_path
from fclib.dataset.ojdata import download_ojdata, split_train_test, FIRST_WEEK_START
from fclib.feature_engineering.feature_utils import (
    week_of_month,
    df_from_cartesian_product,
    gen_sequence_array,
    static_feature_array,
    normalize_columns,
)
from fclib.models.dilated_cnn import create_dcnn_model
from fclib.evaluation.evaluation_utils import MAPE
from fclib.common.plot import plot_predictions_with_history
from keras.src.saving import serialization_lib
serialization_lib.enable_unsafe_deserialization()
warnings.filterwarnings("ignore")


# Use False if you've already downloaded and split the data
DOWNLOAD_SPLIT_DATA = False

# Data directories
DATA_DIR = os.path.join(git_repo_path(), "ojdata")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Forecasting settings
N_SPLITS = 10
HORIZON = 5
GAP = 2
FIRST_WEEK = 40
LAST_WEEK = 156

# Parameters of the model
SEQ_LEN = 15
DROPOUT_RATE = 0.01
BATCH_SIZE = 64
LEARNING_RATE = 0.015
EPOCHS = 25

# Feature columns
DYNAMIC_FEATURES = ["deal", "feat", "month", "week_of_month", "price", "price_ratio"]
STATIC_FEATURES = ["store", "brand"]

# Maximum store ID and brand ID
MAX_STORE_ID = 137
MAX_BRAND_ID = 11

# Fix random seeds
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)

if DOWNLOAD_SPLIT_DATA:
    download_ojdata(DATA_DIR)
    split_train_test(
        DATA_DIR,
        n_splits=N_SPLITS,
        horizon=HORIZON,
        gap=GAP,
        first_week=FIRST_WEEK,
        last_week=LAST_WEEK,
        write_csv=True,
    )
    print("Finished data downloading and splitting.")


def create_features(pred_round, train_dir, pred_steps, offset):
    """Create a dataframe of the input features.

    Args:
        pred_round (int): Prediction round (1, 2, ...)
        train_dir (str): Path of the training data directory
        pred_steps (int): Number of prediction steps
        offset (int): Length of training data skipped in retraining

    Returns:
        pd.Dataframe: Dataframe including the input features in original scale
        pd.Dataframe: Dataframe including the normalized features
        int: Last week of the training data
    """
    # Load training data
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, "train_" + str(pred_round) + ".csv"))
    train_df["move"] = train_df["logmove"].apply(lambda x: round(math.exp(x)))
    train_df = train_df[["store", "brand", "week", "move"]]

    # Create a dataframe to hold all necessary data
    store_list = train_df["store"].unique()
    brand_list = train_df["brand"].unique()
    train_end_week = train_df["week"].max()
    week_list = range(FIRST_WEEK + offset, train_end_week + GAP + HORIZON)
    d = {"store": store_list, "brand": brand_list, "week": week_list}
    data_grid = df_from_cartesian_product(d)
    data_filled = pd.merge(data_grid, train_df, how="left", on=["store", "brand", "week"])

    # Get future price, deal, and advertisement info
    aux_df = pd.read_csv(os.path.join(TRAIN_DIR, "auxi_" + str(pred_round) + ".csv"))
    data_filled = pd.merge(data_filled, aux_df, how="left", on=["store", "brand", "week"])

    # Create relative price feature
    price_cols = [
        "price1",
        "price2",
        "price3",
        "price4",
        "price5",
        "price6",
        "price7",
        "price8",
        "price9",
        "price10",
        "price11",
    ]
    data_filled["price"] = data_filled.apply(lambda x: x.loc["price" + str(int(x.loc["brand"]))], axis=1)
    data_filled["avg_price"] = data_filled[price_cols].sum(axis=1).apply(lambda x: x / len(price_cols))
    data_filled["price_ratio"] = data_filled["price"] / data_filled["avg_price"]
    data_filled.drop(price_cols, axis=1, inplace=True)

    # Fill missing values
    data_filled = data_filled.groupby(["store", "brand"]).apply(
        lambda x: x.fillna(method="ffill").fillna(method="bfill")
    ).reset_index(drop=True)

    # Create datetime features
    data_filled["week_start"] = data_filled["week"].apply(
        lambda x: FIRST_WEEK_START + datetime.timedelta(days=(x - 1) * 7)
    ).reset_index(drop=True)
    data_filled["month"] = data_filled["week_start"].apply(lambda x: x.month)
    data_filled["week_of_month"] = data_filled["week_start"].apply(lambda x: week_of_month(x))
    data_filled["day"] = data_filled["week_start"].apply(lambda x: x.day)
    data_filled.drop("week_start", axis=1, inplace=True)

    # Normalize the dataframe of features
    cols_normalize = data_filled.columns.difference(["store", "brand", "week"])
    data_scaled, min_max_scaler = normalize_columns(data_filled, cols_normalize)

    return data_filled, data_scaled, train_end_week


def prepare_training_io(data_filled, data_scaled, train_end_week):
    """Prepare input and output for model training.

    Args:
        data_filled (pd.Dataframe): Dataframe including the input features in original scale
        data_scaled (pd.Dataframe): Dataframe including the normalized features
        train_end_week (int): Last week of the training data

    Returns:
        np.array: Input sequences of dynamic features
        np.array: Input sequences of categorical features
        np.array: Output sequences of the target variable
    """
    # Create sequence array for 'move'
    start_timestep = 0
    end_timestep = train_end_week - FIRST_WEEK - HORIZON - GAP + 1
    train_input1 = gen_sequence_array(
        data_scaled, SEQ_LEN, ["move"], "store", "brand", start_timestep, end_timestep - offset,
    )

    # Create sequence array for other dynamic features
    start_timestep = HORIZON + GAP - 1
    end_timestep = train_end_week - FIRST_WEEK
    train_input2 = gen_sequence_array(
        data_scaled, SEQ_LEN, DYNAMIC_FEATURES, "store", "brand", start_timestep, end_timestep - offset,
    )

    seq_in = np.concatenate((train_input1, train_input2), axis=2)

    # Create array of static features
    total_timesteps = train_end_week - FIRST_WEEK - SEQ_LEN - HORIZON - GAP + 3
    cat_fea_in = static_feature_array(data_filled, total_timesteps - offset, STATIC_FEATURES, "store", "brand")

    # Create training output
    start_timestep = SEQ_LEN + GAP - 1
    end_timestep = train_end_week - FIRST_WEEK
    train_output = gen_sequence_array(
        data_filled, HORIZON, ["move"], "store", "brand", start_timestep, end_timestep - offset,
    )
    train_output = np.squeeze(train_output)

    return seq_in, cat_fea_in, train_output


def prepare_testing_io(data_filled, data_scaled, train_end_week):
    """Prepare input and output for model training.

    Args:
        data_filled (pd.Dataframe): Dataframe including the input features in original scale
        data_scaled (pd.Dataframe): Dataframe including the normalized features
        train_end_week (int): Last week of the training data

    Returns:
        np.array: Input sequences of dynamic features
        np.array: Input sequences of categorical features
    """
    # Get inputs for prediction
    start_timestep = train_end_week - FIRST_WEEK - SEQ_LEN + 1
    end_timestep = train_end_week - FIRST_WEEK
    test_input1 = gen_sequence_array(
        data_scaled, SEQ_LEN, ["move"], "store", "brand", start_timestep - offset, end_timestep - offset,
    )

    start_timestep = train_end_week + GAP + HORIZON - FIRST_WEEK - SEQ_LEN
    end_timestep = train_end_week + GAP + HORIZON - FIRST_WEEK - 1
    test_input2 = gen_sequence_array(
        data_scaled, SEQ_LEN, DYNAMIC_FEATURES, "store", "brand", start_timestep - offset, end_timestep - offset,
    )

    seq_in = np.concatenate((test_input1, test_input2), axis=2)

    total_timesteps = 1
    cat_fea_in = static_feature_array(data_filled, total_timesteps, STATIC_FEATURES, "store", "brand")

    return seq_in, cat_fea_in


# Model file name and log directory
model_file_name = "dcnn_model.keras"
log_dir = os.path.join("logs", "scalars")

# Remove log directory if it exists
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)
    print("Removed existing log directory {} \n".format(log_dir))

# Train models and make predictions
pred_all = []
for r in range(1, N_SPLITS + 1):
    print("---- Round " + str(r) + " ----")
    # Use offset to remove older data during retraining
    offset = 0 if r == 1 else 40 + (r - 1) * HORIZON
    # Create features
    data_filled, data_scaled, train_end_week = create_features(r, TRAIN_DIR, HORIZON, offset)

    # Prepare input and output for model training
    seq_in, cat_fea_in, train_output = prepare_training_io(data_filled, data_scaled, train_end_week)

    # Create and train model
    if r == 1:
        model = create_dcnn_model(
            seq_len=SEQ_LEN,
            n_dyn_fea=1 + len(DYNAMIC_FEATURES),
            n_outputs=HORIZON,
            n_dilated_layers=3,
            kernel_size=2,
            n_filters=3,
            dropout_rate=DROPOUT_RATE,
            max_cat_id=[MAX_STORE_ID, MAX_BRAND_ID],
        )
        adam = optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss="mape", optimizer=adam, metrics=["mape", "mae"])
        # Define checkpoint and fit model
        checkpoint = ModelCheckpoint(model_file_name, monitor="loss", save_best_only=True, mode="min", verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1,
        )
        callbacks_list = [checkpoint, tensorboard_callback]
        history = model.fit(
            [seq_in, cat_fea_in],
            train_output,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks_list,
            verbose=1,
        )
    else:
        model = load_model(model_file_name)
        checkpoint = ModelCheckpoint(model_file_name, monitor="loss", save_best_only=True, mode="min", verbose=1)
        callbacks_list = [checkpoint]
        history = model.fit(
            [seq_in, cat_fea_in], train_output, epochs=1, batch_size=BATCH_SIZE, callbacks=callbacks_list, verbose=1,
        )

    # Prepare input for model testing
    seq_in, cat_fea_in = prepare_testing_io(data_filled, data_scaled, train_end_week)

    # Make prediction
    pred = np.round(model.predict([seq_in, cat_fea_in]))

    # Create dataframe for submission
    exp_output = data_filled[data_filled.week >= train_end_week + GAP].reset_index(drop=True)
    exp_output = exp_output[["store", "brand", "week"]]
    pred_df = (
        exp_output.sort_values(["store", "brand", "week"]).loc[:, ["store", "brand", "week"]].reset_index(drop=True)
    )
    pred_df["round"] = r
    pred_df["prediction"] = np.reshape(pred, (pred.size, 1))
    pred_all.append(pred_df)

    # Show the current predictions
    print("\n Prediction results:")
    print(pred_df.head(5))
    print("")

pred_all = pd.concat(pred_all, axis=0)
pred_all.rename(columns={"move": "prediction"}, inplace=True)
pred_all = pred_all[["round", "week", "store", "brand", "prediction"]]

CHECK_TENSORBOARD = True
tensorboard_path = "" # Replace this with the path you find from terminal
if CHECK_TENSORBOARD:
    if not tensorboard_path:
        # Try to find path of the TensorBoard binary
        tensorboard_path = module_path("forecasting_env", "tensorboard")
    if tensorboard_path:
        os.environ["TENSORBOARD_BINARY"] = tensorboard_path
        # Display TensorBoard
    else:
        print("Can't find TensorBoard binary. TensorBoard visualization is skipped.")


# Evaluate prediction accuracy
test_all = []
test_dir = os.path.join(DATA_DIR, "test")
for r in range(1, N_SPLITS + 1):
    test_df = pd.read_csv(os.path.join(test_dir, "test_" + str(r) + ".csv"))
    test_all.append(test_df)
test_all = pd.concat(test_all, axis=0).reset_index(drop=True)
test_all["actual"] = test_all["logmove"].apply(lambda x: round(math.exp(x)))
test_all.drop("logmove", axis=1, inplace=True)
combined = pd.merge(pred_all, test_all, on=["store", "brand", "week"], how="left")
metric_value = MAPE(combined["prediction"], combined["actual"]) * 100
sb.glue("MAPE", metric_value)
print("MAPE of the predictions is {}".format(metric_value))

results = combined[["week", "store", "brand", "prediction"]]
results.rename(columns={"prediction": "move"}, inplace=True)
actual = combined[["week", "store", "brand", "actual"]]
actual.rename(columns={"actual": "move"}, inplace=True)
store_list = combined["store"].unique()
brand_list = combined["brand"].unique()

plot_predictions_with_history(
    results,
    actual,
    store_list,
    brand_list,
    "week",
    "move",
    grain1_name="store",
    grain2_name="brand",
    min_timestep=137,
    num_samples=6,
    predict_at_timestep=135,
    line_at_predict_time=False,
    title="Prediction results for a few sample time series",
    x_label="time step",
    y_label="target value",
    random_seed=6,
)