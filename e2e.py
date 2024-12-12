import numpy as np
import pandas as pd
import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory
from fclib.models.dilated_cnn import create_dcnn_model
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.src.saving import serialization_lib
import time

serialization_lib.enable_unsafe_deserialization()

start = time.perf_counter()
DROPOUT_RATE = 0.01
MAX_SKU_ID = 100
MAX_STORE_ID = 50
BATCH_SIZE = 1024
LEARNING_RATE = 0.015
EPOCHS = 50
seq_len = 5
b = 9
h = 1
model_file_name = "e2e.keras"
log_dir = os.path.join("logs", "scalars")
train_input_dynamic_ = []
train_output_ = []
cate_feature_ = []

for sku in range(1, MAX_SKU_ID + 1):
    mu_demand = np.random.randint(20)
    std_demand = 3
    mu_lead_time = np.random.randint(5)
    std_lead_time = 1
    service_level = 0.99
    for store in range(1, MAX_STORE_ID + 1):

        initial_inventory = np.random.randint(50, 150)  # sku initial stock level
        horizon = 60  # number of days in review period (consider x days in the future for optimal inventory position)
        interval = 10  # number of days between consecutive reorder points

        reorder_point_arr = get_reorder_point(horizon, interval)
        n_reorder_pts = len(np.where(reorder_point_arr)[0])  # number of reorder points in review period
        history_lead_time = get_lead_time(mu_lead_time, std_lead_time, 50)
        history_demand = abs(get_demand(mu_demand, std_demand, 50))
        future_lead_time = abs(get_lead_time(mu_lead_time, std_lead_time, n_reorder_pts))
        future_demand = abs(get_demand(mu_demand, std_demand, horizon))
        safety_stock = get_safety_stock(history_lead_time, history_demand, service_level)
        # e2e recover optimal re-order quantity
        t = 0
        inventory_level = np.zeros(horizon)
        inventory_level[0] = initial_inventory
        # generate po reaching
        po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
        if po_reaching[-1] < horizon:
            po_reaching.append(po_reaching[-1] + int(np.ceil(np.diff(po_reaching).mean())))
            optimal_rq = np.zeros(horizon)
            for i in range(1, horizon):
                inventory_level[i] = inventory_level[i - 1] + inventory_level[i] - future_demand[i]
                if reorder_point_arr[i] and t <= len(po_reaching):
                    vm = po_reaching[t]
                    delta_t = int(np.floor(b * (po_reaching[t + 1] - po_reaching[t]) / (h + b)))  # s*
                    cum_demand = np.sum(future_demand[vm:vm + delta_t])
                    optimal_qty = max(cum_demand + np.sum(future_demand[i:vm]) - inventory_level[i], 0)
                    inventory_level[vm] += optimal_qty
                    optimal_rq[i] = optimal_qty
                    t += 1

            # inventory level, future_demand, future_lead_time, reorder_point, reorder_
            lead_time = np.zeros(horizon)
            lead_time[np.where(reorder_point_arr)[0]] = np.int32(np.ceil(future_lead_time))
            data = pd.DataFrame.from_dict({'inventory': inventory_level,
                                           'demand': future_demand,
                                           'lead_time': lead_time,
                                           'reorder_point': reorder_point_arr,
                                           'reorder_qty': optimal_rq})
            data["sku"] = sku
            data["store"] = store
            data["day"] = list(range(1, horizon + 1))

            dynamic_features = ['inventory', 'demand', 'lead_time', 'reorder_point', 'day']
            n_dynamic_features = len(dynamic_features)
            target = ["reorder_qty"]
            cate_feature = data[['sku', 'store']].values
            arr = data[dynamic_features].values
            output_arr = np.squeeze(data[target].values)
            n = horizon - seq_len + 1
            train_input_dynamic = np.empty((n, seq_len, n_dynamic_features))
            for i in range(n):
                train_input_dynamic[i, :, :] = arr[i:i + seq_len]

            train_output = np.empty((n, seq_len))
            for i in range(n):
                train_output[i, :] = output_arr[i:i + seq_len]
            train_input_dynamic_.append(train_input_dynamic)
            train_output_.append(train_output)
            cate_feature_.append(cate_feature[:n, :])

train_output_final = np.concatenate(train_output_)
train_input_dynamic_final = np.concatenate(train_input_dynamic_)
cate_feature_final = np.concatenate(cate_feature_)
print(f"Running time for data generation is {time.perf_counter() - start}")
model = create_dcnn_model(
    seq_len=seq_len,
    n_dyn_fea=len(dynamic_features),
    n_outputs=seq_len,
    n_dilated_layers=3,
    kernel_size=2,
    n_filters=3,
    dropout_rate=DROPOUT_RATE,
    max_cat_id=[MAX_SKU_ID, MAX_STORE_ID],
)
adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mse", optimizer=adam, metrics=["mse", "mae"])
# Define checkpoint and fit model
checkpoint = ModelCheckpoint(model_file_name, monitor="loss", save_best_only=True, mode="min", verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1,
)
callbacks_list = [checkpoint, tensorboard_callback]
history = model.fit(
    [train_input_dynamic_final, cate_feature_final],
    train_output_final,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1,
)

for r in range(0):
    model = load_model(model_file_name)
    checkpoint = ModelCheckpoint(model_file_name, monitor="loss", save_best_only=True, mode="min", verbose=1)
    callbacks_list = [checkpoint]
    history = model.fit(
        [train_input_dynamic_final, cate_feature_final],
        train_output_final, epochs=1, batch_size=BATCH_SIZE, callbacks=callbacks_list, verbose=1,
    )

###############################################################
initial_inventory = np.random.randint(50, 150)  # sku initial stock level
horizon = 60  # number of days in review period (consider x days in the future for optimal inventory position)
interval = 10  # number of days between consecutive reorder points

reorder_point_arr = get_reorder_point(horizon, interval)
n_reorder_pts = len(np.where(reorder_point_arr)[0])  # number of reorder points in review period
history_lead_time = get_lead_time(mu_lead_time, std_lead_time, 50)
history_demand = abs(get_demand(mu_demand, std_demand, 50))
future_lead_time = abs(get_lead_time(mu_lead_time, std_lead_time, n_reorder_pts))
future_demand = abs(get_demand(mu_demand, std_demand, horizon))
safety_stock = get_safety_stock(history_lead_time, history_demand, service_level)
# e2e recover optimal re-order quantity
t = 0
inventory_level = np.zeros(horizon)
inventory_level[0] = initial_inventory
# generate po reaching
po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
if po_reaching[-1] < horizon:
    po_reaching.append(po_reaching[-1] + int(np.ceil(np.diff(po_reaching).mean())))
    optimal_rq = np.zeros(horizon)
    for i in range(1, horizon):
        inventory_level[i] = inventory_level[i - 1] + inventory_level[i] - future_demand[i]
        if reorder_point_arr[i] and t <= len(po_reaching):
            vm = po_reaching[t]
            delta_t = int(np.floor(b * (po_reaching[t + 1] - po_reaching[t]) / (h + b)))  # s*
            cum_demand = np.sum(future_demand[vm:vm + delta_t])
            optimal_qty = max(cum_demand + np.sum(future_demand[i:vm]) - inventory_level[i], 0)
            inventory_level[vm] += optimal_qty
            optimal_rq[i] = optimal_qty
            t += 1

    # inventory level, future_demand, future_lead_time, reorder_point, reorder_
    lead_time = np.zeros(horizon)
    lead_time[np.where(reorder_point_arr)[0]] = np.int32(np.ceil(future_lead_time))
    data = pd.DataFrame.from_dict({'inventory': inventory_level,
                                   'demand': future_demand,
                                   'lead_time': lead_time,
                                   'reorder_point': reorder_point_arr,
                                   'reorder_qty': optimal_rq})
    data["sku"] = sku
    data["store"] = store
    data["day"] = list(range(1, horizon + 1))

    dynamic_features = ['inventory', 'demand', 'lead_time', 'reorder_point', 'day']
    n_dynamic_features = len(dynamic_features)
    target = ["reorder_qty"]
    cate_feature = data[['sku', 'store']].values
    arr = data[dynamic_features].values
    output_arr = np.squeeze(data[target].values)
    n = horizon - seq_len + 1
    train_input_dynamic = np.empty((n, seq_len, n_dynamic_features))
    for i in range(n):
        train_input_dynamic[i, :, :] = arr[i:i + seq_len]

    train_output = np.empty((n, seq_len))
    for i in range(n):
        train_output[i, :] = output_arr[i:i + seq_len]

test = train_input_dynamic
# test[:, :, 0] += 20
pred = np.round(model.predict([test, cate_feature_final[0:n]]))

# recover inventory level
rq = pred[:, 0][np.where(reorder_point_arr)[0]]
t = 0
inventory_level = np.zeros(horizon)
inventory_level[0] = initial_inventory
# generate po reaching
po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
if po_reaching[-1] < horizon:
    po_reaching.append(po_reaching[-1] + int(np.ceil(np.diff(po_reaching).mean())))
    optimal_rq = np.zeros(horizon)
    for i in range(1, horizon):
        inventory_level[i] = inventory_level[i - 1] + inventory_level[i] - future_demand[i]
        if reorder_point_arr[i] and t <= len(po_reaching):
            vm = po_reaching[t]

            optimal_qty = rq[t]
            inventory_level[vm] += optimal_qty
            optimal_rq[i] = optimal_qty
            t += 1

plt.plot(inventory_level)
plt.show()