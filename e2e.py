import numpy as np
import pandas as pd
import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory
from model import create_e2e_model, custom_loss
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
LEARNING_RATE = 1e-4
EPOCHS = 200

horizon = 90  # number of days in review period (consider x days in the future for optimal inventory position)
train_horizon = 60
test_horizon = horizon - train_horizon
interval = 10  # number of days between consecutive reorder points
forecast_horizon = 5  # number of days in advance to forecast (at dt, forecast demand from dt + 1 to dt + 5)
seq_len = 15
b = 9
h = 1
week = [i % 7 for i in range(horizon)]
dynamic_features = ['inventory', 'demand', 'lead_time', 'reorder_point', 'day', "week"]
n_dynamic_features = len(dynamic_features)
target = ["reorder_qty", "demand", "lead_time"]

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

        initial_inventory = np.random.randint(1, 150)  # sku initial stock level

        reorder_point_arr = get_reorder_point(train_horizon, interval)
        n_reorder_pts = len(np.where(reorder_point_arr)[0])  # number of reorder points in review period
        history_lead_time = get_lead_time(mu_lead_time, std_lead_time, 50)
        hm, hs, history_demand = get_demand(mu_demand, std_demand, 50)
        future_lead_time = abs(get_lead_time(mu_lead_time, std_lead_time, n_reorder_pts))
        fm, fs, future_demand = get_demand(mu_demand, std_demand, train_horizon)
        safety_stock = get_safety_stock(history_lead_time, history_demand, service_level)
        # e2e recover optimal re-order quantity
        t = 0
        inventory_level = np.zeros(train_horizon)
        inventory_level[0] = initial_inventory
        # generate po reaching
        po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
        if po_reaching[-1] < train_horizon:
            po_reaching.append(po_reaching[-1] + int(np.ceil(np.diff(po_reaching).mean())))
            optimal_rq = np.zeros(train_horizon)
            for i in range(1, train_horizon):
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
            lead_time = np.zeros(train_horizon)
            lead_time[np.where(reorder_point_arr)[0]] = np.int32(np.ceil(future_lead_time))
            # shift reorder point backward by 1 day
            tmp = np.zeros(train_horizon)
            rp = np.where(reorder_point_arr)[0] - 1
            tmp[rp] = 1
            reorder_point_arr = tmp
            data = pd.DataFrame.from_dict({'inventory': inventory_level,
                                           'demand': future_demand,
                                           'lead_time': lead_time,
                                           'reorder_point': reorder_point_arr,
                                           'reorder_qty': optimal_rq,
                                           'week': week[:train_horizon]})
            data["sku"] = sku
            data["store"] = store
            data["day"] = list(range(1, train_horizon + 1))

            cate_feature = data[['sku', 'store']].values
            arr = data[dynamic_features].values
            output_arr = np.squeeze(data[target].values)
            n = train_horizon - seq_len - forecast_horizon + 1 - 5
            train_input_dynamic = np.empty((n, seq_len, n_dynamic_features))
            for i in range(n):
                train_input_dynamic[i, :, :] = arr[i:i + seq_len]

            train_output = np.zeros((n, forecast_horizon, len(target)))
            for i in range(len(train_output)):
                train_output[i, :] = output_arr[i + seq_len: i + seq_len + forecast_horizon, :]
            train_input_dynamic_.append(train_input_dynamic)
            train_output_.append(train_output)
            cate_feature_.append(cate_feature[:n, :])

train_output_final = np.concatenate(train_output_)
train_input_dynamic_final = np.concatenate(train_input_dynamic_)
cate_feature_final = np.concatenate(cate_feature_)

vlt_input = train_input_dynamic_final[:, :, 2]
rp_in = train_input_dynamic_final[:, :, 3]
initial_stock_in = train_input_dynamic_final[:, :, 0]

rp_out = train_output_final[:, :, 0][:, 0]  # replenishment decision (use history seq_len points to predict 1)
df_out = train_output_final[:, :, 1]  # demand forecast (use history seq_len points to predict future forecast_horizon)
vlt_out = train_output_final[:, :, 2][:, 0]  # vendor lead time forecast (use history seq_len points to predict 1)
print(f"Running time for data generation is {time.perf_counter() - start}")
model = create_e2e_model(
    seq_len=seq_len,
    n_dyn_fea=len(dynamic_features) - 3,
    n_outputs=forecast_horizon,
    n_dilated_layers=5,
    kernel_size=2,
    n_filters=3,
    dropout_rate=DROPOUT_RATE,
    max_cat_id=[MAX_SKU_ID, MAX_STORE_ID],
)
adam = optimizers.Adam(learning_rate=LEARNING_RATE)
sgd = optimizers.SGD()
model.compile(loss="mse", optimizer=adam)
# Define checkpoint and fit model
checkpoint = ModelCheckpoint(model_file_name, monitor="loss", save_best_only=True, mode="min", verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1,
    write_graph=True)
callbacks_list = [checkpoint, tensorboard_callback]
history = model.fit(
    [train_input_dynamic_final[:, :, (1, 4, 5)], cate_feature_final, vlt_input, rp_in, initial_stock_in],
    [df_out, rp_out, vlt_out],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1,
)

# get a brand new sku
initial_inventory = 10  # sku initial stock level

reorder_point_arr = get_reorder_point(horizon, interval)
n_reorder_pts = len(np.where(reorder_point_arr)[0])  # number of reorder points in review period
history_lead_time = get_lead_time(mu_lead_time, std_lead_time, 50)
hm, hs, history_demand = get_demand(mu_demand, std_demand, 50)
future_lead_time = abs(get_lead_time(mu_lead_time, std_lead_time, n_reorder_pts))
fm, fs, future_demand = get_demand(mu_demand, std_demand, horizon)
safety_stock = get_safety_stock(history_lead_time, history_demand, service_level)
# e2e recover optimal re-order quantity
t = 0
inventory_level = np.zeros(horizon)
inventory_level[0] = initial_inventory
# generate po reaching
po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))

po_reaching.append(po_reaching[-1] + int(np.ceil(np.diff(po_reaching).mean())))
optimal_rq = np.zeros(horizon)
for i in range(1, train_horizon):
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
# shift reorder point backward by 1 day
tmp = np.zeros(horizon)
rp = np.where(reorder_point_arr)[0] - 1
tmp[rp] = 1
reorder_point_arr = tmp
data = pd.DataFrame.from_dict({'inventory': inventory_level,
                               'demand': future_demand,
                               'lead_time': lead_time,
                               'reorder_point': reorder_point_arr,
                               'reorder_qty': optimal_rq,
                               'week': week[:horizon]})
data["sku"] = sku
data["store"] = store
data["day"] = list(range(1, horizon + 1))

cate_feature = data[['sku', 'store']].values
arr = data[dynamic_features].values
output_arr = np.squeeze(data[target].values)
n = train_horizon - seq_len - forecast_horizon + 1 - 5
train_input_dynamic = np.empty((n, seq_len, n_dynamic_features))
for i in range(n):
    train_input_dynamic[i, :, :] = arr[i:i + seq_len]

train_output = np.zeros((n, forecast_horizon, len(target)))
for i in range(len(train_output)):
    train_output[i, :] = output_arr[i + seq_len: i + seq_len + forecast_horizon, :]
train_input_dynamic_.append(train_input_dynamic)
train_output_.append(train_output)
cate_feature_.append(cate_feature[:n, :])

train_input_dynamic_final = train_input_dynamic
cate_feature_final = cate_feature[:n, :]
vlt_input = train_input_dynamic_final[:, :, 2]
rp_in = train_input_dynamic_final[:, :, 3]
initial_stock_in = train_input_dynamic_final[:, :, 0]

pred_len = test_horizon

df_pred, optimal_qty_pred, vlt_pred = model.predict([train_input_dynamic_final[-pred_len:, :, (1, 4, 5)],
                                                     cate_feature_final[-pred_len:], vlt_input[-pred_len:],
                                                     rp_in[-pred_len:], initial_stock_in[-pred_len:]])

ilt = inventory_level[:train_horizon].copy()
it = np.zeros(horizon)
it[:train_horizon] = ilt
optimal_rq = np.concatenate((optimal_rq, np.zeros(test_horizon)))
inventory_level = it
future_lead_time = abs(get_lead_time(future_lead_time.mean(), future_lead_time.std(), n_reorder_pts))
fm, fs, future_demand = get_demand(fm, fs, horizon)
rp_qty = np.where(optimal_qty_pred > 5)[0]
po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
po_reaching = po_reaching[int(train_horizon / interval):]
cnt = 0
t = 0
for i in range(train_horizon, horizon):
    inventory_level[i] = inventory_level[i - 1] + inventory_level[i] - future_demand[i]
    if reorder_point_arr[i] and t <= len(po_reaching):
        if cnt == len(rp_qty):
            break
        vm = po_reaching[t]
        optimal_qty = optimal_qty_pred[rp_qty[cnt]]
        inventory_level[vm] += optimal_qty
        optimal_rq[i] = optimal_qty
        print(vm, optimal_qty)
        t += 1
        cnt += 1

fig, ax = plt.subplots()
plt.plot(inventory_level)
pred_inv = inventory_level.copy()
plt.plot(list(range(train_horizon, horizon)), inventory_level[train_horizon:])
# add axix label
plt.xlabel("Days")
plt.ylabel("Inventory Level")

plt.legend(["Training Inventory Level", "Testing Inventory Level"])
plt.title("E2E policy DCNN backbone Result")

text_str = (f"Holding Cost: {h} \n"
            f"Backorder Cost: {b} \n"
            f"Demand Mean: {future_demand.mean(): .2f} , Std: {future_demand.std():.2f} \n"
            f"VLT Mean: {future_lead_time.mean(): .2f}, Std: {future_lead_time.std(): .2f}")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.6, 0.1, text_str, horizontalalignment='left', fontsize=8,
        verticalalignment='center', bbox=props, transform=ax.transAxes)
plt.savefig("./E2E result.png")
plt.show()

plt.plot(future_demand)
plt.plot(list(range(train_horizon, horizon)), df_pred[:, 0])
plt.title("Demand Forecast Result with DCNN backbone")
plt.legend(['Actual Demand', 'Forecasted Demand'])
plt.show()
