import numpy as np
import matplotlib.pyplot as plt
from invutils import get_demand, get_lead_time, get_safety_stock, get_reorder_point, get_target_inventory

# constants
initial_inventory = 100  # sku initial stock level
horizon = 60  # number of days in review period (consider x days in the future for optimal inventory position)
interval = 10  # number of days between consecutive reorder points

mu_demand = 10
std_demand = 6
mu_lead_time = 5
std_lead_time = 1

service_level = 0.99

reorder_point_arr = get_reorder_point(horizon, interval)
n_reorder_pts = len(np.where(reorder_point_arr)[0])  # number of reorder points in review period
history_lead_time = get_lead_time(mu_lead_time, std_lead_time, 50)
hm, hs, history_demand = get_demand(mu_demand, std_demand, 50)
future_lead_time = abs(get_lead_time(mu_lead_time, std_lead_time, n_reorder_pts))
fm, fs, future_demand = get_demand(mu_demand, std_demand, horizon)
safety_stock = get_safety_stock(history_lead_time, history_demand, service_level)

# deduct inventory by demand
inventory_level = np.zeros(horizon)
inventory_level[0] = initial_inventory

reorder_ptr = 0  # the pointer to count number of times reorder happens within review period
po = dict()
po_reaching = []
# calculate optimal replenishment quantity
for i in range(1, horizon):
    reorder_qty = 0
    if reorder_point_arr[i]:
        # sS inventory replenishment policy
        # criterion 1: current inventory level - all demand in lead time >= safety stock
        # criterion 2: current inventory level -all demand before next reorder point >= safety stock
        demand_forecast = abs(future_demand[reorder_ptr] + np.random.normal(0, 0))
        delta1 = inventory_level[i] - demand_forecast * mu_lead_time - safety_stock
        if delta1 < 0:
            reorder_qty = - delta1

        delta2 = inventory_level[i] - demand_forecast * interval - safety_stock
        if delta2 < 0:
            reorder_qty = max(reorder_qty, - delta2)
        lead_time = future_lead_time[reorder_ptr]
        po_reaching.append(i + int(np.ceil(lead_time)))
        if i + int(np.ceil(lead_time)) < horizon:
            inventory_level[i + int(np.ceil(lead_time))] += reorder_qty
        reorder_ptr += 1
        po[i] = reorder_qty

    inventory_level[i] = inventory_level[i] + inventory_level[i - 1] - future_demand[i]
fig, ax = plt.subplots()
line1, = ax.plot(inventory_level, label='Inventory Level')

for key, val in po.items():
    line3 = ax.vlines(x=key, color='r', ymin=0, ymax=inventory_level[key], linestyles="dotted", label=f"Reorder Points")

for val in po_reaching:
    if val < horizon:
        line4 = ax.vlines(x=val, colors="g", ymin=0, ymax=inventory_level[val], linestyles="dotted",
                          label=f"Order Arrival Times")

po_lst = list(po.keys())
# for i in range(len(po_reaching)):
#     plt.arrow(po_lst[i], 25, po_reaching[i] - po_lst[i], 0, head_width=0.1, width=0.05)
# plt.text((po_lst[i] + po_reaching[i]) / 2 - 3, 20, "lead time", fontsize=8)
# plt.text((po_lst[i] + po_reaching[i]) / 2 - 3, 13, f"{future_lead_time[i]:.2f} days", fontsize=8)
plt.title("Inventory Replenishment: Ss Policy VS Labelling Policy")

b = 9
h = 1
t = 1
# e2e recover optimal re-order quantity
t = 0
inventory_level = np.zeros(horizon)
inventory_level[0] = initial_inventory
# generate po reaching
po_reaching = list(np.where(reorder_point_arr)[0] + np.int32(np.ceil(future_lead_time)))
po_reaching.append(po_reaching[-1] + np.ceil(np.diff(po_reaching).mean()))
for i in range(1, horizon):
    inventory_level[i] = inventory_level[i - 1] + inventory_level[i] - future_demand[i]
    if reorder_point_arr[i] and t <= len(po_reaching):
        vm = po_reaching[t]
        delta_t = int(np.floor(b * (po_reaching[t + 1] - po_reaching[t]) / (h + b)))  # s*
        cum_demand = np.sum(future_demand[vm:vm + delta_t])
        optimal_qty = max(cum_demand + np.sum(future_demand[i:vm]) - inventory_level[i], 0)
        inventory_level[vm] += optimal_qty
        t += 1
        print(optimal_qty)
line2, = ax.plot(inventory_level, label="Labelling Policy")

ax.legend(handles=[line1, line2, line3, line4])
# ax.legend(handles=[line2])
plt.savefig("inventory.png")
plt.show()

# inventory level, future_demand, future_lead_time, reorder_point, reorder_
