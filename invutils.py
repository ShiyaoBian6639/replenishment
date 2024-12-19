import numpy as np
import scipy.stats as st


# safety stock determination
def get_safety_stock(lead_time: list, demand: list, service_level: float) -> float:
    z_score = st.norm.ppf(service_level)
    lead_time_mean = np.mean(lead_time)
    lead_time_var = np.var(lead_time)
    demand_mean = np.mean(demand)
    demand_var = np.var(demand)
    safety_stock = z_score * np.sqrt(np.sqrt(lead_time_mean * demand_var + demand_mean ** 2 * lead_time_var))
    return safety_stock


def get_reorder_point(horizon, interval):
    rp = np.zeros(horizon, dtype=np.int8)
    cnt = 2
    while cnt < horizon:
        rp[cnt] = 1
        cnt += interval
    return rp


def get_target_inventory():
    pass


def get_lead_time(mean, std, n):
    return np.round(np.abs(np.random.normal(loc=mean, scale=std, size=n)))


def get_demand(mean, std, n):
    rand_demand = np.round(np.abs(np.random.normal(loc=mean, scale=std, size=n)))
    cycle = mean * np.sin(np.linspace(0, 8 * np.pi, n))
    return mean, std, np.abs(rand_demand + np.abs(cycle))
