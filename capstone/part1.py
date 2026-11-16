"""part 1: estimation, inference and a regression baseline (all 800 rows)"""
import math
import numpy as np
from common import load_data, write_results

X, y, v = load_data()
n = len(v)
x4 = X[:, 3]
res = []   # (key, value) in the order of the tasks


# ---------- 1.1 estimate with an interval ----------
mean_value = v.mean()
sd_value = v.std(ddof=1)
se_value = sd_value / math.sqrt(n)
ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value
print("n", n, "mean", round(mean_value, 4))   # checkpoint 800, 8.9382
res += [("n", n), ("mean_value", mean_value), ("sd_value", sd_value), ("se_value", se_value),
        ("ci95_low", ci95_low), ("ci95_high", ci95_high)]


# ---------- 1.2 test the planning claim (mu = 8.0) ----------
z_stat = (mean_value - 8.0) / se_value
reject_h0 = abs(z_stat) > 1.96
res += [("z_stat", z_stat), ("reject_h0", reject_h0)]


# ---------- 1.3 size a future sample ----------
E = 0.5
n_required = math.ceil((1.96 * sd_value / E) ** 2)
res += [("n_required", n_required)]


# ---------- 1.4 simple regression, closed form ----------
def simple_fit(x, yy):
    # slope = cov / var with divisor n - 1, intercept from the means
    cov = ((x - x.mean()) * (yy - yy.mean())).sum() / (len(x) - 1)
    var = ((x - x.mean()) ** 2).sum() / (len(x) - 1)
    slope = cov / var
    intercept = yy.mean() - slope * x.mean()
    return slope, intercept


reg_slope, reg_intercept = simple_fit(x4, v)
fitted = reg_intercept + reg_slope * x4
tss = ((v - v.mean()) ** 2).sum()
ess = ((fitted - v.mean()) ** 2).sum()
rss = ((v - fitted) ** 2).sum()
tss_equals_ess_plus_rss = abs(tss - ess - rss) < 1e-6 * tss
r2 = 1 - rss / tss
rmse = math.sqrt(rss / n)
mae = np.abs(v - fitted).mean()
print("tss ess rss", tss, ess, rss)
res += [("reg_slope", reg_slope), ("reg_intercept", reg_intercept), ("tss", tss), ("ess", ess),
        ("rss", rss), ("tss_equals_ess_plus_rss", tss_equals_ess_plus_rss), ("r2", r2),
        ("rmse", rmse), ("mae", mae)]


# ---------- 1.5 same line by gradient descent ----------
mean_x4 = x4.mean()
sd_x4 = x4.std(ddof=1)
z = (x4 - mean_x4) / sd_x4
m = 0.0
c = 0.0
eta = 0.1
costs = [np.mean((v - (m * z + c)) ** 2)]       # cost after 0 updates
for step in range(500):
    e = v - (m * z + c)
    dm = -(2 / n) * (z * e).sum()
    dc = -(2 / n) * e.sum()
    m = m - eta * dm
    c = c - eta * dc
    costs.append(np.mean((v - (m * z + c)) ** 2))
gd_slope = m / sd_x4
gd_intercept = c - m * mean_x4 / sd_x4
gd_slope_gap = abs(gd_slope - reg_slope)
print("gd slope gap", gd_slope_gap)
res += [("gd_J0", costs[0]), ("gd_J1", costs[1]), ("gd_J10", costs[10]), ("gd_J100", costs[100]),
        ("gd_J500", costs[500]), ("gd_slope", gd_slope), ("gd_intercept", gd_intercept),
        ("gd_slope_gap", gd_slope_gap)]


# ---------- 1.6 funnel check and log remedy ----------
def half_sds(resid, fit):
    # sort by fitted value (stable), first 400 = lower group, last 400 = upper group
    order = np.argsort(fit, kind="stable")
    lo = resid[order[:400]].std(ddof=1)
    up = resid[order[400:]].std(ddof=1)
    return lo, up, up / lo


resid_raw = v - fitted
lo_raw, up_raw, ratio_raw = half_sds(resid_raw, fitted)

ly = np.log(v)
lx = np.log(x4)
ll_slope, ll_intercept = simple_fit(lx, ly)
ll_fitted = ll_intercept + ll_slope * lx
ll_resid = ly - ll_fitted
ll_rss = (ll_resid ** 2).sum()
ll_tss = ((ly - ly.mean()) ** 2).sum()
loglog_r2 = 1 - ll_rss / ll_tss
loglog_rmse = math.sqrt(ll_rss / n)
lo_log, up_log, ratio_log = half_sds(ll_resid, ll_fitted)
print("resid sd ratio raw", ratio_raw, "log", ratio_log)
res += [("resid_sd_lower_raw", lo_raw), ("resid_sd_upper_raw", up_raw), ("resid_sd_ratio_raw", ratio_raw),
        ("loglog_intercept", ll_intercept), ("loglog_slope", ll_slope), ("loglog_r2", loglog_r2),
        ("loglog_rmse", loglog_rmse), ("resid_sd_lower_log", lo_log), ("resid_sd_upper_log", up_log),
        ("resid_sd_ratio_log", ratio_log)]


# ---------- 1.7 multiple regression by the normal equation ----------
# columns: ones, ln(X4), X1, X2, X3, X5, X6, X7
A = np.column_stack([np.ones(n), np.log(X[:, 3]), X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(A.T @ A, A.T @ ly)
mr_fit = A @ beta
mr_rss = ((ly - mr_fit) ** 2).sum()
mr_r2 = 1 - mr_rss / ll_tss
p = 7
mr_adj_r2 = 1 - (1 - mr_r2) * (n - 1) / (n - p - 1)
loglog_adj_r2 = 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1)
print("beta", np.round(beta, 4))
for j in range(8):
    res.append(("mr_beta_" + str(j), beta[j]))
res += [("mr_r2", mr_r2), ("mr_adj_r2", mr_adj_r2), ("loglog_adj_r2", loglog_adj_r2)]

write_results("outputs/part1_results.txt", res)
