"""part 4: model-choice and architecture memo numbers
reuses the part 2 models through common.py (same split, same standardisation, same logistic fit, same knn)"""
import math
import numpy as np
from common import (load_data, stratified_split, standardise_with_train, sigmoid, fit_logistic,
                    confusion, metrics, knn_share, knn_predict, knn_cv, write_results, write_csv)

X, y, v = load_data()
res = []

# same split as task 2.1
tr, te = stratified_split(y)
X_train, y_train = X[tr], y[tr]
X_test, y_test = X[te], y[te]
Xs_train, Xs_test = standardise_with_train(X_train, X_test)


# ---------- 4.1 which split is purest (training rows, original units) ----------
def entropy(pos, n):
    if n == 0:
        return 0.0
    out = 0.0
    for c in (pos, n - pos):
        p = c / n
        if p > 0:
            out -= p * math.log2(p)
    return out


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1 - p ** 2 - (1 - p) ** 2


n_tr = len(y_train)
pos_tr = int(y_train.sum())
root_entropy = entropy(pos_tr, n_tr)
root_gini = gini(pos_tr, n_tr)
print("root entropy", root_entropy)       # 0.7462

# yes side of each candidate split
conds = {"S1": X_train[:, 5] == 1,        # X6 = 1
         "S2": X_train[:, 6] == 1,        # X7 = 1
         "S3": X_train[:, 2] > 0.5,       # X3 > 0.5
         "S4": X_train[:, 0] <= 24}       # X1 <= 24
split_rows = []
ig = {}
gg = {}
for name in ("S1", "S2", "S3", "S4"):
    yes = conds[name]
    n_yes = int(yes.sum())
    p_yes = int(y_train[yes].sum())
    n_no = n_tr - n_yes
    p_no = pos_tr - p_yes
    ig[name] = root_entropy - (n_yes / n_tr * entropy(p_yes, n_yes) + n_no / n_tr * entropy(p_no, n_no))
    gg[name] = root_gini - (n_yes / n_tr * gini(p_yes, n_yes) + n_no / n_tr * gini(p_no, n_no))
    split_rows.append([name, n_yes, p_yes, n_no, p_no, ig[name], gg[name]])
write_csv("outputs/part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no",
                                       "info_gain", "gini_gain"], split_rows)
best_ig = max(ig, key=ig.get)
best_gg = max(gg, key=gg.get)
res += [("root_entropy", root_entropy), ("root_gini", root_gini),
        ("best_split_by_info_gain", best_ig), ("best_split_by_gini_gain", best_gg)]


# ---------- 4.2 stump, hard vote, soft vote ----------
yes_tr = conds[best_ig]
stump_rate_yes = y_train[yes_tr].mean()
stump_rate_no = y_train[~yes_tr].mean()
res += [("stump_rate_yes", stump_rate_yes), ("stump_rate_no", stump_rate_no)]

# stump on the test rows: same condition applied to original-unit test features
test_conds = {"S1": X_test[:, 5] == 1, "S2": X_test[:, 6] == 1,
              "S3": X_test[:, 2] > 0.5, "S4": X_test[:, 0] <= 24}
stump_prob = np.where(test_conds[best_ig], stump_rate_yes, stump_rate_no)

# logistic (same model as task 2.2)
w, b, _ = fit_logistic(Xs_train, y_train, 0.1, 1000)
lr_prob = sigmoid(Xs_test @ w + b)

# knn with the cv-chosen K
cv, best_k = knn_cv(Xs_train, y_train)
knn_prob = knn_share(Xs_train, y_train, Xs_test, best_k)
print("knn k", best_k)

pred = {}
pred["logistic"] = (lr_prob >= 0.3).astype(int)
pred["knn"] = knn_predict(knn_prob, best_k)
pred["stump"] = (stump_prob >= 0.3).astype(int)
votes = pred["logistic"] + pred["knn"] + pred["stump"]
pred["hard_vote"] = (votes >= 2).astype(int)
soft = (lr_prob + knn_prob + stump_prob) / 3
pred["soft_vote"] = (soft >= 0.3).astype(int)

vote_rows = []
for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
    tp, fp, tn, fn = confusion(y_test, pred[name])
    m = metrics(tp, fp, tn, fn)
    vote_rows.append([name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]])
write_csv("outputs/part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vote_rows)


# ---------- 4.3 two rounds of adaboost by hand ----------
yb = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
wt = np.full(8, 1 / 8)
alphas = []
ada_rows = []
errs = []
for rnd, h in ((1, h1), (2, h2)):
    eps = wt[h != yb].sum()
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = wt * np.exp(-alpha * yb * h)
    after = unnorm / unnorm.sum()
    for i in range(8):
        ada_rows.append([rnd, i + 1, wt[i], unnorm[i], after[i]])
    if rnd == 1:
        wrong_weight_after_r1 = after[h != yb].sum()
    errs.append(eps)
    alphas.append(alpha)
    wt = after
write_csv("outputs/part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
          ada_rows)
score = alphas[0] * h1 + alphas[1] * h2
ens_pred = np.sign(score)
ensemble_accuracy = np.mean(ens_pred == yb)
print("round 1 error", errs[0], "score", np.round(score, 4), "wrong weight after r1", wrong_weight_after_r1)
res += [("round1_error", errs[0]), ("round1_alpha", alphas[0]), ("round2_error", errs[1]),
        ("round2_alpha", alphas[1]), ("ensemble_accuracy", ensemble_accuracy)]


# ---------- 4.4 out-of-bag share ----------
oob_fraction = (1 - 1 / 560) ** 560
res.append(("oob_fraction", oob_fraction))


# ---------- 4.5 sizing the neural networks (nothing is trained) ----------
def out_size(inp, f, pad, stride):
    return (inp - f + 2 * pad) // stride + 1


# model A option 1: fully connected 784 -> 16 -> 16 -> 10
mlp_hidden1 = 784 * 16 + 16
mlp_hidden2 = 16 * 16 + 16
mlp_output = 16 * 10 + 10
mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

# model A option 2: cnn
s = 28
c1_side = out_size(s, 5, 2, 1)               # conv 5x5, 6 filters, pad 2
cnn_conv1 = 5 * 5 * 1 * 6 + 6
p1_side = out_size(c1_side, 2, 0, 2)         # max pool 2x2 stride 2
c2_side = out_size(p1_side, 5, 0, 1)         # conv 5x5, 16 filters, pad 0
cnn_conv2 = 5 * 5 * 6 * 16 + 16
p2_side = out_size(c2_side, 2, 0, 2)
flat = p2_side * p2_side * 16
cnn_dense1 = flat * 32 + 32
cnn_dense2 = 32 * 10 + 10
cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2
print("cnn shapes", c1_side, p1_side, c2_side, p2_side, "flat", flat)
write_csv("outputs/part4_architecture.csv", ["layer", "parameters"],
          [["mlp_hidden1", mlp_hidden1], ["mlp_hidden2", mlp_hidden2], ["mlp_output", mlp_output],
           ["cnn_conv1", cnn_conv1], ["cnn_conv2", cnn_conv2], ["cnn_dense1", cnn_dense1],
           ["cnn_dense2", cnn_dense2]])

# why convolution scales: 128 x 128 x 3 input
fc_layer_params = 128 * 128 * 3 * 1024 + 1024
conv_layer_params = 3 * 3 * 3 * 64 + 64

# model B: rnn, 64-dim input, 128 hidden, 16 classes
d_in, d_h, d_out = 64, 128, 16
rnn_total = d_h * d_in + d_h * d_h + d_out * d_h + d_h + d_out

res += [("mlp_total", mlp_total), ("cnn_total", cnn_total), ("fc_layer_params", fc_layer_params),
        ("conv_layer_params", conv_layer_params), ("rnn_total", rnn_total)]

# extra text keys used by the README table (shapes are height x width x depth)
res += [("cnn_conv1_out_shape", "%dx%dx6" % (c1_side, c1_side)),
        ("cnn_pool1_out_shape", "%dx%dx6" % (p1_side, p1_side)),
        ("cnn_conv2_out_shape", "%dx%dx16" % (c2_side, c2_side)),
        ("cnn_pool2_out_shape", "%dx%dx16" % (p2_side, p2_side)),
        ("cnn_flatten_size", flat),
        ("round1_wrong_weight_after", wrong_weight_after_r1)]

write_results("outputs/part4_results.txt", res)
