"""part 2: classification and evaluation (logistic regression and knn from scratch)"""
import numpy as np
from common import (load_data, stratified_split, standardise_with_train, sigmoid, bce, fit_logistic,
                    confusion, metrics, knn_share, knn_predict, knn_cv, write_results, write_csv)

X, y, v = load_data()
res = []


# ---------- 2.1 split and standardise ----------
tr, te = stratified_split(y)
X_train, y_train = X[tr], y[tr]
X_test, y_test = X[te], y[te]
Xs_train, Xs_test = standardise_with_train(X_train, X_test)
print("train", len(tr), y_train.sum(), "test", len(te), y_test.sum())   # 560 119 240 51
res += [("train_rows", len(tr)), ("train_positives", int(y_train.sum())),
        ("test_rows", len(te)), ("test_positives", int(y_test.sum()))]


# ---------- 2.2 logistic regression from scratch ----------
w, b, losses = fit_logistic(Xs_train, y_train, 0.1, 1000)
print("loss epoch 0", losses[0])    # 0.6931
test_prob = sigmoid(Xs_test @ w + b)
test_bce = bce(y_test, test_prob)
res += [("loss_epoch_0", losses[0]), ("loss_epoch_1", losses[1]), ("loss_epoch_10", losses[10]),
        ("loss_epoch_100", losses[100]), ("loss_epoch_1000", losses[1000])]
for j in range(7):
    res.append(("weight_X" + str(j + 1), w[j]))
res += [("bias", b), ("test_bce", test_bce)]


# ---------- 2.3 learning-rate effect ----------
for eta in (0.01, 0.1, 1.0):
    _, _, ls = fit_logistic(Xs_train, y_train, eta, 200)
    res.append(("loss_epoch_200_eta_" + str(eta), ls[200]))


# ---------- 2.4 odds ratios ----------
for j in range(7):
    res.append(("odds_ratio_X" + str(j + 1), np.exp(w[j])))


# ---------- 2.5 two thresholds on the test rows ----------
for tau in (0.5, 0.3):
    pred = (test_prob >= tau).astype(int)
    tp, fp, tn, fn = confusion(y_test, pred)
    m = metrics(tp, fp, tn, fn)
    p = "tau_" + str(tau) + "_"
    res += [(p + "tp", tp), (p + "fp", fp), (p + "tn", tn), (p + "fn", fn),
            (p + "accuracy", m["accuracy"]), (p + "precision", m["precision"]),
            (p + "recall", m["recall"]), (p + "f1", m["f1"]), (p + "specificity", m["specificity"])]
    print("tau", tau, tp, fp, tn, fn)
blind_accuracy = np.mean(y_test == 0)     # predict 0 for everybody
res.append(("blind_accuracy", blind_accuracy))


# ---------- 2.6 threshold sweep ----------
rows = []
for k in range(1, 10):
    tau = k / 10
    pred = (test_prob >= tau).astype(int)
    tp, fp, tn, fn = confusion(y_test, pred)
    m = metrics(tp, fp, tn, fn)
    rows.append([tau, tp, fp, tn, fn, m["recall"], m["fpr"]])
write_csv("outputs/part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)


# ---------- 2.7 knn with 5-fold cross-validation ----------
cv, best_k = knn_cv(Xs_train, y_train)
write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], [[k, s] for k, s in cv])
print("best k", best_k)
share = knn_share(Xs_train, y_train, Xs_test, best_k)
pred = knn_predict(share, best_k)
tp, fp, tn, fn = confusion(y_test, pred)
m = metrics(tp, fp, tn, fn)
res += [("knn_best_k", best_k), ("knn_tp", tp), ("knn_fp", fp), ("knn_tn", tn), ("knn_fn", fn),
        ("knn_accuracy", m["accuracy"]), ("knn_precision", m["precision"]),
        ("knn_recall", m["recall"]), ("knn_f1", m["f1"])]


# ---------- 2.8 choose the alert threshold with money ----------
train_prob = sigmoid(Xs_train @ w + b)


def cost_at(yy, prob, tau):
    pred = (prob >= tau).astype(int)
    tp, fp, tn, fn = confusion(yy, pred)
    return 50 * fp + 400 * fn


train_costs = [cost_at(y_train, train_prob, k / 20) for k in range(1, 20)]
best_i = int(np.argmin(train_costs))          # argmin gives the first (smaller tau) on ties
cost_tau = (best_i + 1) / 20
print("train costs", train_costs)
res += [("cost_tau", cost_tau), ("cost_train_at_tau", train_costs[best_i]),
        ("cost_test_at_tau", cost_at(y_test, test_prob, cost_tau)),
        ("cost_test_at_0.5", cost_at(y_test, test_prob, 0.5)),
        ("cost_test_flag_nobody", 400 * int(y_test.sum()))]

write_results("outputs/part2_results.txt", res)
