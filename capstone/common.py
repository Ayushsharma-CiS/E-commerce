"""shared helpers for part2.py and part4.py (importing this file does not run any Part)
plain numpy only. every graded number is computed by the code in this repo."""
import csv
import os
import numpy as np


# ---------- printing helpers ----------
def fmt(v):
    # ints as ints, floats with 4 decimals, bool as yes/no, text as given
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.4f" % v
    return str(v)


def write_results(path, items):
    # items is a list of (key, value) pairs, one line per key: key = value
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for k, v in items:
            f.write(k + " = " + fmt(v) + "\n")
    print("wrote", path, "with", len(items), "keys")


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(v) for v in r])
    print("wrote", path, "with", len(rows), "rows")


# ---------- data ----------
def load_data(path="data/records.csv"):
    # returns X (800 x 7, columns X1..X7), outcome y (0/1) and value v
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        rows = [r for r in rd]
    a = np.array([[float(x) for x in r[1:]] for r in rows])
    X = a[:, 0:7]
    y = a[:, 7].astype(int)
    v = a[:, 8]
    return X, y, v


def stratified_split(y):
    # task 2.1: per class, first floor(7*count/10) rows (file order) go to train
    train_idx = []
    test_idx = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        cut = (7 * len(idx)) // 10
        train_idx += list(idx[:cut])
        test_idx += list(idx[cut:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def standardise_with_train(X_train, X_test):
    # training mean and training sample sd only
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=1)
    return (X_train - mu) / sd, (X_test - mu) / sd


# ---------- logistic regression ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(Xs, y, eta, n_updates):
    # batch gradient descent from zeros, loss recorded before any update and after each one
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    losses = [bce(y, sigmoid(Xs @ w + b))]
    for _ in range(n_updates):
        p = sigmoid(Xs @ w + b)
        err = p - y
        gw = Xs.T @ err / n
        gb = err.mean()
        w = w - eta * gw
        b = b - eta * gb
        losses.append(bce(y, sigmoid(Xs @ w + b)))
    return w, b, losses


# ---------- metrics ----------
def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    spec = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, specificity=spec, fpr=fpr)


# ---------- k nearest neighbours ----------
def knn_share(Xref, yref, Xq, K):
    # share of positive neighbours among the K nearest reference rows for each query
    # ranking = (distance, position in reference order) -> stable argsort
    shares = np.zeros(len(Xq))
    for i in range(len(Xq)):
        d = np.sqrt(((Xref - Xq[i]) ** 2).sum(axis=1))
        order = np.argsort(d, kind="stable")[:K]
        shares[i] = yref[order].mean()
    return shares


def knn_predict(share, K):
    # positive when at least (K + 1) / 2 of the K neighbours are positive
    return (share * K >= (K + 1) / 2 - 1e-9).astype(int)


def knn_cv(Xs_train, y_train, ks=range(1, 22, 2)):
    # 5-fold cv, fold = position in training set mod 5, features standardised once on all training rows
    n = len(y_train)
    fold = np.arange(n) % 5
    kmax = max(ks)
    # neighbour order for every held-out row, computed once with the max K
    out = []
    shares_by_k = {k: np.zeros(n) for k in ks}
    for f in range(5):
        q = np.where(fold == f)[0]
        r = np.where(fold != f)[0]
        for i in q:
            d = np.sqrt(((Xs_train[r] - Xs_train[i]) ** 2).sum(axis=1))
            order = np.argsort(d, kind="stable")
            for k in ks:
                shares_by_k[k][i] = y_train[r][order[:k]].mean()
    for k in ks:
        f1s = []
        for f in range(5):
            q = np.where(fold == f)[0]
            pred = knn_predict(shares_by_k[k][q], k)
            tp, fp, tn, fn = confusion(y_train[q], pred)
            f1s.append(metrics(tp, fp, tn, fn)["f1"])
        out.append((k, float(np.mean(f1s))))
    best_k = out[0][0]
    best = out[0][1]
    for k, s in out:
        if s > best:          # strict > keeps the smaller K on ties
            best = s
            best_k = k
    return out, best_k
