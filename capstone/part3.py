"""part 3: segmentation (k-means, hierarchical) and pca, all from scratch"""
import numpy as np
from common import load_data, write_results, write_csv

X, y, v = load_data()
n = len(y)
X5 = X[:, 0:5]                                  # X1 to X5, original units
mu = X5.mean(axis=0)
sd = X5.std(axis=0, ddof=1)
Z = (X5 - mu) / sd                              # standardised over all 800 rows
res = []


# ---------- 3.1 k-means from scratch ----------
def kmeans(Z, K):
    step = len(Z) // K
    cent = Z[[i * step for i in range(K)]].copy()      # pinned starting rows
    labels = np.full(len(Z), -1)
    passes = 0
    while True:
        passes += 1
        d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)                     # ties go to the lower centroid
        if np.array_equal(new, labels):
            break                                      # a pass where nothing changed
        labels = new
        for k in range(K):
            members = Z[labels == k]
            if len(members) > 0:                       # empty cluster keeps its centroid
                cent[k] = members.mean(axis=0)
    wcss = sum(((Z[labels == k] - cent[k]) ** 2).sum() for k in range(K))
    return labels, cent, wcss, passes


wcss = {}
labels_by_k = {}
for K in (2, 3, 4, 5, 6):
    lab, cent, w, passes = kmeans(Z, K)
    wcss[K] = w
    labels_by_k[K] = lab
    print("K", K, "wcss", round(w, 4), "passes", passes)    # K=4: 21, K=5: 15, K=6: 27
    res.append(("wcss_k" + str(K), w))


# ---------- 3.2 elbow ----------
elbow_k = None
for K in (3, 4, 5):
    if wcss[K] - wcss[K + 1] < 0.5 * (wcss[K - 1] - wcss[K]):
        elbow_k = K
        break
print("elbow k", elbow_k)
res.append(("elbow_k", elbow_k))


# ---------- 3.3 profile the K = 3 segments ----------
lab3 = labels_by_k[3]
order = np.argsort([X5[lab3 == k, 2].mean() for k in range(3)])     # ascending mean X3
seg_of_cluster = {int(order[i]): i for i in range(3)}               # cluster -> 0,1,2 (A,B,C)
seg = np.array([seg_of_cluster[int(c)] for c in lab3])
names = ["A", "B", "C"]
rows = []
for i in range(3):
    m = seg == i
    rows.append([names[i], int(m.sum()), X5[m, 0].mean(), X5[m, 1].mean(), X5[m, 2].mean(),
                 X5[m, 3].mean(), X5[m, 4].mean(), y[m].mean(), v[m].mean()])
    res.append(("size_segment_" + names[i], int(m.sum())))
write_csv("outputs/part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
          rows)
print("sizes add up to", sum(r[1] for r in rows))


# ---------- 3.4 hierarchical clustering on the supplied table ----------
D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def linkage_dist(a, b, rule):
    # a and b are lists of member indices (0 to 5)
    ds = [D[i, j] for i in a for j in b]
    if rule == "single":
        return min(ds)
    if rule == "complete":
        return max(ds)
    return sum(ds) / len(ds)            # average


def name_of(c):
    return "+".join("E" + str(i + 1) for i in sorted(c))


def agglomerate(rule):
    clusters = [[i] for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage_dist(clusters[i], clusters[j], rule)
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        a, b = clusters[i], clusters[j]
        if min(b) < min(a):                  # cluster_a has the smaller smallest member
            a, b = b, a
        merges.append((name_of(a), name_of(b), d))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [sorted(a + b)]
    return merges


hier_rows = []
cut = {}
for rule in ("single", "complete", "average"):
    merges = agglomerate(rule)
    heights = [m[2] for m in merges]
    for k, (a, b, h) in enumerate(merges):
        hier_rows.append([rule, k + 1, a, b, h])
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    g = int(np.argmax(gaps)) + 1              # largest gap lies after merge g
    cut[rule] = 6 - g
    print(rule, [round(float(h), 4) for h in heights], "gaps", [round(float(x), 4) for x in gaps], "clusters", cut[rule])
write_csv("outputs/part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hier_rows)
res += [("single_clusters_at_cut", cut["single"]), ("complete_clusters_at_cut", cut["complete"]),
        ("average_clusters_at_cut", cut["average"])]


# ---------- 3.5 pca by the seven steps ----------
mean_vec = Z.mean(axis=0)                     # step 1 (all zeros after standardising)
print("mean vector", np.round(mean_vec, 10))
Xc = Z - mean_vec                             # step 2
S = Xc.T @ Xc / (n - 1)                       # step 3
vals, vecs = np.linalg.eigh(S)                # step 4
idx = np.argsort(vals)[::-1]                  # step 5
vals = vals[idx]
vecs = vecs[:, idx]
for j in range(5):                            # sign: largest |entry| positive
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
W = vecs[:, 0:2]                              # step 6
P = Xc @ W                                    # step 7

total = vals.sum()
print("eigenvalue sum", total)                # 5.0000
ratio = vals / total
cum = np.cumsum(ratio)
res.append(("eigenvalue_sum", total))
for j in range(5):
    res.append(("eigenvalue_" + str(j + 1), vals[j]))
for j in range(5):
    res.append(("explained_ratio_" + str(j + 1), ratio[j]))
for j in range(5):
    res.append(("cumulative_ratio_" + str(j + 1), cum[j]))
res.append(("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1))
for j in range(5):
    res.append(("pc1_loading_X" + str(j + 1), vecs[j, 0]))
for j in range(5):
    res.append(("pc2_loading_X" + str(j + 1), vecs[j, 1]))
for r in range(3):
    res.append(("row" + str(r + 1) + "_pc1", P[r, 0]))
    res.append(("row" + str(r + 1) + "_pc2", P[r, 1]))
for i in range(3):
    res.append(("mean_pc1_segment_" + names[i], P[seg == i, 0].mean()))

write_results("outputs/part3_results.txt", res)
