Domain: E-commerce

## How to run

Python 3.10 or newer. Only NumPy is used (`numpy==2.2.6`). Run these from the repository root of a clean clone, in this order:

```
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`generate_data.py` is the provided script, unchanged. It prints `800 10 170 c6f249ba693f1ebeed318d6d5e18de23` and writes `data/records.csv`. Each `partN.py` writes its files into `outputs/`. `common.py` holds the helpers shared by `part2.py` and `part4.py` (importing it does not run a Part). Running the scripts on a clean clone reproduces the committed `outputs/` files exactly. All money is in INR. The data is synthetic: it is not Flipkart's data.

## Part 1 findings

Source: `outputs/part1_results.txt`.

**R1.1 Planning claim.** The mean order value is 8.9382 (in hundreds of INR) with a 95% interval of 8.3026 to 9.5738, so INR 830 to INR 957. Testing H0: mu = 8.0 gives `z_stat` = 2.8930, which is above the critical value 1.96, so `reject_h0` = yes and H0 is rejected. A Type I error here would be concluding that the true average order value is not INR 800 when in fact it is, so the team would change a plan that was right. The gap is 8.9382 - 8.0 = 0.9382 units, about INR 94 per order (roughly 12% above the plan). I think that is large enough to matter for category plans, because it understates revenue per order in every category. It is only a moderate gap compared with the spread of single orders (`sd_value` = 9.1723), so it is a plan-level problem, not a problem for any single order.

**R1.2 Sample size.** `n_required` = 1293 against the 800 rows available. With 800 rows the 95% margin is 1.96 x 0.3243 = about 0.64 units (INR 64), wider than the INR 50 the team wants. To reach INR 50 the team needs about 493 more orders. The planning team should not treat the current average as precise to INR 50.

**R1.3 Gradient descent.** `gd_slope_gap` = 0.0000 (below 0.001; the code shows about 3e-16). The cost is a smooth bowl with one minimum, and with standardised X4 and eta = 0.1 each update moves downhill until it sits on the same least-squares line as the closed form (the cost is already flat at 40.3698 by update 100 and still 40.3698 at update 500).

**R1.4 Funnel.** `resid_sd_ratio_raw` = 3.7525 and `resid_sd_ratio_log` = 1.0396. The raw model shows a strong funnel: residuals in the upper half of fitted values are about 3.8 times as spread out as in the lower half. The log-log model brings the ratio down to about 1.04, so the log remedy removes the funnel.

**R1.5 Multiple regression.** The predictors with an absolute coefficient above 0.05 (constant excluded) are ln(X4) (1.1576), X2 complaint count (0.0940), X3 discount share (0.1067) and X6 cod flag (0.1518). This is only the mechanical 0.05 rule on coefficients in different units, not a ranking of importance. `mr_adj_r2` = 0.7403 against `loglog_adj_r2` = 0.6561, so the extra predictors do earn their place: adjusted R2 rises by about 0.084 even after the penalty for six more predictors.

## Part 2 findings

Source: `outputs/part2_results.txt`, `outputs/part2_thresholds.csv`, `outputs/part2_knn_cv.csv`. Split: 560 train rows (119 positive) and 240 test rows (51 positive).

**R2.1 Learning rate.** After 200 updates the losses from lowest to highest are eta = 1.0 (0.3703), eta = 0.1 (0.3776), eta = 0.01 (0.5091). A learning rate that is too small has not finished learning after 200 updates, and the larger rate 1.0 still did not overshoot on this smooth loss (it would not be safe to assume that on other data).

**R2.2 Weights.** The three largest absolute weights are X3 discount share (0.9013), X1 tenure (-0.5554) and X2 complaints (0.5505). The largest odds ratio is `odds_ratio_X3` = 2.4629: for a customer one training standard deviation higher in discount order share, the odds of leaving in the next 90 days are multiplied by about 2.46, other things equal.

**R2.3 Accuracy is weak.** `blind_accuracy` = 0.7875 against a tau = 0.5 accuracy of 0.8042. A model that flags nobody is already 78.75% accurate, and the logistic model at 0.5 is only about 1.7 points better, because only 51 of the 240 test records churn. Lowering tau from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 (24 to 38 of 51 churners caught) and lowered precision from 0.5455 to 0.4634. Accuracy fell to 0.7625, below the blind model, which shows how little accuracy tells us here.

**R2.4 Logistic vs KNN.** At tau = 0.3 the logistic model has `f1` = 0.5714 and recall 0.7451. KNN (best K = 5) has `knn_f1` = 0.3696 and `knn_recall` = 0.3333. The logistic model is better on both.

**R2.5 Deployment.** I would deploy the logistic model with the money-based threshold `cost_tau` = 0.15. On the test rows it costs INR 6,050 (`cost_test_at_tau`), against INR 11,800 at tau = 0.5 (`cost_test_at_0.5`) and INR 20,400 for flagging nobody (`cost_test_flag_nobody`). The threshold was chosen on training rows so that the test rows stay untouched and give an honest estimate of the cost; choosing it on the test rows would make the test cost look better than it really is.

## Part 3 findings

Source: `outputs/part3_results.txt`, `outputs/part3_segments.csv`, `outputs/part3_hierarchical.csv`.

**R3.1 Elbow.** The WCSS drops are: WCSS(2) - WCSS(3) = 659.8282, WCSS(3) - WCSS(4) = 108.0739, WCSS(4) - WCSS(5) = 171.0575, WCSS(5) - WCSS(6) = 45.5466. The elbow supports K = 3 (`elbow_k` = 3): going from 3 to 4 clusters gains only 108.07, less than half of the 659.83 gained from 2 to 3. The drops do not shrink steadily (the 4 to 5 drop is larger than the 3 to 4 drop), because the pinned starting rows give one reproducible solution, not necessarily the global optimum.

**R3.2 Segments** (from `part3_segments.csv`):

| Segment | Size | Business name | Profile | Action |
|---|---|---|---|---|
| A | 322 | Loyal full-price regulars | longest tenure (47.5 months), few complaints (1.95), low discount share (0.225), highest loyalty score (62.2), outcome rate 0.0186 | No coupons (they stay anyway); reward loyalty with early access or plus-tier perks |
| B | 255 | High-spend, high-friction shoppers | highest order value (mean_value 17.1248, average past order 28.7) but most complaints (6.07), outcome rate 0.2353 | Fix service problems first (fast complaint resolution), then a targeted retention offer |
| C | 223 | Bargain-focused newer shoppers | shortest tenure (14.4 months), highest discount share (0.712), lowest loyalty (38.5), outcome rate 0.4664 | Send the retention coupon here first and build a repeat habit with early-life offers |

Segment C has the highest `outcome_rate`, 0.4664. A seasonal group is not visible, since the data has no time column.

**R3.3 Hierarchical clustering.** At the cut, single linkage gives 4 clusters: {E1}, {E4}, {E5} and {E2, E3, E6}. Complete linkage gives 2 clusters: {E1, E4, E5} and {E2, E3, E6}. Single linkage joins clusters through their closest pair, so the last merge happens at only 3.4 (E5 and E6) and the largest height gap (1.4) appears early, which leaves the chained group E1/E4/E5 split up. Complete linkage uses the farthest pair, so the final merge needs 7.9 (E1 and E2), which makes a huge last gap of 5.0 and a clear two-group cut.

**R3.4 PCA.** The mean vector is all zeros after standardising. The first component has positive loadings on X1 tenure (0.5317) and X5 loyalty score (0.4418) and negative loadings on X3 discount share (-0.5187), X2 complaints (-0.4100) and X4 average order value (-0.2916): high PC1 means a long-tenure, loyal, low-discount, low-complaint customer. `components_for_80pct` = 3 (two components keep 0.7593, three keep 0.8616). `mean_pc1_segment_A/B/C` = 1.7773, -0.7831, -1.6709: PC1 orders the segments A, B, C, which is also the reverse order of their outcome rates.

## Part 4 findings

Source: `outputs/part4_results.txt`, `outputs/part4_splits.csv`, `outputs/part4_voting.csv`, `outputs/part4_adaboost.csv`, `outputs/part4_architecture.csv`.

**R4.1 Best split.** S4 (X1 tenure <= 24 months) is best by both measures (`info_gain` 0.1252, `gini_gain` 0.0606): it separates 188 rows with a 45.7% outcome rate (86 positives) from 372 rows with an 8.9% rate (33 positives), which is much purer than the other three candidates. A tree grown until every leaf is pure would memorise the noise of the training rows and do worse on new customers. A rule that stops it is a maximum depth (or a minimum number of rows in a leaf).

**R4.2 Which model to ship.** I would ship the logistic model at tau = 0.3 or, using the money rule, at 0.15. On the test rows: logistic `f1` 0.5714 and recall 0.7451; stump 0.5532 and 0.7647; soft vote 0.5507 and 0.7451; hard vote 0.5344 and 0.6863; KNN 0.3696 and 0.3333. Combining models did not help: neither vote beats the logistic model on f1, and the hard vote lowers recall.

**R4.3 AdaBoost.** After round 1 normalisation the two misclassified points (3 and 7) hold a total weight of 0.5000 (0.2500 each). Round 2 trusts h2 more because h2 gets those two heavy points right, so its weighted error is 0.1667 against 0.2500 for h1, which gives `round2_alpha` = 0.8047 against `round1_alpha` = 0.5493.

**R4.4 Architecture.** Model A option 2 (CNN), input 28 x 28 x 1:

| Layer | Output shape (h x w x d) | Parameters |
|---|---|---|
| conv 5x5, 6 filters, pad 2 | 28 x 28 x 6 | 156 |
| ReLU, max-pool 2x2 | 14 x 14 x 6 | 0 |
| conv 5x5, 16 filters, pad 0 | 10 x 10 x 16 | 2,416 |
| ReLU, max-pool 2x2 | 5 x 5 x 16 | 0 |
| flatten | 400 | 0 |
| dense 32, ReLU | 32 | 12,832 |
| dense 10 | 10 | 330 |
| total | | 15,734 (`cnn_total`) |

For comparison the fully connected option 784 -> 16 -> 16 -> 10 has `mlp_total` = 13,002 parameters. A fully connected layer on a 128 x 128 x 3 input with 1,024 neurons needs `fc_layer_params` = 50,332,672 parameters, while a convolutional layer with 64 filters of 3 x 3 needs only `conv_layer_params` = 1,792.

Model B (RNN): W_xh is 128 x 64, W_hh is 128 x 128 and W_hy is 16 x 128, plus b_h (128) and b_y (16), so `rnn_total` = 26,768. It does not grow with longer sequences because the same weights are reused at every time step.

For 10 classes I would use ReLU in the hidden layers and softmax at the output with categorical cross-entropy loss.

In a very deep plain network the gradient is a product of many small layer derivatives, so by the time it reaches the early layers it is almost zero and they stop learning. A skip connection adds the input to a layer's output, which gives the gradient a direct path back (its derivative contains a plain 1 term) so early layers keep receiving a useful signal.

## Model-choice memo

To: Head of Retention. From: Customer Analytics.

I recommend the logistic regression model for production. On the 240 test customers at threshold 0.3 it has f1 0.5714 and recall 0.7451, catching 38 of 51 churners. KNN reaches only f1 0.3696 and recall 0.3333, and the stump, hard vote and soft vote (f1 0.5532, 0.5344, 0.5507) do not beat it, so a simple model wins.

I recommend alert threshold 0.15, chosen on training customers by the cost rule (INR 50 per wasted coupon, INR 400 per lost customer) and not on test data. On the test customers it costs INR 6,050, against INR 11,800 at 0.5 and INR 20,400 if nobody is flagged. These costs are for the logistic model. I did not compute the cost of the other four models, so I make no money claim about them.

The main risk is that the evidence is a single split with only 51 churners in the test set, so a few customers can move the cost by several hundred rupees, and the data is synthetic. Treat INR 6,050 as an estimate and re-check on real recent customers before launch.

Neither Model A (product pictures) nor Model B (review text) is worth building before this model is live, because the structured model already works from seven columns we hold, while those two need labelled pictures and text we do not have.

## Limitations

1. The data is synthetic, generated by a script from fixed rules. Real Flipkart behaviour, seasonality and policies are not represented, so none of these numbers describe the real business.
2. There is one 70/30 split and the test set has only 51 churners. Cost, recall and the choice of threshold are noisy and valid for this split only; the cost estimate of INR 6,050 could move by several hundred rupees with a different split.
3. The K-means results depend on the pinned starting rows and may not be the global optimum, and there is no time column, so a seasonal segment could not be tested.
4. The INR 50 and INR 400 costs are rough assumptions. The z-test of the mean assumes the normal approximation works for a very skewed order-value column (large SD relative to the mean); with n = 800 this is likely fine but was not checked.
5. The comparison of models for churn only checked one K-nearest-neighbour setup (odd K up to 21, 5 folds) and did not check how well the probabilities are calibrated.
