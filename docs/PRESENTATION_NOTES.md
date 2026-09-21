# Presentation preparation

Use these notes to understand and explain the work. They are a preparation guide,
not a record that the student has already reviewed the implementation.

## A short presentation sequence

1. **Problem and timing.** Predict the sale price after inspection but before
   negotiation or financing. The public dataset has 1,800 cars and 19 columns.
   The final classroom CSV contains 600 unlabeled listings.
2. **Availability and data quality.** Exclude the financing-related loan approval
   field because it would not be available at prediction time. Also exclude the
   ID and target. Retain all rows. Investigate extreme mileage rather than assuming
   every unusual car is an error.
3. **Preprocessing and split.** Use 1,440 rows for training and 360 for validation,
   with seed 42. Fill numeric gaps with training medians. Include missing-value
   indicators. One-hot encode categories and standardize features. Learn all
   preprocessing values from training rows, then reuse them on validation rows.
4. **From-scratch model.** Begin with zero slopes and an intercept equal to the
   training mean price. Compute a gradient and repeatedly move the parameters in
   the opposite direction. Stop when the largest absolute gradient is small.
5. **Part I.** Compare six preprocessing choices. The selected version replaces
   odometer with age times annual mileage. Validation RMSE is $3,683.24. Preserve
   this baseline before L2 is added.
6. **Part II.** Model year and age are very strongly correlated. Adding L2 shrinks
   slopes, but large penalties increase error. The smallest positive penalty
   improves five-split mean RMSE by only about two cents. Keep the unregularized
   model because the predictions are practically tied.
7. **Reliability.** The tests check the math, preprocessing, and exact CSV output.
   Refit the chosen pipeline on all public rows, then generate one finite price
   for each instructor test ID. Do not tune on the hidden data.

## What each core block does

| Block | Explain it this way |
| --- | --- |
| `split_rows` | Shuffle reproducibly with a seed, then hold out 20% of rows. |
| `Preprocessor.fit` | Learn medians, category vocabulary/frequencies, feature means, and scales from training rows. |
| `Preprocessor.transform` | Reuse those exact stored values and feature order on new rows. |
| Missing indicators | A 1 records that an original numeric value was absent, even after filling it. |
| Unknown-category fallback | Use training category proportions; the standardized category block becomes zero. |
| `objective_gradient` | Directly calculate squared-error loss and its derivatives, with optional L2. |
| Cached `gram` and `cross` | Store repeated matrix products so every update is faster. There is no inverse or equation solver. |
| `theta -= learning_rate * gradient` | Update the coefficients and intercept by gradient descent. |
| `gradient[:-1]` penalty | Add L2 only to slopes; the final parameter is the unpenalized intercept. |
| `part1` / `part2` | Select and preserve the baseline, then perform the controlled regularization experiment. |
| `train_selected` | Fit only the final choice on all public rows and save it. |
| `write_predictions` | Apply the saved pipeline, validate IDs and row count, and write the exact two-column CSV. |

## Questions to practice

**What does RMSE mean?** It is the square root of the average squared prediction
error. It is measured in dollars. Large errors get extra weight, so it is not the
same as average absolute error. Lower RMSE is better.

**Why not use loan approval if its correlation is so high?** The intended timing
is before financing. A later financial value could make validation look excellent
while being unavailable when the prediction is actually needed.

**Why is age times annual mileage allowed in linear regression?** It is one
engineered predictor. Predictions are still linear in the learned coefficients.
It is a mileage-related proxy, not a claim that the true odometer is known exactly.

**Why scale the dummy variables too?** The penalty acts on coefficient size.
Standardizing all nonconstant features makes those sizes more comparable. A
standardized coefficient is not a dollar change per original raw unit.

**Why keep both year and age?** The controlled experiment asks whether L2 helps
with correlated predictors. Their presence does not automatically mean prediction
error will improve with regularization. Individual slopes can be unstable even
when predictions are similar.

**Why are all one-hot levels kept?** Gradient descent does not need an invertible
matrix. Full one-hot encoding plus centering creates redundant columns, so plain
OLS slopes need not be unique. Zero initialization gives a reproducible fitted
solution. L2 gives a unique slope vector for a positive penalty.

**Why is the intercept not penalized?** It represents the overall price level.
The required objective penalizes feature coefficients only. Shrinking the
intercept would pull the overall estimate toward zero without that requirement.

**Did the model fully converge?** Yes. The frozen Part I model met a maximum
absolute gradient tolerance of 0.00001 after 69,583 updates. The larger iteration
cap provides room for other splits. The final all-data run also converged.

**Why not claim that L2 improved the model?** Its five-split mean improvement is
about $0.01849, while split-to-split RMSE varies by hundreds of dollars. That is
not persuasive evidence of a useful improvement. The $1 tie rule was chosen after
reviewing the results and is a practical judgment, not a statistical test.

**Does the score guarantee the competition result?** No. The hidden sample can
have different missingness, new categories, and other distribution changes. The
public validation scores guided development and are not hidden-test scores.

## Classroom command

```bash
python main.py --mode predict --test hidden_data/car_price_test.csv
```

Open the produced `predictions.csv` and confirm the header is
`listing_id,sale_price_usd_pred`. The program checks the required 600 rows and
finite predictions before writing. Keep the hidden CSV out of GitHub.
