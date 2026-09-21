# AI collaboration log

This log summarizes actual work performed with OpenAI ChatGPT/Codex. It is not a
verbatim transcript and does not claim the student has already reviewed the code.

## 1. Requirements and inspection

- User request: complete Competition 1 & 2 using the supplied PDF and public CSV.
- Constraints extracted: plain gradient descent first, separate L2 experiment,
  only standard-library/NumPy/Pandas/Matplotlib code, RMSE for selection, no direct
  fitting shortcuts, train-only preprocessing, and meaningful Git commits.
- Evidence: 1,800 rows, 19 columns, no duplicate IDs; saved missingness, ranges,
  category counts, and pairwise numeric correlations.
- Decision: exclude `listing_id` and `loan_approval_value_usd` from predictors.
  The loan field appears financing-related and is therefore assumed unavailable
  at the stated prediction time. Its correlation with the target is about 0.9991.
  High correlation alone is not the reason for exclusion; availability is.
- Extreme odometer readings are not automatically labeled errors. Compare
  training-fitted treatments and retain all public rows.
- No hidden test data, labels, or synthetic generator were available.

## Git history

Commits record the actual development stages during this session and are
attributed to OpenAI Codex. They are not backdated or presented as student work.

## 2. Preprocessing

- Implemented a fitted preprocessor with numeric medians, missing indicators,
  one-hot categories, and feature standardization.
- Unknown categories use training category proportions, giving a zero-centered
  category block after scaling. No test categories are added to the vocabulary.
- Compared six candidates: raw odometer, a training-fitted 99th-percentile cap,
  log odometer, no odometer, an age-by-annual-mileage interaction, and age squared.
- Smoke checks confirmed fixed feature order and finite transforms with all
  numeric values missing and unknown categories.

## 3. Ordinary gradient descent and Part I

- Implemented plain MSE using iterative full-batch gradient descent. Cached
  products calculate exactly the same gradient faster, without solving equations.
- A known straight line verified the slope and intercept.
- Compared learning rates, iteration limits, and gradient tolerances. A learning
  rate of 0.5 was rejected for increasing the training objective. Slow/incomplete
  runs were recorded and not selected as a converged baseline.
- Saved Part I in a distinct Git commit before implementing L2. Chosen settings:
  age_mileage, rate 0.05, max 150000, tolerance 0.00001, validation RMSE 3683.2353.

## 4. L2 and decision review

- Added the specified lambda times squared coefficient norm, excluding intercept.
- Compared lambda zero and six positive strengths with the same split, feature
  set, preprocessing, initialization, and training settings.
- Verified zero-lambda predictions exactly match the preserved Part I result.
- Tested the selected positive lambda 0.00001 against zero using five additional
  seeds that were specified in code before their results were inspected.
- The first automatic rule chose the lower mean RMSE even for a tiny difference.
  Review found the advantage was just $0.01849. The final rule treats differences
  below $1 as practically tied and keeps the baseline. This rule was introduced
  after seeing that negligible difference; it was not preregistered.
- Conclusion: L2 shrinks coefficients, but these results do not establish a
  meaningful predictive need for regularization. Retain lambda zero.

## 5. Verification and final preparation

- Eight automated tests cover exact toy cases, finite-difference gradients,
  cached versus residual updates, zero-lambda equivalence, unpenalized intercept,
  train-only statistics, unavailable-feature exclusion, extreme missingness,
  unseen categories, column order, malformed inputs, and 600-row CSV output.
- Refit only the selected unregularized pipeline on all 1,800 public rows.
- Created the report, README, presentation notes, and exact hidden-test command.
- No real hidden test was supplied, so no hidden score or real submission
  predictions are claimed. GitHub publication still requires account access.

## Student review to complete

Explain the objective and its gradients, the availability decision, the split,
training-only preprocessing, the learning-rate failure, the regularization
comparison, and the final prediction command. Do not claim to have done this
review until it has actually been completed.
