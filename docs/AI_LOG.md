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
