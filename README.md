# Used-car price prediction from scratch

Competition 1 & 2. Predict `sale_price_usd` after inspection and before negotiation
and financing. The only third-party dependencies are NumPy, Pandas, and Matplotlib.
No regression library, optimizer, inverse, or direct least-squares solver is used.

## Setup and quick run

Use Python 3.12 (tested with 3.12.14).

```bash
python -m pip install -r requirements.txt
python main.py --mode train
python -m unittest -v test_pipeline
```

Run commands from this project folder. The selected settings and all experiment
results are already supplied. `--mode train` fits only the selected pipeline on all
1,800 public rows. It saves `artifacts/model.pkl`, its metadata, coefficients, and
loss trace. Pickle files should only be loaded from a trusted source.

## Predict the instructor's hidden CSV

Create `hidden_data/` and put the instructor's CSV there. Then run:

```bash
python main.py --mode predict --test hidden_data/car_price_test.csv
```

This writes `predictions.csv` in the project folder with exactly:

```text
listing_id,sale_price_usd_pred
```

The default check requires 600 rows and unique, nonempty IDs. Every prediction must
be finite. Input order is preserved. Missing predictor values and unseen category
levels are supported; required predictor columns must still be present. If the
instructor explicitly supplies a different row count, set `--expected-rows N`.
There is no fabricated `predictions.csv` in this project: the real hidden CSV has
not been provided. A separate temporary 600-row fixture was used only to test I/O.

The train/test/output paths are near the top of `main.py`: `TRAIN_PATH`, `TEST_PATH`,
and `PREDICTIONS_PATH`. They can also be set with `--train`, `--test`, and `--output`.
The training default is `data/car_price_train.csv`. If no saved model exists,
predict mode trains the already selected configuration first. If its configuration
or training file has changed, rerun `--mode train`. Predictions are raw linear-model
outputs; no post-hoc rounding, nonnegative clipping, or hidden-data tuning is used.

## Reproduce all development experiments

```bash
python inspect_data.py
python experiments.py --part 1
python experiments.py --part 2
python main.py --mode train
python -m unittest -v test_pipeline
```

Run Part I first. Its selected baseline, parameters, validation predictions, and
loss trace are saved before Part II. The same gradient-descent class implements
both parts. Part II checks that lambda zero reproduces the frozen predictions.
Model fitting uses no `np.linalg` functions, inverse, pseudoinverse, polynomial
fitting helper, or optimization routine. The cached matrix products in gradient
descent simply speed up the exact same iterative gradient updates.

## Main findings

- Exclude ID, target, and financing-related `loan_approval_value_usd` from predictors.
- Use an 80/20 split with seed 42: 1,440 training and 360 validation rows.
- Fit medians, category levels, category frequencies, means, and scales on training
  rows only. Refit them separately within every additional split.
- Of six feature/preprocessing choices, `age_mileage` had the lowest seed-42 RMSE.
  It replaces odometer with age times annual mileage and retains all public rows.
- Frozen Part I: learning rate 0.05, maximum 150,000 updates, gradient tolerance
  0.00001; validation RMSE **$3,683.24**; 49 transformed features.
- Compare lambda 0 with six positive strengths from 0.00001 through 1.0 while
  holding all other settings fixed. The best positive strength is 0.00001.
- Across five additional split seeds, mean RMSE is **$3,904.69** without L2 and
  **$3,904.67** with that positive strength. The improvement is only $0.01849.
- Keep **lambda = 0**. Sub-dollar RMSE differences are treated as a practical tie.
  This tie rule was added after reviewing the negligible measured differences.
  It is not a statistical significance test. L2 is not necessary on this evidence.

These are development validation results, not a hidden-test score. Repeated
holdouts overlap, and the same public data informed model selection. Their mean
does not constitute an independent final performance estimate.

## Files to read

| File | Purpose |
| --- | --- |
| `preprocessing.py` | Train-only missing-value handling, encoding, and scaling |
| `regression.py` | From-scratch MSE/L2 gradient descent with `fit` and `predict` |
| `experiments.py` | Feature/optimizer selection, frozen baseline, controlled L2 runs |
| `main.py` | Final refit and hidden-test prediction command |
| `test_pipeline.py` | Eight correctness and I/O tests |
| `docs/Competition_Report.pdf` | Required two-page technical report |
| `docs/PRESENTATION_NOTES.md` | Code walkthrough and presentation preparation |
| `docs/AI_LOG.md` | Actual AI assistance and verification decisions |
| `results/` | Numeric evidence, configurations, and diagnostic plots |

## GitHub and submission

This ZIP includes an initialized `.git` folder and meaningful development commits.
Use `git log --oneline` to inspect them. They were created as the work progressed,
including a frozen Part I commit before L2 was implemented. They are attributed
to OpenAI Codex. A local Git repository is not yet a GitHub-hosted repository.

To publish it yourself, create an empty repository on GitHub, choose visibility
according to course policy, and do not add a README or other initial files there.
From this folder run, replacing the URL with your actual repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

This preserves the existing history. GitHub authentication is required. Keep
instructor hidden features and labels inside ignored `hidden_data/`. The
`.gitignore` also excludes virtual environments, caches, temporary files, hidden
CSV patterns, saved model artifacts, and `predictions.csv`.

The supplied `Week 1/Competition.pdf`, slide 8, requires one team solution package:

1. A **two-page technical report** covering the problem, method, experimental
   design, results, limitations, and conclusions. Use `docs/Competition_Report.pdf`.
2. Source code **via GitHub**, readable and executable by the opposing team.
3. A README with installation, dependencies, commands, and expected outputs.

The team should provide the report and the actual GitHub repository URL with its
Canvas submission. The slides do not specify exactly where Canvas should receive
the URL; follow any assignment-page instructions for that field or attachment.
Ensure the instructor and reviewing team can access the repository if it is private.
The posted deadline is September 22, 2026 at 11:59 pm. One submission is required
per team, not a separate competing repository for every student.

The September 29 presentation is a **20-minute talk/live demonstration followed
by 10 minutes of Q&A** (slide 9). Reproduce a main result from the submitted code.
Everyone must understand the entire project because the other team may choose
any member to answer. The notes include a timed outline and live-demo commands.

The opposing-team review is a separate, later **one-page evidence-based report**
(slide 11), and the team must actively ask **at least five technical questions**
(slide 13). Actual review findings require the opposing team's files and cannot
be written honestly before those files are received.

The assignment handout, section 5, says the instructor provides the 600-row hidden
CSV **during the final test**. It is not part of the initial two-file release and
is not needed to prepare the model or submit the development work. Watch the
instructor's release instructions rather than assuming a test CSV is already in
Canvas. Never commit hidden features or labels to the repository.

AI assistance: OpenAI ChatGPT/Codex generated and checked the implementation,
experiments, and documentation. The student must review the core code before the
presentation. No claim of student review is made by this repository. The handout
explicitly permits AI to generate the program. `docs/AI_LOG.md` records its use.

## Team and contributions

- Sasha Maric: ran the Codex sessions that generated the implementation, experiments, and report (the commits authored "codex").
- Christopher Chalas: set up the team GitHub repository and ran the fresh-clone verification.
- Rami: tested the code and verified that the data analysis was accurate and the pipeline worked as intended.
