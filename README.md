# Used-car price prediction from scratch

Competition 1 & 2. Predict `sale_price_usd` after inspection and before negotiation
and financing. The only third-party dependencies are NumPy, Pandas, and Matplotlib.
No regression library, optimizer, inverse, or direct least-squares solver is used.

## Setup

Use Python 3.12 (tested with 3.12.14).

```bash
python -m pip install -r requirements.txt
python inspect_data.py
```

The supplied public training data is `data/car_price_train.csv`. Keep instructor
hidden features and labels outside Git, preferably in the ignored `hidden_data/`.

## Development stages

1. Inspect the data and identify unavailable predictors.
2. Implement reusable preprocessing.
3. Implement and validate ordinary gradient descent, then freeze Part I.
4. Add L2 and run controlled comparisons.
5. Refit the selected pipeline and prepare the prediction interface.

AI assistance: OpenAI ChatGPT/Codex generated and checked the implementation,
experiments, and documentation. The student must review the core code before the
presentation. No claim of student review is made by this repository.

