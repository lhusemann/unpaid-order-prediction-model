# Predicting Unpaid Orders in an Online Shop

A machine learning project from my bachelor's degree given to me by the [Decision and Operation Technologies Group](https://www.uni-bielefeld.de/fakultaeten/wirtschaftswissenschaften/lehrbereiche/dot/)
of Bielefeld University. The goal is to estimate the probability that an order in a fictional online shop is **not paid for**, using information about the order, the customer, and the customer's previous orders.

The whole project is in a single Jupyter notebook, [`unpaid_order_prediction.ipynb`](unpaid_order_prediction.ipynb).

## Task

I was given a labelled training set (`train.csv`) where each order is described by a set of features and a label that says whether the customer paid. Using this data I had to build a model and then predict, for every order in a second, unlabelled dataset (`test.csv`), how likely it is that the order will not be paid.

| File        | Rows   | Description                                   |
|-------------|--------|-----------------------------------------------|
| `train.csv` | 40,000 | Orders with features and the `TARGET_FRAUD` label |
| `test.csv`  | 10,000 | Orders with the same features, no label       |

The dataset is imbalanced: only about **5.9 %** of the training orders (2,341 of 40,000) are unpaid.

## Approach

### 1. Preprocessing
- **One-hot encoding** of the categorical columns `P_METHOD`, `P_CARD_TYPE` and `DEL_DAY`.
- **Missing values** in the reminder levels (`REM_CUR`, `REM_MAX`) are filled with `0` (no reminder so far).
- **Dropped columns:**
  - `ANUMBER_01`–`ANUMBER_05`: I tried one-hot encoding the article numbers, but it only made the model less accurate.
  - `P_CARD_TYPE_nan`, `P_CARD_TYPE_Kundenkarte` and `P_METHOD_Credit Card`: these duplicate information already carried by other encoded columns.

### 2. Feature engineering
I added five features, each based on an assumption about how fraudulent orders behave:

| Feature                | Description |
|------------------------|-------------|
| `WEEKEND`              | The order was placed on a Saturday or Sunday. My assumption is that genuine customers shop more on weekends, while professional fraudsters tend to work on weekdays. |
| `AVG_ITEM_COST`        | Average price per item in the current order (`DEL_VALUE / DEL_AMOUNT`). |
| `AVG_ITEM_COST_USUALLY`| Average price per item across the customer's previous orders. |
| `DISTANCE_USUAL_COST`  | Absolute difference between the two values above. A large gap may point to unusual behaviour. |
| `RECENTLY_ORDERED`     | Combines all `CHK_*` flags into one: was there another order with the same address, account, card, cookie or IP in the last 3 days? |

Recursive feature elimination (`RFECV`) ranked four of these new features among the top 3 ranks. Removing the lowest-ranked features made accuracy worse, so I kept all features.

### 3. Hyperparameter tuning
I tuned each base model separately with [Optuna](https://optuna.org/) (150–200 trials per model), using 5-fold stratified cross-validation:

- Random Forest
- Logistic Regression
- Gradient Boosting
- Decision Tree

The tuning and feature selection cells are commented out in the notebook because they take a long time to run. The best parameters they found are written into the training cell.

### 4. Final model
The final model is a **soft-voting ensemble** (`VotingClassifier`) of the four tuned models, trained on standardised features. A stratified 67/33 train/validation split is used for evaluation.

## Results

| Metric   | Train  | Validation |
|----------|--------|------------|
| Accuracy | 0.9421 | 0.9414     |
| ROC AUC  | 0.8358 | 0.7434     |

The model writes its predicted probabilities for `test.csv` to `final_test_prediction.csv` (`Id`, `TARGET_FRAUD`).

**A note on accuracy:** because only about 5.9 % of orders are unpaid, a model that always predicts "paid" would already reach about 94.1 % accuracy. The accuracy score therefore says little here. ROC AUC is the more useful metric for this task, since the goal is to output a probability rather than a hard yes/no label. In hindsight, it would have been better to tune the models for ROC AUC (or log loss) instead of accuracy.

## Running the notebook

The notebook was written in Google Colab, but it also runs in any local Jupyter environment.

```bash
pip install pandas numpy scikit-learn jupyter
pip install optuna   # only needed if you re-enable the tuning cells
jupyter notebook unpaid_order_prediction.ipynb
```

`train.csv` and `test.csv` must be in the same directory as the notebook. All random processes use the fixed seed `13`, so results can be reproduced.

## Feature description

| #  | Feature              | Description |
|----|----------------------|-------------|
| 1  | `O_EMAIL`            | Was an email address provided during the order (yes/no) |
| 2  | `O_TELEPHONE`        | Was a phone number provided during the order (yes/no) |
| 3  | `FLAG_DIIDENTICAL`   | Delivery and billing address are identical (yes/no) |
| 4  | `FLAG_NEWSLETTER`    | Was the newsletter ordered during the order (yes/no) |
| 5  | `CHK_LADR`           | Orders with the same delivery address within 3 days (yes/no) |
| 6  | `CHK_RADR`           | Orders with the same billing address within 3 days (yes/no) |
| 7  | `CHK_KTO`            | Orders with the same account connection within 3 days (yes/no) |
| 8  | `CHK_CARD`           | Orders with the same card number within 3 days (yes/no) |
| 9  | `CHK_COOKIE`         | Orders with the same browser cookie within 3 days (yes/no) |
| 10 | `CHK_IP`             | Orders with the same browser IP within 3 days (yes/no) |
| 11 | `FAIL_LPLZ`          | Delivery address: ZIP code unknown (yes/no) |
| 12 | `FAIL_LORT`          | Delivery address: city unknown (yes/no) |
| 13 | `FAIL_LPLZORTMATCH`  | Delivery address: ZIP code and city do not match (yes/no) |
| 14 | `FAIL_RPLZ`          | Billing address: ZIP code unknown (yes/no) |
| 15 | `FAIL_RORT`          | Billing address: city unknown (yes/no) |
| 16 | `FAIL_RPLZORTMATCH`  | Billing address: ZIP code and city do not match (yes/no) |
| 17 | `NEW_CUSTOMER`       | Is the customer a new customer (yes/no) |
| 18 | `P_METHOD`           | Selected payment method |
| 19 | `P_CARD_TYPE`        | Selected card type, if payment by card |
| 20 | `DEL_DAY`            | Day of the week of the order |
| 21 | `DEL_AMOUNT`         | Number of items ordered |
| 22 | `DEL_VALUE`          | Order value in euros |
| 23 | `SESSION_TIME`       | Duration of the online ordering session in minutes |
| 24 | `DEL_AMOUNT_TOTAL`   | Total number of items ordered in previous orders, if available |
| 25 | `DEL_VALUE_TOTAL`    | Total value of previous orders, if available |
| 26 | `REM_CUR`            | Current reminder level of the customer, if available |
| 27 | `REM_MAX`            | Highest reminder level the customer has reached so far, if available |
| 28 | `ANUMBER_01`         | Article number of an ordered item |
| 29 | `ANUMBER_02`         | Article number of an ordered item |
| 30 | `ANUMBER_03`         | Article number of an ordered item |
| 31 | `ANUMBER_04`         | Article number of an ordered item |
| 32 | `ANUMBER_05`         | Article number of an ordered item |
| 33 | `TARGET_FRAUD`       | Target: the order was not paid for |

## LLM notice

This README was partially written using *Claude (Anthropic, Claude Code)*.
The code itself was written entirely without the help of an LLM.
