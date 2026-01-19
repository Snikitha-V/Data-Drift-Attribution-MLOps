TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"

RANDOM_STATE = 42

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "random_state": RANDOM_STATE,
}
