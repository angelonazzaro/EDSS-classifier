RANDOM_SEED = 42
TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1

CLASS_THRESHOLDS = {
    "binary": {
        "negative": [0.0, 2.0],
        "positive": [2.0, 10.0],
    },
    "multi-class": {
        "normal": [0.0, 2.0],
        "mild": [2.0, 4.0],
        "severe": [4.0, 10.0]
    }
}
