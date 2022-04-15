import numpy as np

TRUE_VAL_ROCKS_COUNT = [11, 9, 6, 8, 6, 7, 9, 11, 11, 14, 11]
TRUE_VAL_ROCKS_AREAS = []


def count_mse_count(predicted_counts: list):
    "log MSE on val dataset"
    MSE = np.sum((np.array(predicted_counts) - np.array(TRUE_VAL_ROCKS_COUNT)) ** 2) / len(TRUE_VAL_ROCKS_COUNT)
    print (f"MSE on val_dataset: {MSE}")


def count_mse_biggest_area(pred_biggest_sizes: list):
    "log MSE of biggest rock size"
    MSE = np.sum((np.array(pred_biggest_sizes) - np.array(TRUE_VAL_ROCKS_COUNT)) ** 2) / len(TRUE_VAL_ROCKS_AREAS)
    print (f"MSE on val_dataset: {MSE}")
