import numpy as np


def select_threshold(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray, threshold_params: dict
) -> float:
    thresholds = np.r_[thresholds, 1.0]

    if "precision" in threshold_params:
        desired = threshold_params["precision"]
        feasible = np.where(precision >= desired)[0]
        if len(feasible) == 0:
            raise ValueError("can' find threshold with precision greater than %.5f", desired)
        best = np.argmax(recall[feasible])
        return thresholds[feasible][best]
    elif "recall" in threshold_params:
        desired = threshold_params["recall"]
        feasible = np.where(recall >= desired)[0]
        if len(feasible) == 0:
            raise ValueError("can' find threshold with precision greater than %.5f", desired)
        best = np.argmax(precision[feasible])
        return thresholds[feasible[best]]

    return 0.5
