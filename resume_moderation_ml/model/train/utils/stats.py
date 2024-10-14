from typing import Any

import numpy as np


class RunningStatistic:
    def __init__(self, accumulate_sum: bool = False) -> None:
        self.seen = 0
        self._value: Any = None
        self.accumulate_sum = accumulate_sum
        if accumulate_sum:
            self._sum: Any = None

    @property
    def value(self) -> Any:
        return self._value

    @property
    def sum(self) -> Any:
        if not self.accumulate_sum:
            raise Exception("instance doesn't accumulate sum")
        return self._sum

    def feed(self, value: Any, num_samples: int) -> None:
        if not num_samples or not np.isfinite(value):
            return
        if not self.seen:
            self._value = value
        else:
            self._value /= 1.0 + num_samples / self.seen
            self._value += value / (1.0 + self.seen / num_samples)
        self._sum = (0 if not self.seen else self._sum) + value
        self.seen += num_samples

    def reset(self) -> None:
        self.seen = 0
        self._value = None
        self._sum = None

    def __str__(self) -> str:
        if self.accumulate_sum:
            return f"<{self.seen=}, {self.value=}, {self.sum=}>"
        else:
            return f"<{self.seen=}, {self.value=}>"

    def __repr__(self) -> str:
        return self.__str__()


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
