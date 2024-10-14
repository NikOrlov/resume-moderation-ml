import numpy as np


class ModerationTargets(object):
    def __init__(self):
        self.statuses = []
        self.validation_schema = []
        self.flags = []

    def add(self, status, incompleted, flags):
        self.statuses.append(status)
        self.validation_schema.append(incompleted)
        self.flags.append(set(flags))

    @property
    def size(self):
        return len(self.statuses)

    @property
    def approve_target(self):
        return np.array([status == "approved" for status in self.statuses], dtype=np.int32)

    @property
    def incompleted(self):
        return np.array([schema == "incomplete" for schema in self.validation_schema], dtype=np.int32)

    def get_flag_target(self, flag):
        return np.array([flag in self.flags[idx] for idx in range(len(self.flags))], dtype=float)
