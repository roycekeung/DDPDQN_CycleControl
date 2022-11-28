import enum
class AlgoType(enum.IntEnum):
    Noraml = 0
    Double = 1
    Dueling = 2

class ArchType(enum.IntEnum):
    Dense = 0
    Conv1d_dense = 1
    Conv2d_dense = 2
    