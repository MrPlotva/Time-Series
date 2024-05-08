import numpy as np

def GetElementsForPattern(data: np.array, pattern: list, pos: int):
    return [data[pos - id] for id in pattern][::-1]

def Mode(a):
    return max(set(a), key=a.count)

def WeightedMedian(values : np.array, weights : np.array):
    #pairs = np.vstack((values, weights)).T
    sum = weights.sum() / 2
    order = np.lexsort((weights, values))
    for id in order:
        sum -= weights[id]
        if sum <= 0:
            return values[id]
    raise RuntimeError("How the f")


def Normilize(values : np.array):
    return (values - values.min()) / (values.max() - values.min())