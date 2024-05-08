import numpy as np
from .wishart_lib import Wishart
from itertools import groupby
from .utility import *

def ClassterCenter(vecs):
    best_dist = 1e18
    best_vec = None
    for v in vecs:
        dist = np.sqrt(np.sum((vecs - v)**2, axis=1)).sum()
        if best_dist > dist:
            best_dist = dist
            best_vec = v
    return best_vec

def ClassterWeight(vecs):
    return len(vecs)

def GetCenters(x_train, clip_bound, WISHART_R, WISHART_U):
    wishart = Wishart(WISHART_R, WISHART_U)
    labels = wishart.fit(x_train[:,0:clip_bound])
    sorted_by_cluster = sorted(range(len(labels)), key=lambda x: labels[x])
    centers = []
    for wi, cluster in groupby(sorted_by_cluster, lambda x: labels[x]):
        if wi == 0:
            continue
        cluster = list(cluster)
        vecs = np.array([x_train[id] for id in cluster])
        centers.append([ClassterCenter(vecs), ClassterWeight(vecs)])

    return centers


def GetMotifs(data, patterns, horizon, WISHART_R = 10, WISHART_U = 0.2):
    centers = dict()
    sz = len(data) - horizon
    max_pattern_len = max([max(i) for i in patterns])

    # Добавить многопоточку сюда

    for pattern in patterns:
        all_vecs = [np.concatenate((np.array(GetElementsForPattern(data, pattern, id)), 
                                    data[id:id+horizon])) 
                    for id in range(max_pattern_len, sz)]
        centers[tuple(pattern)] = GetCenters(np.array(all_vecs), len(pattern), WISHART_R, WISHART_U)

    return centers

def SaveCenters(centers, file):
    arr_to_save = []
    for key in centers:
        cur = [key, centers[key]]
        arr_to_save.append(cur)
    np.save(file, np.array(arr_to_save, dtype=object))