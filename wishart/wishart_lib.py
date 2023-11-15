import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma
from collections import defaultdict
from random import choice, random, randint

def kth_element(arr, l: int, r: int, k: int):
    if r - l == 1:
        return arr[l]
    pivot = choice(arr[l: r])
    lq = l
    rq = r - 1
    while lq <= rq:
        if arr[lq] <= pivot:
            lq += 1
        else:
            arr[lq], arr[rq] = arr[rq], arr[lq]
            rq -= 1
    print(l, r, pivot, k)
    if k < lq - l:
        return kth_element(arr, l, lq, k)
    return kth_element(arr, lq, r, k - (lq - l))

def get_kth_element(arr, k):
    k = min(k, len(arr) - 1)
    # return kth_element(arr, 0, len(arr), k)
    return sorted(arr)[k]


def volume(r, m):
    return np.pi ** (m / 2) * r ** m / gamma(m / 2 + 1)


class Wishart:
    def __init__(self, r, u):
        self.radius = r
        self.u = u

    def significant(self, cluster, p):
        dif = [abs(p[i] - p[j]) for i, j in zip(cluster, cluster)]
        return max(dif) >= self.u


    def fit(self, x):
        n = len(x)
        m = len(x[0])
        dist = squareform(pdist(x))
        dr = []
        # print(dist)
        for i in range(n):
            dr.append(get_kth_element(dist[i], self.radius - 1))

        p = [self.radius / (volume(i, m) * n) for i in dr]
        label = 1
        w = np.full(n, 0)
        completed = {0: False}
        vertices = []
        for d, i in sorted(zip(dr, range(n))):
            neighbours = []
            neighbours_w = []
            clusters = defaultdict(list)
            for j in vertices:
                if dist[i][j] <= d:
                    neighbours.append(j)
                    neighbours_w.append(w[j])
                    clusters[w[j]].append(j)
            vertices.append(i)
            if len(neighbours) == 0:
                w[i] = label
                completed[label] = False
                label += 1
            elif len(neighbours_w) == 1:
                wj = min(neighbours_w)
                if completed[wj]:
                    w[i] = 0
                else:
                    w[i] = wj
            else:
                if all([completed[l] for l in neighbours_w]):
                    w[i] = 0
                    continue
                significant_clusters = set(wj for wj in neighbours_w if self.significant(clusters[wj], p))
                if len(significant_clusters) > 1:
                    w[i] = 0
                    for wj in neighbours_w:
                        if wj in significant_clusters:
                            completed[wj] = (wj != 0)
                            continue
                        for v in clusters[wj]:
                            w[v] = 0
                else:
                    if len(significant_clusters) == 1:
                        c1 = next(iter(significant_clusters))
                    else:
                        c1 = min(neighbours_w)
                    w[i] = c1
                    for wj in neighbours_w:
                        for v in clusters[wj]:
                            w[v] = c1

        return w