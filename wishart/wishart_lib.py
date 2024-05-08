import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma
from collections import defaultdict
from random import choice, random, randint
from itertools import product
import copy


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
        dif = [abs(p[i] - p[j]) for i, j in product(cluster, cluster)]
        # print(max(dif))
        return max(dif) >= self.u

    def normalize(self, x):
        for i in range(len(x[0])):
            max_val = x[0][i]
            min_val = x[0][i]
            for j in range(len(x)):
                max_val = max(max_val, x[j][i])
                min_val = min(min_val, x[j][i])
            if abs(max_val - min_val) > 0.0001:
                for j in range(len(x)):
                    x[j][i] -= min_val
                    x[j][i] /= max_val - min_val

        return x

    def fit(self, x_):
        x = copy.deepcopy(x_)
        # self.normalize(x)
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
        vertices = set()
        for d, i in sorted(zip(dr, range(n))):
            neighbours = []
            neighbours_w = set()
            clusters = defaultdict(list)
            for j in vertices:
                if dist[i][j] <= d:
                    neighbours.append(j)
                    neighbours_w.add(w[j])
                    clusters[w[j]].append(j)
            vertices.add(i)
            if len(neighbours) == 0:
                w[i] = label
                completed[label] = False
                label += 1
            elif len(neighbours_w) == 1:
                wj = next(iter(neighbours_w))
                if completed[wj]:
                    w[i] = 0
                else:
                    w[i] = wj
            else:
                if all([completed[l] for l in neighbours_w]):
                    w[i] = 0
                    continue
                significant_clusters = set(wj for wj in neighbours_w if self.significant(clusters[wj], p))
                if len(significant_clusters) > 1 or next(iter(neighbours_w)) == 0:
                    w[i] = 0
                    for wj in neighbours_w:
                        if wj in significant_clusters:
                            completed[wj] = (wj != 0)
                            continue
                        for v in clusters[wj]:
                            w[v] = 0
                else:
                    if len(significant_clusters):
                        c1 = next(iter(significant_clusters))
                    else:
                        c1 = next(iter(neighbours_w))
                    w[i] = c1
                    for wj in neighbours_w:
                        for v in clusters[wj]:
                            w[v] = c1

        return w



    def fit_with_visualization(self, x, step):
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
        vertices = set()

        shots_w = []
        cnt = 0
        for d, i in sorted(zip(dr, range(n))):
            cnt += 1
            if cnt % step == 0:
                shots_w.append(w.copy())
            neighbours = []
            neighbours_w = set()
            clusters = defaultdict(list)
            for j in vertices:
                if dist[i][j] <= d:
                    neighbours.append(j)
                    neighbours_w.add(w[j])
                    clusters[w[j]].append(j)
            vertices.add(i)
            if len(neighbours) == 0:
                w[i] = label
                completed[label] = False
                label += 1
            elif len(neighbours_w) == 1:
                wj = next(iter(neighbours_w))
                if completed[wj]:
                    w[i] = 0
                else:
                    w[i] = wj
            else:
                if all([completed[l] for l in neighbours_w]):
                    w[i] = 0
                    continue
                significant_clusters = set(wj for wj in neighbours_w if self.significant(clusters[wj], p))
                if len(significant_clusters) > 1 or next(iter(neighbours_w)) == 0:
                    w[i] = 0
                    for wj in neighbours_w:
                        if wj in significant_clusters:
                            completed[wj] = (wj != 0)
                            continue
                        for v in clusters[wj]:
                            w[v] = 0
                else:
                    if len(significant_clusters):
                        c1 = next(iter(significant_clusters))
                    else:
                        c1 = next(iter(neighbours_w))
                    w[i] = c1
                    for wj in neighbours_w:
                        for v in clusters[wj]:
                            w[v] = c1
        return w, shots_w