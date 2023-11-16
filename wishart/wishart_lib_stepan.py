import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma
from collections import defaultdict
from random import choice, random, randint

from collections import defaultdict
import numpy as np
import pdb
import dill
# from sklearn.datasets.samples_generator import make_blobs
import random
from itertools import combinations, product
from scipy.special import gamma
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform, euclidean
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm
from math import sqrt
import pandas as pd


def volume(r, m):
    return np.pi ** (m / 2) * r ** m / gamma(m / 2 + 1)

def significant(cluster, h, p):
    max_diff = max(abs(p[i] - p[j]) for i, j in product(cluster, cluster))

    # print(max_diff)
    return max_diff >= h

def partition(dist, l, r, order):
    if l == r:
        return l

    pivot = dist[order[(l + r) // 2]]
    left, right = l - 1, r + 1
    while True:
        while True:
            left += 1
            if dist[order[left]] >= pivot:
                break

        while True:
            right -= 1
            if dist[order[right]] <= pivot:
                break

        if left >= right:
            return right

        order[left], order[right] = order[right], order[left]

def nth_element(dist, order, k):
    l, r = 0, len(order) - 1
    while True:
        if l == r:
            break
        m = partition(dist, l, r, order)
        if m < k:
            l = m + 1
        elif m >= k:
            r = m


class WishartClusterization(object):
    def __init__(self, k, h):
        self.k = k
        self.h = h
        
    
    def fit(self, x):
        n = len(x)
        if isinstance(x[0], list):
            m = len(x[0])
        else:
            m = 1
        dist = squareform(pdist(x))

        dk = []
        for i in range(n):
            order = list(range(n))
            nth_element(dist[i], order, self.k - 1)
            dk.append(dist[i][order[self.k - 1]])

        # print(dk)

        p = [self.k / (volume(dk[i], m) * n) for i in range(n)]

        w = np.full(n, 0)
        completed = {0: False}
        last = 1
        vertices = set()
        for d, i in sorted(zip(dk, range(n))):
            neigh = set()
            neigh_w = set()
            clusters = defaultdict(list)
            for j in vertices:
                if dist[i][j] <= dk[i]:
                    neigh.add(j)
                    neigh_w.add(w[j])
                    clusters[w[j]].append(j)

            vertices.add(i)
            if len(neigh) == 0:
                w[i] = last
                completed[last] = False
                last += 1
            elif len(neigh_w) == 1:
                wj = next(iter(neigh_w))
                if completed[wj]:
                    w[i] = 0
                else:
                    w[i] = wj
            else:
                if all(completed[wj] for wj in neigh_w):
                    w[i] = 0
                    continue
                significant_clusters = set(wj for wj in neigh_w if significant(clusters[wj], self.h, p))
                if len(significant_clusters) > 1:
                    w[i] = 0
                    for wj in neigh_w:
                        if wj in significant_clusters:
                            completed[wj] = (wj != 0)
                        else:
                            for j in clusters[wj]:
                                w[j] = 0
                else:
                    if len(significant_clusters) == 0:
                        s = next(iter(neigh_w))
                    else:
                        s = next(iter(significant_clusters))
                    w[i] = s
                    for wj in neigh_w:
                        for j in clusters[wj]:
                            w[j] = s
        self.labels_ = w
        return self.labels_
    
    def fit_with_visualization(self, x, step):
        n = len(x)
        if isinstance(x[0], list):
            m = len(x[0])
        else:
            m = 1
        dist = squareform(pdist(x))

        dk = []
        for i in range(n):
            order = list(range(n))
            nth_element(dist[i], order, self.k - 1)
            dk.append(dist[i][order[self.k - 1]])

        # print(dk)

        p = [self.k / (volume(dk[i], m) * n) for i in range(n)]

        w = np.full(n, 0)
        completed = {0: False}
        last = 1
        vertices = set()
        shots_w = []
        cnt = 0
        for d, i in sorted(zip(dk, range(n))):
            cnt += 1
            if i % step == 0:
                shots_w.append(w)
            neigh = set()
            neigh_w = set()
            clusters = defaultdict(list)
            for j in vertices:
                if dist[i][j] <= dk[i]:
                    neigh.add(j)
                    neigh_w.add(w[j])
                    clusters[w[j]].append(j)

            vertices.add(i)
            if len(neigh) == 0:
                w[i] = last
                completed[last] = False
                last += 1
            elif len(neigh_w) == 1:
                wj = next(iter(neigh_w))
                if completed[wj]:
                    w[i] = 0
                else:
                    w[i] = wj
            else:
                if all(completed[wj] for wj in neigh_w):
                    w[i] = 0
                    continue
                significant_clusters = set(wj for wj in neigh_w if significant(clusters[wj], self.h, p))
                if len(significant_clusters) > 1:
                    w[i] = 0
                    for wj in neigh_w:
                        if wj in significant_clusters:
                            completed[wj] = (wj != 0)
                        else:
                            for j in clusters[wj]:
                                w[j] = 0
                else:
                    if len(significant_clusters) == 0:
                        s = next(iter(neigh_w))
                    else:
                        s = next(iter(significant_clusters))
                    w[i] = s
                    for wj in neigh_w:
                        for j in clusters[wj]:
                            w[j] = s
        self.labels_ = w
        return self.labels_, shots_w