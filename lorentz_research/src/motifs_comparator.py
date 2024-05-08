from collections import defaultdict
import random
from math import gamma

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product
import copy
from scipy.interpolate import make_interp_spline
from scipy.spatial.distance import squareform, pdist


class Lorenz:
    def __init__(self, s=10.0, r=28.0, b=8 / 3):
        self.s = s
        self.r = r
        self.b = b

    # Differential equations of a Lorenz System
    def X(self, x, y, s):
        return s * (y - x)

    def Y(self, x, y, z, r):
        return (-x) * z + r * x - y

    def Z(self, x, y, z, b):
        return x * y - b * z

    # RK4 for the differential equations
    def RK4(self, x, y, z, s, r, b, dt):
        k_1 = self.X(x, y, s)
        l_1 = self.Y(x, y, z, r)
        m_1 = self.Z(x, y, z, b)

        k_2 = self.X((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), s)
        l_2 = self.Y((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), (z + m_1 * dt * 0.5), r)
        m_2 = self.Z((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), (z + m_1 * dt * 0.5), b)

        k_3 = self.X((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), s)
        l_3 = self.Y((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), (z + m_2 * dt * 0.5), r)
        m_3 = self.Z((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), (z + m_2 * dt * 0.5), b)

        k_4 = self.X((x + k_3 * dt), (y + l_3 * dt), s)
        l_4 = self.Y((x + k_3 * dt), (y + l_3 * dt), (z + m_3 * dt), r)
        m_4 = self.Z((x + k_3 * dt), (y + l_3 * dt), (z + m_3 * dt), b)

        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt * (1 / 6)
        y += (l_1 + 2 * l_2 + 2 * l_3 + l_4) * dt * (1 / 6)
        z += (m_1 + 2 * m_2 + 2 * m_3 + m_4) * dt * (1 / 6)

        return (x, y, z)

    def generate(self, dt=0.1, steps=100000):
        # Initial values and Parameters
        x_0, y_0, z_0 = 1, 1, 1

        # RK4 iteration
        x_list = [x_0]
        y_list = [y_0]
        z_list = [z_0]

        i = 0

        while i < steps:
            x = x_list[i]
            y = y_list[i]
            z = z_list[i]

            position = self.RK4(x, y, z, self.s, self.r, self.b, dt)

            x_list.append(position[0])
            y_list.append(position[1])
            z_list.append(position[2])

            i += 1

        x_array = np.array(x_list)
        y_array = np.array(y_list)
        z_array = np.array(z_list)

        return x_array, y_array, z_array


def lorenz_generation(s=10.0, r=28.0, b=8 / 3):
    data, _, _ = Lorenz(s, r, b).generate()
    data = data[250:]
    data = (data - data.min()) / (data.max() - data.min())
    return data


def lorenz_visualisation(data):
    plt.figure(figsize=(20, 8))
    plt.plot(data[:2500])
    plt.xticks([i for i in range(0, 2500, 100)])
    plt.grid()
    plt.show()


class Daemon:
    def __init__(self, mode="simple", is_pred=True, quantiles=(0, 1), gap=0.05):
        self.mode = mode
        self.is_pred = is_pred
        self.quantiles = quantiles
        self.gap = gap

    def mean_d(self, preds):
        sum_weight = sum(map(lambda x: x[1], preds))
        s = sum(map(lambda x: x[0] * x[1], preds))
        return s / sum_weight

    def mean_q(self, preds):
        sum_weight = sum(map(lambda x: x[2], preds))
        s = sum(map(lambda x: x[0] * x[2], preds))
        return s / sum_weight

    def mean_d_q(self, preds):
        cleaned = []
        vals = np.array(preds)[:, 0]
        df = pd.DataFrame(vals)
        low = df[0].quantile(self.quantiles[0])
        high = df[0].quantile(self.quantiles[1])
        for elem in preds:
            if low <= elem[0] <= high:
                cleaned.append(elem)
        if len(cleaned) == 0:
            return None
        sum_weight = sum(map(lambda x: x[1] * x[2], cleaned))
        s = sum(map(lambda x: x[0] * x[1] * x[2], cleaned))
        return s / sum_weight

    def predict(self, possible_values):
        if self.is_pred and not self.is_predictable(possible_values):
            return None
        if self.mode == "simple":
            return np.mean(list(map(lambda x: x[0], possible_values)))
        elif self.mode == "simple_d":
            return self.mean_d(possible_values)
        elif self.mode == "simple_q":
            return self.mean_q(possible_values)
        else:
            return self.mean_d_q(possible_values)
        # return self.mean_q(self.possible_values)

    def is_predictable(self, possible_values):
        vals = np.array(possible_values)[:, 0]
        df = pd.DataFrame(vals)
        low = df[0].quantile(self.quantiles[0])
        high = df[0].quantile(self.quantiles[1])
        if high - low > self.gap:
            return False
        return True


class IdealDeamon(object):
    def __init__(self, real_vals, eps=0.05, mode='simple'):
        self.eps = eps
        self.mode = mode
        self.real_vals = real_vals
        # self.predictions = po

    @property
    def label(self):
        return 'Ideal model of demon'

    def predict(self, start_point, step, prediction):
        if abs(prediction - self.real_vals[start_point + step]) > self.eps:
            return None
        return prediction

    def is_predictable(self, start_point, step, prediction):
        return abs(prediction - self.real_vals[start_point + step]) <= self.eps


def GenPatterns(L, x):
    elements = range(1, x + 1)  # Создаем последовательность элементов от 1 до x
    sequences = product(elements, repeat=L)  # Генерируем все возможные комбинации длиной L
    return sequences


def get_val_for_pattern_and_pos(data: np.array, pattern: list, pos: int, bad):
    val = []
    sum = 0
    for i in range(len(pattern) - 1, -1, -1):
        sum += pattern[i]
        val.append(data[pos - sum])
        if bad[pos - sum]:
            return np.array([])
    val = val[::-1]
    return np.array(val)


def base_prediction(data, daemon: Daemon, h: int, L: int = 3, kmax: int = 10, eps: float = 0.1, QVALUE=0.99,
                    return_possible_values=False, centers=None):
    t = len(data)
    prediction = np.zeros(shape=(t + h, 2))
    bad = np.array([0 for i in range(t + h)])
    for i in range(t):
        prediction[i][0] = data[i]
        prediction[i][1] = 1
    possible_values = [[] for i in range(h)]

    steps = 0
    for i in range(h):

        for pattern in GenPatterns(L - 1, kmax):
            val_for_pattern_with_q = get_val_for_pattern_and_pos(prediction, pattern, t + i, bad)

            if len(val_for_pattern_with_q) == 0:
                continue
            val_for_pattern = val_for_pattern_with_q[:, 0]
            val_q = val_for_pattern_with_q[:, 1]

            for c in centers[pattern]:
                if len(c) == 0:
                    continue
                steps += 1
                dist = np.linalg.norm(c[:-1] - val_for_pattern)
                if dist < eps:
                    weight_d = (eps - dist) / eps
                    weight_q = np.mean(val_q) * QVALUE
                    possible_values[i].append([c[-1], weight_d, weight_q])
                #   possible_values[i].append([c[-1], weight_d, weight_q])

        if len(possible_values[i]):
            pred = daemon.predict(possible_values[i])
            if pred is not None:
                prediction[t + i][0] = pred
            else:
                bad[t + i] = 1
                prediction[t + i][0] = 0
            prediction[t + i][1] = np.mean(list(map(lambda x: x[2], possible_values[i])))
        else:
            bad[t + i] = 1
            prediction[t + i][0] = 0

        # print(prediction[t + i])
    if return_possible_values:
        return [prediction, bad, possible_values]
    return [prediction, bad]


def base_prediction_ideal(data, daemon: Daemon, ideal_daemon: IdealDeamon, h: int, L: int = 3, kmax: int = 10,
                          eps: float = 0.1, QVALUE=0.99,
                          return_possible_values=False, centers=None):
    t = len(data)
    prediction = np.zeros(shape=(t + h, 2))
    bad = np.array([0 for i in range(t + h)])
    for i in range(t):
        prediction[i][0] = data[i]
        prediction[i][1] = 1
    possible_values = [[] for i in range(h)]

    steps = 0
    for i in range(h):

        for pattern in GenPatterns(L - 1, kmax):
            val_for_pattern_with_q = get_val_for_pattern_and_pos(prediction, pattern, t + i, bad)

            if len(val_for_pattern_with_q) == 0:
                continue
            val_for_pattern = val_for_pattern_with_q[:, 0]
            val_q = val_for_pattern_with_q[:, 1]

            for c in centers[pattern]:
                if len(c) == 0:
                    continue
                steps += 1
                dist = np.linalg.norm(c[:-1] - val_for_pattern)
                if dist < eps:
                    weight_d = (eps - dist) / eps
                    weight_q = np.mean(val_q) * QVALUE
                    possible_values[i].append([c[-1], weight_d, weight_q])
                #   possible_values[i].append([c[-1], weight_d, weight_q])

        if len(possible_values[i]):
            pred = ideal_daemon.predict(0, i, daemon.predict(possible_values[i]))
            if pred is not None:
                prediction[t + i][0] = pred
            else:
                bad[t + i] = 1
                prediction[t + i][0] = 0
            prediction[t + i][1] = np.mean(list(map(lambda x: x[2], possible_values[i])))
        else:
            bad[t + i] = 1
            prediction[t + i][0] = 0

    return [prediction, bad, possible_values]


def smooth_plot(x, y, plt):
    xnew = np.linspace(x.min(), x.max(), 1000)

    spl = make_interp_spline(x, np.array(y), k=3)
    power_smooth = spl(xnew)
    line, = plt.plot(xnew, power_smooth)
    return line


def values_by_motif(data: np.array, motif: list):
    val = []
    for i in motif:
        val.append(data[i])
    return np.array(val)


def get_metrics(data, h, start_points, centers):
    results = [0] * h
    bads = [0] * h
    for st in start_points:
        prediction, bad, possible = base_prediction_ideal(data[:st], Daemon(mode="simple_d_q", is_pred=False),
                                                          IdealDeamon(data[st:], 0.05), h, L=4, eps=0.009,
                                                          return_possible_values=True, centers=centers)
        prediction = prediction[-h:]
        bad = bad[-h:]
        for hz in range(0, h):
            bads[hz] += bad[hz]
            if bad[hz]:
                continue
            results[hz] += (prediction[hz][0] - data[st + hz]) ** 2

    for i in range(h):
        if len(start_points) == bads[i]:
            results[i] = 0
        else:
            results[i] /= (len(start_points) - bads[i])
        bads[i] /= len(start_points)
        results[i] **= 0.5
        # 0.5 cause RMSE

    return results, bads


def draw_graphs(val, lx, ly, hz):
    plt.figure(figsize=(18, 12))
    smooth_plot(np.array(range(1, hz + 1)), np.array(val), plt).set_label("Simple")
    plt.xlabel(lx, size=20)
    plt.ylabel(ly, size=20)
    plt.show()


def GenerateMotifsByPattern(pattern, t):
    L = len(pattern)
    idx = []
    idx.append(0)
    for i in range(L):
        idx.append(idx[len(idx) - 1] + pattern[i])
    motifs = []
    while idx[len(idx) - 1] != t + 1:
        motifs.append(idx.copy())
        for i in range(len(idx)):
            idx[i] += 1
    return motifs


def GenerateAllMotifs(Kmax, L, t):
    pattern = []
    for i in range(L - 1):
        pattern.append(0)
    patterns = GenPatterns(L - 1, Kmax)
    motifsByPatterns = []
    for p in patterns:
        motifs = GenerateMotifsByPattern(p, t)
        motifsByPatterns.append([p, motifs])
    return motifsByPatterns


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


from itertools import groupby

WISHART_R = 10
WISHART_U = 0.2


def get_centers(x_train):
    N = len(x_train[0])
    wishart = Wishart(WISHART_R, WISHART_U)
    labels = wishart.fit(x_train)
    sorted_by_cluster = sorted(range(len(labels)), key=lambda x: labels[x])
    centers = []
    for wi, cluster in groupby(sorted_by_cluster, lambda x: labels[x]):
        if wi == 0:
            continue
        cluster = list(cluster)
        center = np.full(N, 0.0)
        for i in cluster:
            center += x_train[i]
        centers.append(center / len(cluster))

    return centers


def save_centers(centers, file):
    arr_to_save = []

    for key in centers:
        cur = [key, centers[key]]
        arr_to_save.append(cur)

    np.save(file, np.array(arr_to_save, dtype=object))
    # arr = np.load('centers.npy', allow_pickle=True)
    # return arr
    # print(arr)


S = 10
R = 28
B = 8 / 3
ROWS_CNT = 10
# Train size must be >= 50
TRAIN_SIZE = 1000
TEST_SIZE = 10000
kmax = 10
L = 4
MEANING_PTS = 1000
PRECALCULATED = False


def precalc_lorentz_by_eps(eps: float, index: int):
    mtf = GenerateAllMotifs(kmax, L, TRAIN_SIZE - 1)
    mtf_by_patterns = defaultdict(list)
    centers = dict()
    if not PRECALCULATED:
        for it in range(ROWS_CNT):
            r = R - eps + 2.0 * eps / (ROWS_CNT - 1) * it
            print(r)
    return centers


epsilons = []
eps_left = 0.00000001
eps_right = 1
for i in range(ROWS_CNT):
    epsilons.append(eps_left + (eps_right - eps_left) / (ROWS_CNT - 1.0) * i)

from multiprocessing import Pool
import itertools
import os

o_data = lorenz_generation(S, R, B)
h = 100


def worker(index):
    print(index)
    eps = epsilons[index]
    # directory_name = "NNPP_by_eps-" + str(index)
    # filepath = f"{os.getcwd()}/{directory_name}"
    # os.mkdir(filepath)
    centers = precalc_lorentz_by_eps(eps, index)
    pts = MEANING_PTS
    start_points = random.sample([i for i in range(10000, 15000)], pts)
    results, prefix_bads = get_metrics(o_data, h, start_points, centers)
    mts = [prefix_bads[25], prefix_bads[50], prefix_bads[75], prefix_bads[-1]]
    np.save("NNPP_similar" + str(index), np.array(mts))

def get_motifs(eps):
    mtf = GenerateAllMotifs(kmax, L, TRAIN_SIZE - 1)
    data = lorenz_generation(S, R + eps, B)[:TRAIN_SIZE]
    mtf_by_patterns = defaultdict(list)
    for pattern, all_motifs in mtf:
        vals = [values_by_motif(data, motif) for motif in all_motifs]
        for v in vals:
            mtf_by_patterns[pattern].append(v)
    return mtf_by_patterns

def dist(motif1, motif2):
    r = 0
    assert(len(motif1) == len(motif2))
    for i in range(len(motif1)):
        r += abs(motif1[i] - motif2[i])
    return r

import random

def get_similarity(mtf1, mtf2):
    u = 0
    c = 0
    ks = mtf1.keys()
    for pattern in ks:
        random.shuffle(mtf1[pattern])
        random.shuffle(mtf2[pattern])
    for pattern in ks:
        iters = 0
        for motif1 in mtf1[pattern]:
            iters += 1
            dst = 100
            for motif2 in mtf2[pattern]:
                dst = min(dist(motif1, motif2), dst)
            if dst < 0.04:
                c += 1
            else:
                u += 1
            if iters >= 3:
                break
    print(u, c)
    return c / (u + c) * 100.0

def compare_motifs(eps1, eps2):
    mtf1 = get_motifs(eps1)
    mtf2 = get_motifs(eps2)
    return (get_similarity(mtf1, mtf2) + get_similarity(mtf2, mtf1)) / 2.0

if __name__ == '__main__':
    #precalc_lorentz_by_eps(0.01, 0)
    epsilons = []
    eps_left = 0.01
    eps_right = 1
    for i in range(ROWS_CNT):
        epsilons.append(eps_left + (eps_right - eps_left) / (ROWS_CNT - 1.0) * i)

    comparisons = []
    for i in range(ROWS_CNT):
        comparisons.append(compare_motifs(epsilons[i], 0))
    print(comparisons)
    #compare_motifs(0.000001, 0, 1)
    #compare_motifs(0.0000000000000000001, 0, 2)
    #compare_motifs(0.001, 0.01, 3)
    #compare_motifs(0, 0, 4)
    # worker(0)
