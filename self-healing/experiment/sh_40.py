import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import make_interp_spline, BSpline
from scipy.special import gamma
from collections import defaultdict
from random import choice, random, randint
from itertools import product
from statistics import mean
from matplotlib import pyplot as plt
from itertools import groupby
from tabulate import tabulate
import copy
import math

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

def GenerateMotifsByPattern(pattern, t):
  # Returns motifs according to given pattern
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

def IteratePatterns(patterns, pattern, i, L, sum, Kmax):
  # Generate all patterns with sum <= Kmax and length = L - 1
  if i == L - 1:
    patterns.append(pattern.copy())
  else:
    for j in range(1, Kmax - sum + 1):
      pattern[i] = j
      IteratePatterns(patterns, pattern, i + 1, L, sum + j, Kmax)

# def GenPatterns(patterns, pattern, i, L, Kmax):
#   if i == L - 1:
#     patterns.append(pattern.copy())
#   else:
#     for j in range(1, Kmax + 1):
#       pattern[i] = j
#       GenPatterns(patterns, pattern, i + 1, L, Kmax)

def GenPatterns(L, x):
    elements = range(1, x+1)  # Создаем последовательность элементов от 1 до x
    sequences = product(elements, repeat=L)  # Генерируем все возможные комбинации длиной L
    return sequences


def GenerateAllMotifs(Kmax, L, t):
  # Returns map [pattern, [motifs...]]
  patterns = []
  pattern = []
  for i in range(L - 1):
    pattern.append(0)
  # IteratePatterns(patterns, pattern, 0, L, 0, Kmax)
  patterns = GenPatterns(L - 1, Kmax)
  # print(len(patterns))
  motifsByPatterns = []
  for p in patterns:
    motifs = GenerateMotifsByPattern(p, t)
    motifsByPatterns.append([p, motifs])
  return motifsByPatterns


# print(*GenPatterns(4, 10))


def center(data):
    n = data.shape[0]
    m = data.shape[1]
    summ = np.zeros(m)
    for i in range(n):
        summ += data[i]
    summ /= np.array([n for i in range(m)])
    return summ


def dist(x, y):
    diff = x - y
    return np.sqrt(sum(diff ** 2))


def math_norm(x):
    x = np.array([x])
    return np.sqrt((x @ x.T)[0, 0])


def cleaning(data, cluster):
    # cleans all zeros!!
    clean_data = []
    clean_cluster = []
    was = dict()
    for i in range(data.shape[0]):
        if cluster[i] == 0:
            continue
        if cluster[i] not in was:
            was[cluster[i]] = len(was)
        clean_data.append(data[i])
        clean_cluster.append(was[cluster[i]])

    return np.array(clean_data), clean_cluster


class MeasureIndexes:
    def __init__(self, data, cluster):
        data, cluster = cleaning(data, cluster)
        # data = data set, cluster[i] = cluster of i-th point
        self.data = data  # data set
        self.N = data.shape[0]  # number of object in data set
        self.center = center(data)  # center of data set
        self.cluster = cluster  # cluster[i] = cluster of i-th point
        self.P = 0  # ???
        self.NC = max(cluster) + 1  # cnt clusters

        self.C = []  # C[i] - i-th cluster
        for i in range(self.NC):
            self.C.append([])
        for i in range(self.N):
            self.C[cluster[i]].append(data[i])
        for i in range(self.NC):
            self.C[i] = np.array(self.C[i])

        self.n = [0] * self.NC  # n[i] - len of i-th cluster
        for i in range(self.NC):
            self.n[i] = len(self.C[i])

        self.c = [0] * self.NC  # c[i] - center of i-th cluster
        for i in range(self.NC):
            self.c[i] = center(self.C[i])

    def RS(self):
        s1 = 0
        s2 = 0
        for x in self.data:
            s1 += math_norm(x - self.center) ** 2

        for i in range(self.NC):
            for x in self.C[i]:
                s2 += math_norm(x - self.c[i]) ** 2

        return (s1 - s2) / s1

    def G(self):
        s = 0
        for i in range(self.N):
            for j in range(self.N):
                x = self.data[i]
                y = self.data[j]
                ci = self.cluster[i]
                cj = self.cluster[j]
                s += dist(x, y) * dist(self.c[ci], self.c[cj])

        return (2 * s) / (self.N * (self.N - 1))

    def CH(self):
        s1 = 0
        for i in range(self.NC):
            s1 += self.n[i] * (dist(self.c[i], self.center) ** 2)
        s1 /= self.NC - 1

        s2 = 0
        for i in range(self.NC):
            for x in self.C[i]:
                s2 += dist(x, self.c[i]) ** 2
        s2 /= self.N - self.NC

        return s1 / s2

    def D(self):
        max_d = 0
        for k in range(self.NC):
            for x, y in product(self.C[k], self.C[k]):
                max_d = max(max_d, dist(x, y))

        min_d = 10 ** 10
        for i in range(self.NC):
            for j in range(self.NC):
                if i == j:
                    continue
                for x, y in product(self.C[i], self.C[j]):
                    min_d = min(min_d, dist(x, y))

        return min_d / max_d

    def S(self):
        s = 0
        for i in range(self.NC):
            cur = 0
            for x in self.C[i]:
                ax = 0
                for y in self.C[i]:
                    ax += dist(x, y)
                if self.n[i] > 1:
                    ax /= self.n[i] - 1

                bx = 10 ** 20
                for j in range(self.NC):
                    if i == j:
                        continue
                    res = 0
                    for y in self.C[j]:
                        res += dist(x, y)
                    res /= self.n[j]
                    bx = min(bx, res)
                cur += (bx - ax) / max(bx, ax)
            s += cur / self.n[i]

        return s / self.NC

    def DB(self):
        res = 0
        for i in range(self.NC):
            mx = 0
            for j in range(self.NC):
                if j == i:
                    continue
                s1 = 0
                for x in self.C[i]:
                    s1 += dist(x, self.c[i])
                s1 /= self.n[i]

                s2 = 0
                for x in self.C[j]:
                    s2 += dist(x, self.c[j])
                s2 /= self.n[j]

                s = (s1 + s2) / dist(self.c[i], self.c[j])
                mx = max(mx, s)
            res += mx
        return res / self.NC

    def XB(self):
        s1 = 0
        for i in range(self.NC):
            for x in self.C[i]:
                s1 += dist(x, self.c[i]) ** 2
        s2 = 10 ** 20
        for i in range(self.NC):
            for j in range(self.NC):
                if i == j:
                    continue
                s2 = min(s2, dist(self.c[i], self.c[j]) ** 2)

        s2 *= self.N

        return s1 / s2

    def Calculate(self):
        res = dict()
        # res["RS"] = self.RS()
        # res["G"] = self.G()
        res["CH"] = self.CH()
        res["D"] = self.D()
        res["S"] = self.S()
        res["DB"] = self.DB()
        res["XB"] = self.XB()

        return res


index_list = ["CH", "D", "S", "DB", "XB"]


class Tester:
    def __init__(self):
        self.results = []

    def add_test(self, data, cluster):
        calculator = MeasureIndexes(data, cluster)
        res = calculator.Calculate()
        res["id"] = len(self.results) + 1
        self.results.append(res)

    def display(self):
        col_names = ["id"] + index_list
        data = []

        for i in range(len(self.results)):
            row = [round(self.results[i][key], 3) for key in col_names]
            data.append(row)

        for i in range(1, len(data[0])):
            max_val = data[0][i]
            min_val = data[0][i]
            for j in range(len(data)):
                max_val = max(max_val, data[j][i])
                min_val = min(min_val, data[j][i])

            target_val = max_val
            if col_names[i] == "DB" or col_names[i] == "XB":
                target_val = min_val

            for j in range(len(data)):
                if target_val == data[j][i]:
                    data[j][i] = '\033[36m' + str(data[j][i]) + '\033[0m'

        print(tabulate(data, headers=col_names))

class Generator:
    def __init__(self, rad_frac_=2.0, lim_=40, cnt_=200, clusters_cnt_=5):
        # generating 2D points for better visualisation

        self.rad_frac = rad_frac_
        # self.rad_frac = N / rad, where N is side of square inside which we generate points
        # so the less self.rad is, the bigger is rad, so the harder for the algorithm it becomes (clusters become closer to each other)

        self.lim = lim_
        # side of a square inside which we generate points

        self.cnt = cnt_
        # ~ number of points

        self.clusters_cnt = clusters_cnt_
        return

    # useful functions
    def rand_point_in_disk(self, c, r):
        theta = 2 * math.pi * random()
        s = r * random()
        x, y = c
        return x + s * math.cos(theta), y + s * math.sin(theta)

    def rand_point_on_disk_side(self, c, r):
        theta = 2 * math.pi * random()
        x, y = c
        return x + r * math.cos(theta), y + r * math.sin(theta)

    def rand_cluster(self, n, c, r):
        # return n random points inside disk with center = c and radius = r
        x, y = c
        points = []
        for i in range(int(n)):
            points.append(self.rand_point_in_disk(c, r))
        return points

    def rand_clusters(self, k, n, r, lim):
        # return k clusters of n points each, centers of clusters are (x, y), where 0 <= x <= lim, 0 <= y <= lim
        clusters = []
        for _ in range(int(k)):
            x = lim * random()
            y = lim * random()
            clusters.extend(self.rand_cluster(n, (x, y), r))
        return clusters

    def dist_2d(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def rand_subcluster(self, n, c1, c2):
        # generate two subclusters n points each as points inside two disks with one point of intersection
        r = self.dist_2d(c1, c2) / 2.3
        cluster1 = self.rand_cluster(n, c1, r)
        cluster2 = self.rand_cluster(n, c2, r)
        return cluster1, cluster2

    def rand_subclusters(self, k, n, r, lim):
        # generate k pairs of 2 subclusters n points each
        clusters = []
        for _ in range(k):
            x1 = lim * random()
            y1 = lim * random()
            x2, y2 = self.rand_point_on_disk_side((x1, y1), r * 2.0)
            cluster1, cluster2 = self.rand_subcluster(n, (x1, y1), (x2, y2))
            clusters.extend(cluster1)
            clusters.extend(cluster2)
        return clusters

    # generating functions
    def generate_with_noise(self, noise_percentage=5.0, ret_points=False):
        num_of_clusters = self.clusters_cnt
        n = self.cnt / num_of_clusters
        r = self.lim / (self.rad_frac * num_of_clusters)
        clusters = self.rand_clusters(num_of_clusters, n, r, self.lim)

        noise_cnt = int(len(clusters) / 100.0 * noise_percentage)
        noise_points = []
        for _ in range(noise_cnt):
            curp = self.rand_point_in_disk((self.lim / 2.0, self.lim / 2.0), self.lim / 2.0)
            noise_points.append(curp)
            clusters.append(curp)
        if ret_points:
            return clusters, noise_points
        return clusters

    def generate_with_density(self, density_multiplier=2.0):
        # assume one of cluster has two times more points than others
        num_of_clusters = self.clusters_cnt
        n = self.cnt / num_of_clusters
        r = self.lim / (self.rad_frac * num_of_clusters)
        clusters = self.rand_clusters(num_of_clusters - 1, n, r, self.lim)
        x = self.lim * random()
        y = self.lim * random()
        clusters.extend(self.rand_cluster(n * density_multiplier, (x, y), r))
        return clusters

    def generate_with_subclusters(self):
        num_of_pairs_of_clusters = self.clusters_cnt
        n = self.cnt / (num_of_pairs_of_clusters * 2)
        r = self.lim / (self.rad_frac * num_of_pairs_of_clusters * 2)
        clusters = self.rand_subclusters(num_of_pairs_of_clusters, n, r, self.lim)
        return clusters

    def generate_with_skew(self, multiplier=2.0):
        # assume one of cluster has two times more radius than others
        num_of_clusters = self.clusters_cnt
        n = self.cnt / num_of_clusters
        r = self.lim / (self.rad_frac * num_of_clusters)
        clusters = self.rand_clusters(num_of_clusters - 1, n, r, self.lim)
        x = self.lim * random()
        y = self.lim * random()
        clusters.extend(self.rand_cluster(n, (x, y), r * multiplier))
        return clusters


class Lorentz:
    def __init__(self, s = 10, r = 28, b = 8/3):
        self.s = s
        self.r = r
        self.b = b

    #Differential equations of a Lorenz System
    def X(self, x, y, s):
        return s * (y - x)

    def Y(self, x, y, z, r):
        return (-x) * z + r * x - y

    def Z(self, x, y, z, b):
        return x * y - b * z

    #RK4 for the differential equations
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

        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt * (1/6)
        y += (l_1 + 2 * l_2 + 2 * l_3 + l_4) * dt * (1/6)
        z += (m_1 + 2 * m_2 + 2 * m_3 + m_4) * dt * (1/6)

        return (x, y, z)

    def generate(self, dt, steps):
        #Initial values and Parameters
        x_0, y_0, z_0 = 1, 1, 1

        #RK4 iteration
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

lorents = Lorentz()
data, _, _ = Lorentz().generate(0.1, 100000)
data = data[250:]
data = (data - data.min()) / (data.max() - data.min())


TRAIN_SIZE = 10000
TEST_SIZE = 1000
size = len(data)
train_data = data[:TRAIN_SIZE]
test_data = data[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]


# def get_val_for_pattern_and_pos(data: np.array, pattern: list, pos: int, bad):
#     val = []
#     sum = 0
#     for i in range(len(pattern) - 1, -1, -1):
#       sum += pattern[i]
#       val.append(data[pos - sum])
#       if bad[pos - sum]:
#           return np.array([])
#     val = val[::-1]
#     return np.array(val)

def get_val_for_motifs(data: np.array, motif: list, bad):
    val = []
    for i in motif:
        val.append(data[i])
        if bad[i]:
            return np.array([])
    return np.array(val)


def mode(a):
    return max(set(a), key=a.count)

def is_predictable(a):
    return abs(mean(a) - mode(a)) / mode(a) <= 0.1

def get(a):
    return abs(mean(a) - mode(a)) / mode(a)


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


def get_all_centers_for_patterns(data, L: int = 3, kmax: int = 10):
    centers = dict()
    t = len(data)

    bad = np.array([0 for i in range(t)])
    for pattern, all_motifs  in GenerateAllMotifs(kmax, L, t - 1):

        all_val = [get_val_for_motifs(data, motif, bad) for motif in all_motifs]
        centers[pattern] = get_centers(all_val)

        if len(centers) % 200 == 0:
            print(len(centers))

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

# LOADING OF CENTERS DATA
# CHANGE IF NECESSARY TO REGENERATE
    
class Daemon:
    def __init__(self, mode="simple", is_pred=True):
        self.mode = mode
        self.is_pred = is_pred

    def mean_d(self, preds):
        sum_weight = sum(map(lambda x: x[1], preds))
        s = sum(map(lambda x: x[0] * x[1], preds))
        return s / sum_weight
    
    def mean_q(self, preds):
        sum_weight = sum(map(lambda x: x[2], preds))
        s = sum(map(lambda x: x[0] * x[2], preds))
        return s / sum_weight
    
    def mean_d_q(self, preds):
        sum_weight = sum(map(lambda x: x[1] * x[2], preds))
        s = sum(map(lambda x: x[0] * x[1] * x[2], preds))
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
        wishart = Wishart(WISHART_R, WISHART_U)
        labels = (wishart.fit([[i[0]] for i in possible_values]))
        # print(labels) 
        if len(set(labels)) > 3:
            return False
        # print(possible_values)
        # if max(np.array(possible_values)[:,0]) - min(np.array(possible_values)[:,0]) > 0.1:
        #     return False
        return True
    
class Daemon3ptsMinMaxGrowth:
    def __init__(self, mode="simple", is_pred=True):
        self.mode = mode
        self.is_pred = is_pred
        self.cl = []

    def mean_d(self, preds):
        sum_weight = sum(map(lambda x: x[1], preds))
        s = sum(map(lambda x: x[0] * x[1], preds))
        return s / sum_weight
    
    def mean_q(self, preds):
        sum_weight = sum(map(lambda x: x[2], preds))
        s = sum(map(lambda x: x[0] * x[2], preds))
        return s / sum_weight
    
    def mean_d_q(self, preds):
        sum_weight = sum(map(lambda x: x[1] * x[2], preds))
        s = sum(map(lambda x: x[0] * x[1] * x[2], preds))
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
        # wishart = Wishart(WISHART_R, WISHART_U)
        vals = [i[0] for i in possible_values]
        # labels = set(wishart.fit([[i[0]] for i in possible_values]))
        # cnt = len(labels) - (0 in labels)
        self.cl.append(max(vals) - min(vals))
        # print(self.cl)
        # print(max(vals) - min(vals))
        # return (cnt <= 5)
        # return max(vals) - min(vals) < 0.15
        if len(self.cl) >= 3:
            return not (self.cl[-3] < self.cl[-2] < self.cl[-1])
        return True

class IdealDeamon(object):
    def __init__(self, real_vals, eps=0.05, mode='simple'):
        self.eps = eps
        self.mode = mode 
        self.real_vals=real_vals
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
    

from tqdm import tqdm
def get_val_for_pattern_and_pos(data: np.array, pattern: list, pos: int, bad):
    val = []
    def get_predictions(self):
       return self.predictions
    sum = 0
    for i in range(len(pattern) - 1, -1, -1):
      sum += pattern[i]
      val.append(data[pos - sum])
      if bad[pos - sum]:
          return np.array([])
    val = val[::-1]
    return np.array(val)


def get_val_for_motifs(data: np.array, motif: list, bad):
    val = []
    for i in motif:
        val.append(data[i])
        if bad[i]:
            return np.array([])
    return np.array(val)

def base_prediction(data, daemon: Daemon, ideal_daemon=None, h: int=1, L: int = 3, kmax: int = 10, eps: float = 0.1, QVALUE = 0.99,
                    return_possible_values=False):
    t = len(data)
    prediction = np.zeros(shape=(t + h, 2))
    bad = np.array([0 for i in range(t + h)])
    for i in range(t):
        prediction[i][0] = data[i]
        prediction[i][1] = 1
    possible_values = [[] for i in range(h)]

    steps = 0
    for i in range(h):
        print(i)
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
              dist = np.linalg.norm(c[:-1] - val_for_pattern[:-1])
              if dist < eps:
                  weight_d = (eps - dist) / eps;
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
        else:
            bad[t + i] = 1
            prediction[t + i][0] = 0

        prediction[t + i][1] = np.mean(list(map(lambda x: x[2], possible_values[i])))
        # print(prediction[t + i])
    print(steps)
    if return_possible_values:
        return [prediction, bad, possible_values]
    return [prediction, bad]


def base_prediction_ideal(t, data, delta, daemon: Daemon, ideal_daemon: IdealDeamon, h: int, L: int = 3, kmax: int = 10, eps: float = 0.1, QVALUE = 0.99,
                    return_possible_values=False):
    # t = len(data)

    prediction = np.zeros(shape=(t + h + delta, 2))
    bad = np.array([0 for i in range(t + h + delta)])
    for i in range(t):
        prediction[i][0] = data[i]
        prediction[i][1] = 1
    for i in range(t + h, t + h + delta):
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
              dist = np.linalg.norm(c[:-1] - val_for_pattern[:-1])
              if dist < eps:
                  weight_d = (eps - dist) / eps;
                  weight_q = np.mean(val_q) * QVALUE
                  possible_values[i].append([c[-1], weight_d, weight_q])
                #   possible_values[i].append([c[-1], weight_d, weight_q])
                
        if len(possible_values[i]):
            pred = ideal_daemon.predict(0, i, daemon.predict(possible_values[i]))
            # print(test_data[i], pred)
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
    print(steps)
    return [prediction, bad, possible_values]


def get_val_for_pattern_and_pos(data: np.array, pattern: list, pos: int, bad, pred_pos: int = -1):
    sum = 0

    if pred_pos == -1:
        pred_pos = pos
        
    if not bad[pos] or pred_pos == pos:
        val = [data[pos]]
    else:
        return np.array([])
    
    for i in range(len(pattern) - 1, -1, -1):
        sum += pattern[i]
        val.append(data[pos - sum])
        if bad[pos - sum] and pred_pos != pos:
            return np.array([])
    val = val[::-1]
    return np.array(val)



def iterate_prediction(t, prediction, bad, daemon: Daemon, ideal_daemon: IdealDeamon, h: int, L: int = 3, kmax: int = 10, eps: float = 0.1, QVALUE = 0.99,
                    return_possible_values=False, iterations: int = 1):
    steps = 0
    for it in range(iterations):
        print("ITERATION")
        possible_values = [[] for i in range(h)]
        for i in range(h):
            # print(bad[t + i])
            # if not bad[t + i]:
            #     continue
            for pattern in GenPatterns(L - 1, kmax):
                for pos in range(L):
                    end_index = t + i + sum(pattern[pos:])
                    if len(prediction) <= end_index:
                        continue
                    val_for_pattern_with_q = get_val_for_pattern_and_pos(prediction, pattern, 
                                                                         pos=end_index, bad=bad, pred_pos=t + i)
                    
                    # print(t + i)
                    # print(pattern)
                    # print(end_index)

                    if len(val_for_pattern_with_q) == 0:
                        continue
                    val_for_pattern = val_for_pattern_with_q[:, 0]
                    val_q = val_for_pattern_with_q[:, 1]


                    for c in centers[pattern]:
                        if len(c) == 0:
                            continue
                        steps += 1
                        dist = np.linalg.norm(np.delete(c, pos) - np.delete(val_for_pattern, pos))
                        if dist < eps:
                            # print(c, val_for_pattern, pos)
                            weight_d = (eps - dist) / eps;
                            weight_q = np.mean(val_q) * QVALUE
                            possible_values[i].append([c[pos], weight_d, weight_q])
                    #   possible_values[i].append([c[-1], weight_d, weight_q])
                    
            # if bad[t + i]:
            #     print(t + i)
            print("possible ", len(possible_values[i]))
            if len(possible_values[i]):
                pred = ideal_daemon.predict(0, i, daemon.predict(possible_values[i]))
                # print(ideal_daemon.real_vals[i], daemon.predict(possible_values[i]))
                print(prediction[t + i], bad[t + i], pred, t + i)
                if pred is not None:
                    prediction[t + i][0] = pred
                    bad[t + i] = 0
                else:
                    prediction[t + i][0] = 0
                    bad[t + i] = 1
                prediction[t + i][1] = np.mean(list(map(lambda x: x[2], possible_values[i])))

        # print(prediction[t + i])
    print(steps)
    return [prediction, bad, possible_values]


def calc_metrics(max_hor, num, func, start_points):
    results_ideal = [0] * max_hor
    num_bad_ideal = [0] * max_hor
    start_point = TRAIN_SIZE
    end_point = TRAIN_SIZE + num
    # print(data[:start_point])
    for start in start_points:
        # pred = find_prediction(start, max_hor, motifs, MakePatterns(pattern_type='bound_max'), 0.01)
        # answers = calc_result(pred)
        answers, bad, pos = func(list(data[:start]), Daemon(mode="simple_d_q", is_pred=False), IdealDeamon(data[start:], 0.05), max_hor, L=4, eps=0.009, return_possible_values=True)
        answers = answers[-max_hor:]
        bad = bad[-max_hor:]
        for hor in range(1, max_hor):
            num_bad_ideal[hor - 1] += bad[hor - 1]
            if bad[hor - 1]:
                continue
            results_ideal[hor - 1] += (answers[hor - 1][0] - data[start + hor - 1]) ** 2

    for i in range(len(results_ideal)):
        results_ideal[i] = (results_ideal[i]**0.5) / (end_point - start_point)
        num_bad_ideal[i] = (num_bad_ideal[i]) / (end_point - start_point)

    return results_ideal, num_bad_ideal


def calc_self_healing_metrics(max_hor, base_func, iterate_func, start_points, prefix):
    global ITERATIONS
    results_ideal = [[0 for i in range(len(start_points))] for j in range(ITERATIONS + 1)]
    num_bad_ideal = [[0 for i in range(len(start_points))] for j in range(ITERATIONS + 1)]
    # answers = [[] for j in range(ITERATIONS + 1)]
    # bad = [[] for j in range(ITERATIONS + 1)]
    # pos = [[] for j in range(ITERATIONS + 1)]
    answers = dict()
    bad = dict()
    pos = dict()
    num = len(start_points)
    for iteration in range(ITERATIONS + 1):
        if not iteration:
            calc_func = base_func
        else:
            calc_func = iterate_func
        for start in start_points:
            # pred = find_prediction(start, max_hor, motifs, MakePatterns(pattern_type='bound_max'), 0.01)
            # answers = calc_result(pred)
            if not iteration:
                answers[start], bad[start], pos[start] = base_prediction_ideal(start, list(data), 100, Daemon(mode="simple_d_q", is_pred=False), IdealDeamon(data[start:], 0.05), max_hor, L=4, eps=0.009, return_possible_values=True)
            else:
                answers[start], bad[start], pos[start] = iterate_prediction(start, answers[start], bad[start], Daemon(mode="simple_d_q", is_pred=False), IdealDeamon(data[start:], 0.05), max_hor, L=4, eps=0.009, return_possible_values=True)
            # answers[iteration] = answers[iteration][-max_hor:]
            # bad[iteration] = bad[iteration][-max_hor:]
            save(answers[start], f"{prefix}/start={start}_answer_{max_hor}.npy")
            save(bad[start], f"{prefix}/start={start}_bad_{max_hor}.npy")
            save(pos[start], f"{prefix}/start={start}_pos_{max_hor}.npy")
            print(start, answers[start][start:start + max_hor])
            # save(num_bad_ideal[iteration], f"{prefix}/bad_iter={iteration}_{max_hor}.npy")
            # print(answers[start][::,0][-max_hor:])
            # print(bad)
            index = start_points.index(start)
            num_bad_ideal[iteration][index] = sum(bad[start][start + 1:start + max_hor + 1])
            for hor in range(1, max_hor + 1):
                # if num_bad_ideal[]
                if bad[start][start + hor]:
                    continue
                results_ideal[iteration][index] += (answers[start][start + hor][0] - data[start + hor]) ** 2

        save(results_ideal[iteration], f"{prefix}/res_iter={iteration}_{max_hor}.npy")
        save(num_bad_ideal[iteration], f"{prefix}/bad_iter={iteration}_{max_hor}.npy")
    # for i in answers:
    #     print([j[0] for j in answers[i]])
    # for i in bad:
    #     print([j[0] for j in answers[i]])
    return results_ideal, num_bad_ideal


def save(arr, file):
    np.save(file, np.array(arr, dtype=object))


def save_iterations(prediction_i, bad_i):
    global ITERATIONS, k
    iter_predictions = [prediction_i]
    iter_bad = [bad_i]
    for iter in range(ITERATIONS):
        next_pred, next_bad, next_possible = iterate_prediction(iter_predictions[-1], iter_bad[-1], 1, train_data, Daemon(mode="simple_d_q", is_pred=False), IdealDeamon(test_data, 0.1), h, L=4, eps=0.009, return_possible_values=True)

        save(next_pred, f"experiment_k={k}/pred_iter={iter + 1}.npy")
        save(next_bad, f"experiment_k={k}/bad_iter={iter + 1}.npy")
        iter_predictions.append(next_pred)
        iter_bad.append(next_bad)

def save_sh_metrics(max_hor, base_func, iterate_func, start_points, prefix):
    global ITERATIONS
    save(start_points, f"{prefix}/start_points_{max_hor}.npy")
    res, bad = calc_self_healing_metrics(max_hor, base_func, iterate_func, start_points, prefix)
    # for iter in range(ITERATIONS):
    save(res, f"{prefix}/res_{max_hor}.npy")
    save(bad, f"{prefix}/bad_{max_hor}.npy")


data_c = np.load('centers.npy', allow_pickle=True)
centers = dict()
for key, val in data_c:
    centers[key] = val

    
ITERATIONS=100
HORIZONT = 100
DIRECTORY = "final_multithread"
NUM_POINTS = 100
AMOUNT_FOR_THREAD = 2
p_for_thread = [2 for i in range(20)] + [3 for i in range(20)]
# ITERATIONS=2
# HORIZONT = 10
# DIRECTORY = "final_multithread"
# NUM_POINTS = 3
# AMOUNT_FOR_THREAD = 2
# p_for_thread = [1 for i in range(3)]
# p_for_thread = [2 for i in range(2)] + [3 for i in range(2)]
print(sum(p_for_thread))

import random
start_points = random.sample([i for i in range(10000, 15000)], NUM_POINTS)

print(start_points)

import os
from multiprocessing import Pool
import itertools
 
def worker(args):
    s, ind = args
    os.mkdir(f"{DIRECTORY}/t{ind}")
    print(s, ind)
    save_sh_metrics(HORIZONT, base_prediction_ideal, iterate_prediction, 
                                                                  s, 
                                                                  f"{DIRECTORY}/t{ind}")

if __name__ == '__main__':
  start_pos = 0
  elem = []
  for i in range(len(p_for_thread)):
      elem.append((start_points[start_pos: start_pos + p_for_thread[i]], i))
      start_pos += p_for_thread[i]

  print(elem)
  pool = Pool(processes=40)
  pool.map(worker, elem)