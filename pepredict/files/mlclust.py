import numpy as np
from .wishart_lib import Wishart
from itertools import groupby
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score, davies_bouldin_score


def GetClustersWishart(vectors, add_info = None, WISHART_R = 10, WISHART_U = 0.2):
    wishart = Wishart(WISHART_R, WISHART_U)
    labels = wishart.fit(vectors)

    sorted_by_cluster = sorted(range(len(labels)), key=lambda x: labels[x])
    clusters = []
    for wi, cluster in groupby(sorted_by_cluster, lambda x: labels[x]):
        if add_info is None:
            clusters += [[np.array(vectors[id]) for id in cluster]]
        else:
            clusters += [[[np.array(vectors[id]), add_info[id]] for id in cluster]]
            
    return clusters

def GetClustersDBSCAN_1D(vectors, eps=0.05, min_samples=3):
    vectors = vectors.reshape((-1, 1))
    X = StandardScaler().fit_transform(vectors)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    sorted_by_cluster = sorted(range(len(labels)), key=lambda x: labels[x])
    clusters = []
    noise = []
    for wi, cluster in groupby(sorted_by_cluster, lambda x: labels[x]):
        if wi != -1:
            clusters += [[vectors[id][0] for id in cluster]]
        else:
            noise += [vectors[id][0] for id in cluster]
    return [clusters, noise]

def GetFeatures(pts : np.array):
    """
    pts is array of floats in [0; 1]
    """
    clusters, noise = GetClustersDBSCAN_1D(pts)
    max_cluster_w = 0 if len(clusters) == 0 else len(max(clusters, key=lambda x: len(x)))

    return np.array([max_cluster_w / len(pts), max_cluster_w, len(pts), len(clusters), len(noise)])

def KNN(dataset, answers):
    clf = KNeighborsClassifier(3)
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(dataset, answers)
    return clf