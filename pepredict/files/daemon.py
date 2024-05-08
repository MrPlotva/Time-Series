from .utility import *
from .mlclust import *


class Daemon:
    def __init__(self, mode="simple", is_predictable_type="quantile", always_pred=False):
        """
        mode : simple, weighted, wishart_clustered, random
        is_predictable_type : 1) max_cluster, quantile - can be used after initialization
                              2) fitted_knn, fitted_dec_tree - requires fitting after initialization 
        """
        self.mode = mode
        self.always_pred = always_pred
        self.is_predictable_type = is_predictable_type

    def FindUPV(self, possible_pts : np.array, weights : np.array):
        if self.mode == "simple":
            return np.median(possible_pts)
        elif self.mode == "weighted":
            return WeightedMedian(possible_pts, weights)
        elif self.mode == "wishart_clustered":
            weights = np.ones(weights.shape)
            possible_pts.shape = list(possible_pts.shape) + [1]
            clusters = GetClustersWishart(possible_pts, weights, self.WISHART_R, self.WISHART_U)
            max_cluster_w = max(clusters, key=lambda x : sum([vc[1] for vc in x]))
            return np.median([vec_w[0][0] for vec_w in max_cluster_w])
        elif self.mode == "random":
            return np.random.uniform(0.33, 0.67)
        else:
            raise NameError("Bad mode")

    def FindUPVTraj(self, possible_traj : np.array, weights : np.array):
        traj_len = len(possible_traj[0])
        result = np.zeros(traj_len)
        for id in range(traj_len):
            result[id] = self.FindUPV(possible_traj[:, id], weights)

        return result
    
    def IsPredictable(self, possible_pts : np.array, weights : np.array):
        if self.always_pred:
            return True
        elif self.is_predictable_type == "max_cluster":
            return self.IsPredictableClusterBetter(possible_pts)
        elif self.is_predictable_type == "quantile":
            return self.IsPredictableQuantile(possible_pts)
        elif self.is_predictable_type == "fitted_knn":
            return self.clf.predict(GetFeatures(possible_pts).reshape(1, -1))[0]
        else:
            raise TypeError("No such is_predictable mode")
    
    def IsPredictableTraj(self, possible_traj : np.array, weights : np.array):
        traj_len = len(possible_traj[0])
        is_predictable = np.zeros(traj_len)
        for id in range(traj_len):
            is_predictable[id] = self.IsPredictable(possible_traj[:, id], weights)
        return is_predictable
    
    def Predict(self, possible_traj : np.array, weights : np.array):
        return [self.FindUPVTraj(possible_traj, weights), 
                self.IsPredictableTraj(possible_traj, weights)]

    def IsPredictableClusterBetter(self, possible_pts : np.array):
        possible_pts.shape = list(possible_pts.shape) + [1]

        clusters, noise = GetClustersDBSCAN_1D(possible_pts, 0.02, 3)

        if len(clusters) == 0:
            return 0
        max_cluster_w = max(clusters, key=lambda x : len(x))
        if len(max_cluster_w)  > 0.7 * sum([len(el) for el in clusters]):
            return 1
        else:
            return 0
        
    def IsPredictableQuantile(self, possible_pts : np.array):
        if np.quantile(possible_pts, 0.90) - np.quantile(possible_pts, 0.10) < 0.03:
            return 1
        else:
            return 0
    
    def FitPredict(self, data, start, all_motifs, patterns, hor, eps):
        end = len(data) - hor + 1
        answers = []
        dataset = []

        for pos in range(start, end):
            possible_traj, weights = PredictionTrajectory(data[:pos], all_motifs, self, 
                                                 hor, patterns, eps, return_mode="possible_traj_only")
            if len(possible_traj) == 0:
                continue
            for h in range(hor):
                possible_pts = possible_traj[:, h]
                prediction = self.FindUPV(possible_pts, weights)
                delta = abs(prediction - data[pos + h])

                answers.append(delta < eps)
                dataset.append(GetFeatures(possible_pts))
        
        self.clf = KNN(dataset, answers)


class IdealDeamon(object):
    def __init__(self, real_vals, eps=0.05):
        self.eps = eps
        self.real_vals=real_vals
        
    @property
    def label(self):
        return 'Ideal model of demon'

    def Predict(self, start_point, step, prediction):
        if not self.IsPredictable(start_point, step, prediction):
            return None
        return prediction
    
    def IsPredictable(self, start_point, step, prediction):
       return abs(prediction - self.real_vals[start_point + step]) <= self.eps