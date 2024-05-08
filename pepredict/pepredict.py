import sys

from .files.utility import *
from .files.motifs import *
from .files.patterns import *
from .files.wishart_lib import Wishart
from .files.mlclust import *
from .files.daemon import * 
from .files.prediction import *
from .files.visualization import *

class TrajPrediction:
    def __init__(self, data_train, pattern_len=5, pattern_size=4, horizon=50, eps=0.05):
        """
        
        """
        self.all_patterns = MakePatterns(pattern_len, pattern_size)
        self.all_motifs = GetMotifs(data_train, self.all_patterns, horizon, WISHART_R=11, WISHART_U=0.2)
        self.max_pattern_len = pattern_len * pattern_size
        self.eps = eps
        self.horizon = horizon

    def init_daemon(self, mode="weighted", is_predictable_type="quantile", always_predict=False):
        """
        mode : simple, weighted, wishart_clustered, random
        is_predictable_type : 1) max_cluster, quantile - can be used after initialization
                              2) fitted_knn, fitted_dec_tree - requires fitting after initialization 
        """
        self.daemon = Daemon(mode, is_predictable_type, always_predict)

    def fit_daemon(self, valid_data):
        self.daemon.FitPredict(self, valid_data, self.max_pattern_len, 
                               self.all_motifs, self.all_patterns, 
                               self.horizon, self.eps)

    def predict(self, data):
        """
        Returns a prediction of the next "horizon" values and 
        "data" must have a size of pattern_len * pattern_size
        """
        pred, ispred = PredictionTrajectory(data, self.all_motifs, self.daemon, 
                                             self.horizon, self.all_patterns, self.eps, self.eps * 1.5)
        ispred = ispred == 0
        return [pred[-self.horizon:], ispred[-self.horizon:]]
    
    def plot_prediction(self, prediction, is_predictable, true_values):
        plt.figure(figsize=(6, 4))
        plt.ylim(0, 1)
        plt.scatter(x=np.arange(self.horizon)[is_predictable], y=prediction[is_predictable], color='r', label='prediction')
        plt.plot(true_values, 'b', label='true values')
        plt.title("Visualization")
        plt.legend(loc='best')
        plt.show()