from .prediction import *
from matplotlib import pyplot as plt

def visualize_predictions(prediction : np.array, test_data : np.array, 
                          hor : int, xlabel=None, ylabel=None):
    plt.figure(figsize=(6, 4))
    plt.ylim(0, 1)
    plt.plot(prediction[-hor:], 'r', label='prediction')
    plt.plot(list(test_data[:hor]), 'b', label='test')
    plt.title("Visualization")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.show()

def HorizonPlot(data : np.array, start_point : int, end_point : int, 
                all_motifs : list, all_patterns : list, max_hor : int, daemons : list, 
                eps, size : int, max_dynamic_eps=None):
    if type(eps) is float:
        eps = [eps] * len(daemons)
    if max_dynamic_eps == None:
        max_dynamic_eps = eps
    fig, all_axes = plt.subplots(1, 3, figsize=(15, 5))
    for num in range(len(daemons)):
        results = np.zeros(max_hor)
        nonpred_sum = np.zeros(max_hor)
        accuracy = np.zeros(max_hor)
        cnt = 0
        for start in range(start_point, end_point, (end_point - start_point) // size):
            cnt += 1
            preds, nonpreds = PredictionTrajectory(data[:start], all_motifs, daemons[num], 
                                                   max_hor, all_patterns, eps[num], max_dynamic_eps[num])
            for hor in range(max_hor):
                accuracy[hor] += int(nonpreds[start + hor]) ^ int(abs(preds[start + hor] - data[start + hor]) < eps[num])

                if nonpreds[start + hor]:
                    nonpred_sum[hor] += 1
                else:
                    results[hor] += abs(preds[start + hor] - data[start + hor]) ** 2

        for id in range(len(results)):
            results[id] = (results[id] / (cnt - nonpred_sum[id]))**0.5
            nonpred_sum[id] /= cnt
            accuracy[id] /= cnt
        all_axes[0].plot(results, label="daemon " + str(num + 1))
        all_axes[1].plot(nonpred_sum * 100, label="daemon " + str(num + 1))
        all_axes[2].plot(accuracy * 100, label="daemon " + str(num + 1))

    all_axes[0].set_xlabel("Horizon", size=10)
    all_axes[0].set_ylabel("RMSE", size=10)
    all_axes[0].legend()
    all_axes[1].set_xlabel("Nonpred", size=10)
    all_axes[1].set_ylabel("% of nonpred", size=10)
    all_axes[1].set_ybound(0, 100)
    all_axes[1].legend()
    all_axes[2].set_xlabel("Nonpred accuracy", size=10)
    all_axes[2].set_ylabel("% of accuracy", size=10)
    all_axes[2].set_ybound(0, 100)
    all_axes[2].legend()
    plt.grid()
    plt.show()