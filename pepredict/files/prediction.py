from .utility import *
from .daemon import *

def PredictionTrajectory(data, all_motifs, daemon : Daemon, hor: int, patterns, eps: float = 0.05,
                    max_dynamic_eps=None, return_mode="predict_only"):
    """ 
    Predict trajectory of length $hor$ starting from back of $data$
    return_mode : "predict_only", "possible_traj_only", "all"
    In "possible_traj_only" mode, daemon is ignored
    """
    if max_dynamic_eps == None:
        max_dynamic_eps = eps
    st = len(data)
    prediction = np.zeros(st + hor)
    nonpred = np.zeros(st + hor)
    prediction[:st] = data[:st]
    possible_traj = []
    weights = []

    for pattern in patterns:
        z_vec = np.array(GetElementsForPattern(prediction, pattern, st))
        for motif in all_motifs[pattern]:
            weight_cluster = motif[1]
            motif = motif[0]
            dist = np.linalg.norm(np.array(motif[:len(pattern)]) - z_vec)
            if dist < eps:
                weight_dist = (eps - dist) / eps
                weights.append(weight_dist * weight_cluster)
                possible_traj.append(motif[len(pattern):])
    possible_traj = np.array(possible_traj)    
    weights = np.array(weights)   
    
    if return_mode == "possible_traj_only":
        return [possible_traj, weights]
    if len(possible_traj):
        pred, is_pred = daemon.Predict(possible_traj, weights)
        prediction[st:] = pred
        nonpred[st:] = 1 - is_pred
    else:
        if eps < max_dynamic_eps:
            return PredictionTrajectory(data, all_motifs, daemon, hor, patterns, eps * 1.2,
                                        max_dynamic_eps, return_mode)
        prediction[st:] = 0.5
        nonpred[st:] = 1

    if return_mode == "all":
        return [prediction, nonpred, possible_traj, weights]
    elif return_mode == "predict_only":
        return [prediction, nonpred]
    raise RuntimeError("No such return mode")