import numpy as np


def load_centers(filename):
    centers = dict()
    loaded = np.load(filename, allow_pickle=True)
    for k, v in loaded:
        centers[k] = v
    return centers


c_simple = load_centers('../centers_by_eps/centers_lorenz-for10k0.npy')
c_noise = load_centers('../centers_by_eps/centers_lorenz-for10k1.npy')

it = 0
for k in c_simple.keys():
    v = c_simple[k]
    it += len(v)

it2 = 0
for k in c_noise.keys():
    v = c_noise[k]
    it2 += len(v)

print(it, '\n', it2)