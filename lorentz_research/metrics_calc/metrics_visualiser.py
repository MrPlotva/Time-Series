import numpy as np

for index in range(10):
    loaded = np.load('../nnpp_for_tenk/NNPP_by_epsForTenK-' + str(index) + '.npy', allow_pickle=True)
    print(loaded * 100, end='\n')

print("\n\n\n\n")

loaded = np.load('../nnpp_for_tenk/NNPP_simple.npy',allow_pickle=True)
print(loaded * 100)


print("\n\n\n\n")
loaded = np.load('../nnpp_for_tenk/NNPP_absolutely_similar0.npy', allow_pickle=True)
print(loaded * 100)