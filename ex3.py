import numpy as np
import matplotlib.pyplot as plt
import math

def mc_volume(dim, trials):
    n_inside = 0
    volume = exact_volume(dim)
    points = 2 * np.random.rand(trials, dim) - 1

    for pt in points:
        if np.linalg.norm(pt) <= 1:
            n_inside += 1

    apx_volume = 2**dim * n_inside / trials
    rel_error = np.abs(apx_volume - volume) / volume
    return [apx_volume, rel_error]

        
def exact_volume(dim):
    return np.pi**(dim/2) / math.gamma(dim/2 + 1)


# -------------- 3 dimensions --------------
results_3d = []
# sample_amounts = np.arange(1, 10001, 100)
sample_amounts = np.power(10, np.arange(0,7,0.5))
for n in sample_amounts:
    results_3d.append(mc_volume(3, int(n)))

results_3d = np.array(results_3d)
print(f"3-ball volume approximation ({int(sample_amounts[-1])} trials): {results_3d[-1,0]}")
fig_3d, ax_3d = plt.subplots()
ax_3d.plot(sample_amounts, results_3d[:,1])
ax_3d.set_xscale('log')
ax_3d.set_yscale('log')
ax_3d.set_title("Relative error for 3-ball volume")
ax_3d.set_xlabel("Number of samples")
ax_3d.set_ylabel("Relative error")

# -------------- 4 dimensions --------------
results_4d = []
# sample_amounts = np.arange(1, 10001, 100)
sample_amounts = np.power(10, np.arange(0,7,0.5))
for n in sample_amounts:
    results_4d.append(mc_volume(4, int(n)))

results_4d = np.array(results_4d)
print(f"4-ball volume approximation ({int(sample_amounts[-1])} trials): {results_4d[-1,0]}")
fig_4d, ax_4d = plt.subplots()
ax_4d.plot(sample_amounts, results_4d[:,1])
ax_4d.set_xscale('log')
ax_4d.set_yscale('log')
ax_4d.set_title("Relative error for 4-ball volume")
ax_4d.set_xlabel("Number of samples")
ax_4d.set_ylabel("Relative error")

plt.show()
