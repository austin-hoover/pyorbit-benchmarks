import os
import math
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["axes.linewidth"] = 1.25
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["savefig.dpi"] = 300


path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)



# Scalar history
# --------------------------------------------------------------------------------------


# Phase space distribution
# --------------------------------------------------------------------------------------

def load_bunch(filename: str) -> np.ndarray:
    x = np.loadtxt(filename, usecols=range(6))
    x = x * 1000.0
    return x


bunch_filenames = [
    "outputs/beam_00.dat",
    "outputs/beam_01.dat",
]
bunches = [load_bunch(filename) for filename in bunch_filenames]


xmax = np.std(bunches[1], axis=0) * 3.0
limits = list(zip(-xmax, xmax))

dims = ["x", "px", "y", "py", "t", "pt"]

for axis in [(0, 1), (2, 3), (0, 2)]:
    fig, axs = plt.subplots(ncols=2, figsize=(5, 2.5), sharex=True, sharey=True)
    for ax, x in zip(axs, bunches):
        values, edges = np.histogramdd(x[:, axis], bins=45, range=[limits[k] for k in axis])
        ax.pcolormesh(edges[0], edges[1], values.T)

    filename = f"fig_dist_{dims[axis[0]]}{dims[axis[1]]}.png"
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename)
