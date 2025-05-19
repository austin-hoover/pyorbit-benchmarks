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


filename = "./outputs/history.csv"
history = pd.read_csv(filename)
history.head()


fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(history["s"], history["sig_x"] * 1000.0)
ax.plot(history["s"], history["sig_y"] * 1000.0)
ax.plot(history["s"], history["sig_z"] * 1000.0)
plt.savefig(os.path.join(output_dir, "fig_rms.png"))


fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(history["s"], history["emittance_x"] * 1.00e+06)
ax.plot(history["s"], history["emittance_y"] * 1.00e+06)
ax.plot(history["s"], history["emittance_z"])
plt.savefig(os.path.join(output_dir, "fig_emittance.png"))


# Plot phase space distribution
def load_bunch(filename: str) -> np.ndarray:
    x = np.loadtxt(filename, usecols=range(6), comments="%")
    x = x * 1000.0
    return x


# Load initial/final beams
bunch_filenames = [
    "outputs/bunch_00.dat",
    "outputs/bunch_01.dat",
]
bunches = [load_bunch(filename) for filename in bunch_filenames]


xmax = np.std(bunches[1], axis=0) * 3.0
limits = list(zip(-xmax, xmax))

fig, axs = plt.subplots(ncols=2, figsize=(5, 2.5), sharex=True, sharey=True)
for ax, x in zip(axs, bunches):
    axis = (0, 2)
    values, edges = np.histogramdd(x[:, axis], bins=45, range=[limits[k] for k in axis])
    ax.pcolormesh(edges[0], edges[1], values.T)
plt.savefig(os.path.join(output_dir, "fig_xy.png"))
