import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True


# Load scalar history
filename = "./outputs/history.csv"
history = pd.read_csv(filename)
print(history)

# Plot rms sizes vs. distance
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(history["s"], history["sig_x"] * 1000.0, label="xrms")
ax.plot(history["s"], history["sig_y"] * 1000.0, label="yrms")
ax.plot(history["s"], history["sig_z"] * 1000.0, label="zrms")
plt.savefig(os.path.join("outputs", "fig_history_rms.png"), dpi=300)

# Plot rms emittances vs. distance
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(history["s"], history["emittance_x"] * 1.00e06)
ax.plot(history["s"], history["emittance_y"] * 1.00e06)
ax.plot(history["s"], history["emittance_z"])
plt.savefig(os.path.join("outputs", "fig_history_emittance.png"), dpi=300)


# Load phase space coordinates
def load_bunch(filename: str) -> np.ndarray:
    x = np.loadtxt(filename, usecols=range(6), comments="%")
    x = x * 1000.0
    return x


bunch_filenames = [
    "outputs/bunch_00.dat",
    "outputs/bunch_01.dat",
]
bunches = [load_bunch(filename) for filename in bunch_filenames]


# Plot initial/final x-x' distribution
bins = 75
xmax = np.std(bunches[1], axis=0) * 5.0
limits = list(zip(-xmax, xmax))

fig, axs = plt.subplots(ncols=2, figsize=(5, 2.5), sharex=True, sharey=True)
for ax, x in zip(axs, bunches):
    axis = (0, 1)
    values, edges = np.histogramdd(x[:, axis], bins=bins, range=[limits[k] for k in axis])
    values = values / np.max(values)
    log_values = np.log10(values + 1.00e-08)
    ax.pcolormesh(edges[0], edges[1], log_values.T, vmax=0.0, vmin=-3.0)
plt.savefig(os.path.join("outputs", "fig_bunch_x_xp.png"), dpi=300)
