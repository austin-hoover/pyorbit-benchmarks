import os
import math
import pathlib

import numpy as np
import pandas as pd
import openpmd_api as io

import matplotlib.pyplot as plt


plt.rcParams["axes.linewidth"] = 1.25
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["savefig.dpi"] = 300


path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


filename = "diags/ref_particle.0.0"
history_ref = pd.read_csv(filename, delimiter=" ")
history_ref.head()

filename = "diags/reduced_beam_characteristics.0.0"
history_rms = pd.read_csv(filename, delimiter=" ")
history_rms.head()

history_rms["sig_z"] = history_rms["sig_t"] * history_ref["beta"]
history_rms["emittance_z"] = history_rms["emittance_t"]


fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(history_rms["s"], history_rms["sig_x"] * 1000.0)
ax.plot(history_rms["s"], history_rms["sig_y"] * 1000.0)
ax.plot(history_rms["s"], history_rms["sig_z"] * 1000.0)
plt.savefig(os.path.join(output_dir, "fig_rms.png"))

fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(history_rms["s"], history_rms["emittance_x"] * 1.00e+06)
ax.plot(history_rms["s"], history_rms["emittance_y"] * 1.00e+06)
ax.plot(history_rms["s"], history_rms["emittance_z"] * 1.00e+06)
plt.savefig(os.path.join(output_dir, "fig_emittance.png"))


# Load initial/final bunches
def beam_df_to_np(beam: pd.DataFrame) -> np.ndarray:
    columns = ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
    x = beam.loc[:, columns].values
    x = x * 1000.0
    return x 

series = io.Series("./diags/openPMD/monitor.h5", io.Access.read_only)
last_step = list(series.iterations)[-1]
initial_beam = series.iterations[1].particles["beam"].to_df()
final_beam = series.iterations[last_step].particles["beam"].to_df()

# Extract phase space coordinate arrays
bunches = [initial_beam, final_beam]
bunches = [beam_df_to_np(beam) for beam in bunches]

# Plot initial/final x-y distribution
xmax = np.std(bunches[1], axis=0) * 3.0
limits = list(zip(-xmax, xmax))

fig, axs = plt.subplots(ncols=2, figsize=(5, 2.5), sharex=True, sharey=True)
for ax, x in zip(axs, bunches):
    axis = (0, 2)
    values, edges = np.histogramdd(x[:, axis], bins=45, range=[limits[k] for k in axis])
    ax.pcolormesh(edges[0], edges[1], values.T)
plt.savefig(os.path.join(output_dir, "fig_xy.png"))


