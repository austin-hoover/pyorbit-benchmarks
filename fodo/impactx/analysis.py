import os
import numpy as np
import pandas as pd
import openpmd_api as io

import matplotlib.pyplot as plt


plt.rcParams["figure.constrained_layout.use"] = True


# Setup
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# Load scalar history
filename = "diags/ref_particle.0.0"
history_ref = pd.read_csv(filename, delimiter=" ")

filename = "diags/reduced_beam_characteristics.0.0"
history = pd.read_csv(filename, delimiter=" ")

history["sig_z"] = history["sig_t"] * history_ref["beta"]
history["emittance_z"] = history["emittance_t"]


# Plot rms beam sizes
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(history["s"], history["sig_x"] * 1000.0)
ax.plot(history["s"], history["sig_y"] * 1000.0)
ax.plot(history["s"], history["sig_z"] * 1000.0)
plt.savefig(os.path.join(output_dir, "fig_rms_sizes.png"), dpi=300)

# Plot rms beam sizes
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(history["s"], history["emittance_x"] * 1.00e06)
ax.plot(history["s"], history["emittance_y"] * 1.00e06)
ax.plot(history["s"], history["emittance_z"] * 1.00e06)
plt.savefig(os.path.join(output_dir, "fig_rms_emittances.png"), dpi=300)


# Load phase space coordinates
def beam_df_to_np(beam: pd.DataFrame) -> np.ndarray:
    columns = ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
    x = beam.loc[:, columns].values
    x = x * 1000.0
    return x


series = io.Series("./diags/openPMD/monitor.h5", io.Access.read_only)
last_step = list(series.iterations)[-1]
initial_beam = series.iterations[1].particles["beam"].to_df()
final_beam = series.iterations[last_step].particles["beam"].to_df()

bunches = [initial_beam, final_beam]
bunches = [beam_df_to_np(beam) for beam in bunches]

# Plot initial/final distributions
bins = 75
xmax = np.std(bunches[1], axis=0) * 5.0
limits = list(zip(-xmax, xmax))

dims = ["x", "px", "y", "py", "t", "pt"]
for axis in [(0, 1), (2, 3), (4, 5)]:
    fig, axs = plt.subplots(ncols=2, figsize=(5, 2.5), sharex=True, sharey=True)
    for ax, bunch in zip(axs, bunches):
        values, edges = np.histogramdd(bunch[:, axis], bins=bins, range=[limits[k] for k in axis])
        values = values / np.max(values)
        log_values = np.log10(values + 1.00e-08)
        ax.pcolormesh(edges[0], edges[1], log_values.T, vmax=0.0, vmin=-3.0)

    xlabel = dims[axis[0]]
    ylabel = dims[axis[1]]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    filename = f"fig_dist_{dims[axis[0]]}_{dims[axis[1]]}.png"
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300)
