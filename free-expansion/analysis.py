import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import psdist.plot as psv
import ultraplot as uplt


# Setup
# --------------------------------------------------------------------------------------

output_dir = "outputs/analysis"
os.makedirs(output_dir, exist_ok=True)

uplt.rc["axes.linewidth"] = 1.25
uplt.rc["cmap.discrete"] = False
uplt.rc["cmap.sequential"] = "viridis"
uplt.rc["figure.facecolor"] = "white"
uplt.rc["grid"] = False
uplt.rc["savefig.dpi"] = 300.0


# RMS beam size evolution
# --------------------------------------------------------------------------------------

histories = {}

history = pd.read_csv("./pyorbit/outputs/history.csv")
histories["pyorbit"] = history.copy()

history_ref = pd.read_csv("./impactx/diags/ref_particle.0.0", delimiter=" ")
history_rms = pd.read_csv("./impactx/diags/reduced_beam_characteristics.0.0", delimiter=" ")
history_rms["sig_z"] = history_rms["sig_t"] * history_ref["beta"]
history_rms["emittance_z"] = history_rms["emittance_t"]
histories["impactx"] = history_rms.copy()


fig, axs = uplt.subplots(ncols=3, figheight=1.75)
for ax, key in zip(axs, ["sig_x", "sig_y", "sig_z"]):
    ax.plot(histories["pyorbit"]["s"], histories["pyorbit"][key] * 1000.0, label="impactx", color="blacK", lw=2.0)
    ax.plot(histories["impactx"]["s"], histories["impactx"][key] * 1000.0, label="pyorbit", color="red")
axs.format(xlabel="Distance [m]", ylabel="[mm]")
axs[0].set_title(r"$\sqrt{\langle xx \rangle}$", fontsize="medium")
axs[1].set_title(r"$\sqrt{\langle yy \rangle}$", fontsize="medium")
axs[2].set_title(r"$\sqrt{\langle zz \rangle}$", fontsize="medium")
axs[2].legend(fontsize="medium", ncols=1, loc="right", framealpha=0.0)
plt.savefig(os.path.join(output_dir, "fig_rms_sizes.png"))


fig, axs = uplt.subplots(ncols=2, figheight=1.75)
for ax, key in zip(axs, ["emittance_x", "emittance_y"]):
    ax.plot(histories["pyorbit"]["s"], histories["pyorbit"][key] * 1.0e+06, label="impactx", color="blacK", lw=2.0)
    ax.plot(histories["impactx"]["s"], histories["impactx"][key] * 1.0e+06, label="pyorbit", color="red")
axs.format(xlabel="Distance [m]", ylabel="[mm mrad]")
axs[0].set_title(r"$\varepsilon_x$", fontsize="medium")
axs[1].set_title(r"$\varepsilon_y$", fontsize="medium")
axs[1].legend(fontsize="medium", ncols=1, loc="right", framealpha=0.0)
plt.savefig(os.path.join(output_dir, "fig_rms_emittances.png"))


# Phase space distribution
# --------------------------------------------------------------------------------------

particles = {}

particles["pyorbit"] = []

for filename in [
    "pyorbit/outputs/bunch_00.dat",
    "pyorbit/outputs/bunch_01.dat",
]:
    x = np.loadtxt(filename, usecols=range(6), comments="%")
    x = x * 1000.0
    particles["pyorbit"].append(x.copy())

particles["impactx"] = []

file = h5py.File("./impactx/diags/openPMD/monitor.h5", "r")
data = file["data"]
step_keys = list(data.keys())  # strings representing step number []"1", "102", ...]

for step_key in step_keys:
    x = [
        data[step_key]["particles"]["beam"]["position"]["x"],
        data[step_key]["particles"]["beam"]["momentum"]["x"],
        data[step_key]["particles"]["beam"]["position"]["y"],
        data[step_key]["particles"]["beam"]["momentum"]["y"],
        data[step_key]["particles"]["beam"]["position"]["t"],
        data[step_key]["particles"]["beam"]["momentum"]["t"],
    ]
    x = np.stack(x, axis=-1)
    x = x * 1000.0

    ## [to do] Convert longitudinal coordiantes to z and dE.  
    
    particles["impactx"].append(x.copy())


for axis in [(0, 1), (2, 3), (0, 2)]:
    bins = 64
    xmax = np.std(particles["pyorbit"][-1], axis=0) * 3.0
    limits = list(zip(-xmax, xmax))
    dims = ["x", "xp", "y", "yp", "z", "dE"]
    units = ["mm", "mrad", "mm", "mrad", "mm", "MeV"]
    labels = [f"{dim} [{unit}]" for dim, unit in zip(dims, units)]

    fig, axs = uplt.subplots(ncols=2, nrows=2, figheight=4.0)
    for j, key in enumerate(["pyorbit", "impactx"]):
        for i, x in enumerate(particles[key]):
            ax = axs[i, j]
            values, edges = np.histogramdd(x[:, axis], bins=bins, range=[limits[k] for k in axis])
            ax.pcolormesh(edges[0], edges[1], values.T, cmap="viridis")
    axs.format(
        xlabel=labels[axis[0]],
        ylabel=labels[axis[1]],
        toplabels=["PyORBIT", "ImpactX"],
        leftlabels=["IN", "OUT"],
    )
    filename = f"fig_dist_{dims[axis[0]]}_{dims[axis[1]]}.png"
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename)
    plt.close()

cmap = uplt.Colormap("Blues", left=0.1)

for index in range(2):
    for key in histories:
        grid = psv.CornerGrid(ndim=4, figwidth=5.0)
        grid.set_labels(labels)
        grid.set_limits(limits)
        grid.plot(particles[key][index], bins=64, limits=limits, cmap=cmap, diag_kws=dict(lw=1.3))
    
        filename = f"fig_corner_{key}_{i}.png"
        filename = os.path.join(output_dir, filename)
        plt.savefig(filename)
        plt.close()

