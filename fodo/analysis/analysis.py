import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
import ultraplot as uplt
from omegaconf import DictConfig
from omegaconf import OmegaConf


uplt.rc["axes.linewidth"] = 1.25
uplt.rc["cmap.discrete"] = False
uplt.rc["cmap.sequential"] = "viridis"
uplt.rc["figure.facecolor"] = "white"
uplt.rc["grid"] = False
uplt.rc["savefig.dpi"] = 300.0


# Setup
# --------------------------------------------------------------------------------------

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

cfg = OmegaConf.load("../config.yaml")
print(cfg)


# Scalar history
# --------------------------------------------------------------------------------------

# Collect data
histories = {}

history = pd.read_csv("../pyorbit/outputs/history.csv")
histories["pyorbit"] = history.copy()

history_ref = pd.read_csv("../impactx/diags/ref_particle.0.0", delimiter=" ")
history_rms = pd.read_csv("../impactx/diags/reduced_beam_characteristics.0.0", delimiter=" ")
history_rms["sig_z"] = history_rms["sig_t"] * history_ref["beta"]
history_rms["emittance_z"] = history_rms["emittance_t"]
histories["impactx"] = history_rms.copy()

# Plot rms size vs. distance
fig, axs = uplt.subplots(ncols=3, figheight=1.75)
for ax, key in zip(axs, ["sig_x", "sig_y", "sig_z"]):
    ax.plot(
        histories["pyorbit"]["s"],
        histories["pyorbit"][key] * 1000.0,
        label="impactx",
        color="blacK",
        lw=2.0,
    )
    ax.plot(
        histories["impactx"]["s"], histories["impactx"][key] * 1000.0, label="pyorbit", color="red"
    )
axs.format(xlabel="Distance [m]", ylabel="[mm]")
axs[0].set_title(r"$\sqrt{\langle xx \rangle}$", fontsize="medium")
axs[1].set_title(r"$\sqrt{\langle yy \rangle}$", fontsize="medium")
axs[2].set_title(r"$\sqrt{\langle zz \rangle}$", fontsize="medium")
axs[2].legend(fontsize="medium", ncols=1, loc="right", framealpha=0.0)

filename = "fig_rms_sizes.png"
filename = os.path.join(output_dir, filename)
print(filename)
plt.savefig(filename)


# Plot rms emittance vs. distance
fig, axs = uplt.subplots(ncols=2, figheight=1.75)
for ax, key in zip(axs, ["emittance_x", "emittance_y"]):
    ax.plot(
        histories["pyorbit"]["s"],
        histories["pyorbit"][key] * 1.0e06,
        label="impactx",
        color="blacK",
        lw=2.0,
    )
    ax.plot(
        histories["impactx"]["s"], histories["impactx"][key] * 1.0e06, label="pyorbit", color="red"
    )
axs.format(xlabel="Distance [m]", ylabel="[mm mrad]")
axs[0].set_title(r"$\varepsilon_x$", fontsize="medium")
axs[1].set_title(r"$\varepsilon_y$", fontsize="medium")
axs[1].legend(fontsize="medium", ncols=1, loc="right", framealpha=0.0)

filename = "fig_rms_emittances.png"
filename = os.path.join(output_dir, filename)
print(filename)
plt.savefig(filename)


# Phase space distribution
# --------------------------------------------------------------------------------------


def pyorbit_to_impactx(bunch: np.ndarray, kin_energy: float, mass: float) -> np.ndarray:
    """Convert PyORBIT to ImpactX phase space coordinates.

    Args:
        kin_energy: kinetic energy [MeV]
        mass: rest mass in [MeV / c^2]
    """
    gamma0 = (mass + kin_energy) / mass
    beta0 = np.sqrt(gamma0**2 - 1.0) / gamma0

    # GeV to MeV
    bunch[:, 5] *= 1.00e03

    # x -> x
    # y -> y
    # z -> ct
    dx = bunch[:, 0]
    dy = bunch[:, 2]
    dt = bunch[:, 4] / beta0

    # Unitless momentum
    dgamma = bunch[:, 5] / mass
    gamma = gamma0 + dgamma

    betax = beta0 * bunch[:, 1]
    betay = beta0 * bunch[:, 3]
    gammax = 1.0 / np.sqrt(1.0 - betax**2)  # basically 1
    gammay = 1.0 / np.sqrt(1.0 - betay**2)  # basically 1
    dpx = (betax * gammax) / (beta0 * gamma0)
    dpy = (betay * gammay) / (beta0 * gamma0)
    dpt = (dgamma) / (beta0 * gamma0)

    return np.stack([dx, dpx, dy, dpy, dt, dpt], axis=-1)


particles = {}
particles["pyorbit"] = []
particles["impactx"] = []


# Load PyORBIT
filenames = [
    "../pyorbit/outputs/bunch_00.dat",
    "../pyorbit/outputs/bunch_01.dat",
]
for filename in filenames:
    x = np.loadtxt(filename, usecols=range(6), comments="%")
    x = pyorbit_to_impactx(
        x, kin_energy=(cfg.bunch.kin_energy * 1000.0), mass=(cfg.bunch.mass * 1000.0)
    )
    particles["pyorbit"].append(x.copy())


# Load ImpactX
file = h5py.File("../impactx/diags/openPMD/monitor.h5", "r")
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
    particles["impactx"].append(x.copy())


# Plot 2D projections (x-px, y-py, t-pt)
bins = 85

xmax = np.std(particles["pyorbit"][-1], axis=0) * 5.0 * 1000.0
limits = list(zip(-xmax, xmax))

dims = ["x", "px", "y", "py", "t", "pt"]
labels = dims

for axis in [(0, 1), (2, 3), (4, 5)]:
    for log in [False, True]:
        fig, axs = uplt.subplots(ncols=2, nrows=2, figheight=4.0)
        for j, key in enumerate(["pyorbit", "impactx"]):
            for i, x in enumerate(particles[key]):
                ax = axs[i, j]
                values, edges = np.histogramdd(
                    x[:, axis] * 1000.0, bins=bins, range=[limits[k] for k in axis]
                )
                values = values / np.max(values)

                vmax = 1.0
                vmin = 0.0
                if log:
                    values = np.log10(values + 1.00e-12)
                    vmax = 0.0
                    vmin = -4.0

                ax.pcolormesh(
                    edges[0],
                    edges[1],
                    values.T,
                    cmap="plasma",
                    vmax=vmax,
                    vmin=vmin,
                    N=14,
                    colorbar=(j == 1),
                    # colorbar_kw=dict(width=1.0),
                )

        axs.format(
            xlabel=labels[axis[0]],
            ylabel=labels[axis[1]],
            toplabels=["PyORBIT", "ImpactX"],
            leftlabels=["IN", "OUT"],
        )

        filename = f"fig_dist"
        if log:
            filename = f"{filename}_log"
        filename = f"{filename}_{dims[axis[0]]}_{dims[axis[1]]}.png"
        filename = os.path.join(output_dir, filename)
        print(filename)
        plt.savefig(filename)


# Plot 6D corner plot
import psdist.plot as psv

for index in range(2):
    print("index:", index)
    for code_name in histories:
        print(code_name)

        x = particles[code_name][index].copy()
        x = x * 1000.0
        
        grid = psv.CornerGrid(ndim=6, figwidth=7.0, diag_rspine=True, space=1.0)
        grid.set_labels(labels)
        grid.set_limits(limits)

        plot_kws = {}

        grid.plot(
            x,
            bins=64,
            limits=limits,
            process_kws=dict(scale="max"),
            mask=False,
            offset=1.00e-12,
            offset_type="absolute",
            cmap="viridis",
            diag_kws=dict(lw=1.3),
            **plot_kws,
        )

        filename = f"fig_corner_{code_name}"
        filename = f"{filename}_{index:02.0f}.png"
        filename = os.path.join(output_dir, filename)
        print(filename)
        plt.savefig(filename)
