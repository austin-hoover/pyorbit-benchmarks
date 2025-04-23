import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True


# Setup
# --------------------------------------------------------------------------------------

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

histories = {}


# ImpactX
# --------------------------------------------------------------------------------------

filename = "../impactx/diags/reduced_beam_characteristics.0.0"
history = pd.read_csv(filename, delimiter=" ")
history["x"] = history["x_min"] + history["x_max"]
history["y"] = history["y_min"] + history["y_max"]

histories["impactx"] = history.copy()


# PyORBIT
# --------------------------------------------------------------------------------------

history = pd.read_csv("../pyorbit/outputs/history.csv")

histories["pyorbit"] = history.copy()


# Plot comparison
# --------------------------------------------------------------------------------------

for code_name, history in histories.items():
    fig, ax = plt.subplots()
    ax.plot(history["s"], history["x"] * 1000.0, label="x")
    ax.plot(history["s"], history["y"] * 1000.0, label="y")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("[mm]")
    ax.legend(loc="upper left")

    filename = f"fig_xy_{code_name}.png"
    filename = os.path.join(output_dir, filename)
    fig.savefig(filename, dpi=300)