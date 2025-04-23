import numpy as np
import pandas as pd
import openpmd_api as io

import matplotlib.pyplot as plt


filename = "diags/ref_particle.0.0"
history_ref = pd.read_csv(filename, delimiter=" ")

filename = "diags/reduced_beam_characteristics.0.0"
history = pd.read_csv(filename, delimiter=" ")

print(history)

xmax = history.loc[:, "x_max"].values
xmin = history.loc[:, "x_min"].values
ymax = history.loc[:, "y_max"].values
ymin = history.loc[:, "y_min"].values
s = history.loc[:, "s"].values
x = xmax + xmin
y = ymax + ymin

fig, ax = plt.subplots()
ax.plot(s, x * 1000.0, label="x")
ax.plot(s, y * 1000.0, label="y")
ax.set_xlabel("s [m]")
ax.set_xlabel("[mm]")
ax.legend(loc="upper right")
plt.show()