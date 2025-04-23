import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["axes.linewidth"] = 1.25
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True


# Collect output folder names and sort by timestamp
input_dir = "outputs"
folders = os.listdir(input_dir)
folders = sorted(folders)
folders = [os.path.join(input_dir, f) for f in folders]
print(folders)


# Collect info dicts
infos = []
for folder in folders:
    filename = os.path.join(folder, "info.pkl")
    with open(filename, "rb") as file:
        info = pickle.load(file)
    infos.append(info)

    print(filename)
    print(info)


# Plot number of MPI processes vs. execution time.
mpi_nodes = [info["mpi_size"] for info in infos]
run_times = [info["run_time"] for info in infos]

fig, ax = plt.subplots()
ax.plot(mpi_nodes, run_times, lw=1.5, marker=".")
ax.set_xlabel("MPI processes")
ax.set_ylabel("Run time")
plt.show()
