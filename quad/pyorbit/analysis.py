import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


history = pd.read_csv("outputs/history.csv")

fig, ax = plt.subplots()
ax.plot(history["s"], history["x"] * 1000.0, marker=".", label="x")
ax.plot(history["s"], history["y"] * 1000.0, marker=".", label="y")
ax.set_xlabel("s [m]")
ax.set_xlabel("[mm]")
ax.legend(loc="upper right")
plt.show()