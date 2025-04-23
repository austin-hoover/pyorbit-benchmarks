import os
import numpy as np
import matplotlib.pyplot

folders = os.listdir(".")
folders = [folder for folder in folders if folder.startswith("outputs_")]
folders = sorted(folders)
print(folders)