import copy
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from orbit.core.bunch import Bunch
from orbit.lattice import AccActionsContainer
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import QuadTEAPOT


# Setup
# --------------------------------------------------------------------------------------

# Load config dict
cfg = OmegaConf.load("../config.yaml")

# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
        

# Lattice
# --------------------------------------------------------------------------------------

lattice = TEAPOT_Lattice()

quad_node = QuadTEAPOT()
quad_node.setLength(cfg.quad.length)
quad_node.setParam("kq", cfg.quad.gradient)
quad_node.setnParts(cfg.nsteps)
quad_node.setUsageFringeFieldIN(False)
quad_node.setUsageFringeFieldOUT(False)
lattice.addNode(quad_node)
lattice.initialize()


# Bunch
# --------------------------------------------------------------------------------------

bunch = Bunch()
bunch.mass(cfg.mass)
bunch.getSyncParticle().kinEnergy(cfg.kin_energy)

x = cfg.x
y = cfg.y
z = 0.0
xp = cfg.xp
yp = cfg.yp
dE = 0.0

bunch.addParticle(x, xp, y, yp, z, dE)


# Tracking
# --------------------------------------------------------------------------------------

class Monitor:
    def __init__(self) -> None:
        self.history = {}
        for key in [
            "s",
            "x",
            "y",
            "z",
            "xp",
            "yp",
            "dE",
        ]:
            self.history[key] = []

    def __call__(self, params_dict: dict) -> None:   
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        distance = params_dict["path_length"]

        x = bunch.x(0)
        y = bunch.y(0)
        z = bunch.z(0)

        xp = bunch.xp(0)
        yp = bunch.yp(0)
        dE = bunch.dE(0)

        self.history["s"].append(distance)
        self.history["x"].append(x)
        self.history["y"].append(y)
        self.history["z"].append(z)
        self.history["xp"].append(xp)
        self.history["yp"].append(yp)
        self.history["dE"].append(dE)

        message = ""
        message += "s={:0.3f} ".format(distance)
        message += "x={:0.3f} ".format(x * 1000.0)
        message += "y={:0.3f} ".format(y * 1000.0)
        message += "z={:0.3f} ".format(z * 1000.0)
        print(node)
        print(message)

        
monitor = Monitor()
action_container = AccActionsContainer()
action_container.addAction(monitor, 0)
action_container.addAction(monitor, 1)
action_container.addAction(monitor, 2)

bunch.dumpBunch(os.path.join(output_dir, "bunch_00.dat"))

lattice.trackBunch(bunch, actionContainer=action_container)

bunch.dumpBunch(os.path.join(output_dir, "bunch_01.dat"))

history = pd.DataFrame(monitor.history)

print(history)
history.to_csv(os.path.join(output_dir, "history.csv"))
