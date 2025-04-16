import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.lattice import AccActionsContainer
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import DriftTEAPOT


# Setup
# --------------------------------------------------------------------------------------

# Load config dict
cfg = OmegaConf.load("../config.yaml")

# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
        

# Lattice
# --------------------------------------------------------------------------------------

delta_s = cfg.distance / cfg.nsteps

lattice = TEAPOT_Lattice()
for _ in range(cfg.nsteps):
    node = DriftTEAPOT()
    node.setLength(delta_s)
    lattice.addNode(node)

lattice.initialize()

sc_calc = SpaceChargeCalc3D(cfg.grid.x, cfg.grid.y, cfg.grid.z)
sc_nodes = setSC3DAccNodes(lattice, delta_s, sc_calc)


# Bunch
# --------------------------------------------------------------------------------------

bunch = Bunch()
bunch.mass(cfg.mass)
bunch.getSyncParticle().kinEnergy(cfg.kin_energy)

rng = np.random.default_rng(cfg.seed)
for index in range(cfg.nparts):    
    x = rng.normal(scale=cfg.xrms)
    y = rng.normal(scale=cfg.yrms)
    z = rng.normal(scale=cfg.zrms)
    xp = 0.0
    yp = 0.0
    de = 0.0
    bunch.addParticle(x, xp, y, yp, z, de)
    
size_global = bunch.getSizeGlobal()
macro_size = cfg.intensity / size_global
bunch.macroSize(macro_size)


# Tracking
# --------------------------------------------------------------------------------------

def get_bunch_cov(bunch: Bunch) -> np.ndarray:
    order = 2
    dispersion_flag = 0
    emit_norm_flag = 0

    twiss_calc = BunchTwissAnalysis()
    twiss_calc.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)

    cov_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            cov_matrix[i, j] = cov_matrix[j, i] = twiss_calc.getCorrelation(j, i)
    return cov_matrix


class Monitor:
    def __init__(self) -> None:
        self.history = {}
        for key in [
            "s",
            "sig_x",
            "sig_y",
            "sig_z",
            "emittance_x",
            "emittance_y",
            "emittance_z",
        ]:
            self.history[key] = []

    def __call__(self, params_dict: dict) -> None:   
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        distance = params_dict["path_length"]

        cov_matrix = get_bunch_cov(bunch)
        sigma_x = np.sqrt(cov_matrix[0, 0])
        sigma_y = np.sqrt(cov_matrix[2, 2])
        sigma_z = np.sqrt(cov_matrix[4, 4])

        emittance_x = np.sqrt(np.linalg.det(cov_matrix[0:2, 0:2]))
        emittance_y = np.sqrt(np.linalg.det(cov_matrix[2:4, 2:4]))
        emittance_z = np.sqrt(np.linalg.det(cov_matrix[4:6, 4:6]))

        self.history["s"].append(distance)
        self.history["sig_x"].append(sigma_x)
        self.history["sig_y"].append(sigma_y)
        self.history["sig_z"].append(sigma_z)
        self.history["emittance_x"].append(emittance_x)
        self.history["emittance_y"].append(emittance_y)
        self.history["emittance_z"].append(emittance_z)

        message = ""
        message += "s={:0.3f} ".format(distance)
        message += "xrms={:0.3f} ".format(sigma_x * 1000.0)
        message += "yrms={:0.3f} ".format(sigma_y * 1000.0)
        message += "zrms={:0.3f} ".format(sigma_z * 1000.0)
        print(message)

        
monitor = Monitor()
action_container = AccActionsContainer()
action_container.addAction(monitor, AccActionsContainer.ENTRANCE)
action_container.addAction(monitor, AccActionsContainer.EXIT)

bunch.dumpBunch(os.path.join(output_dir, "bunch_00.dat"))

lattice.trackBunch(bunch, actionContainer=action_container)

bunch.dumpBunch(os.path.join(output_dir, "bunch_01.dat"))

history = pd.DataFrame(monitor.history)
history.to_csv(os.path.join(output_dir, "history.csv"))