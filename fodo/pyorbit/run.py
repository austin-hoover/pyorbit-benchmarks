import os
import time
import sys

import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import GaussDist2D
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.space_charge.sc3d import SC3D_AccNode
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import DriftTEAPOT
from orbit.teapot import QuadTEAPOT


# Setup
# --------------------------------------------------------------------------------------

_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

cfg = OmegaConf.load("../config.yaml")
if _mpi_rank == 0:
    print(cfg)

output_dir = "outputs"
if _mpi_rank == 0:
    os.makedirs(output_dir, exist_ok=True)

rng = np.random.default_rng(cfg.seed)


# Lattice
# --------------------------------------------------------------------------------------

length = cfg.lattice.length
fill_fraction = cfg.lattice.fill_fraction

drift_node_1 = DriftTEAPOT("d1")
drift_node_2 = DriftTEAPOT("d2")
drift_node_1.setLength(length * fill_fraction * 0.50)
drift_node_2.setLength(length * fill_fraction * 0.50)

quad_node_1 = QuadTEAPOT("q1")
quad_node_2 = QuadTEAPOT("q2")
quad_node_3 = QuadTEAPOT("q3")
quad_node_1.setLength(length * fill_fraction * 0.25)
quad_node_2.setLength(length * fill_fraction * 0.50)
quad_node_3.setLength(length * fill_fraction * 0.25)
quad_node_1.setParam("kq", +cfg.lattice.quad_gradient)
quad_node_2.setParam("kq", -cfg.lattice.quad_gradient)
quad_node_3.setParam("kq", +cfg.lattice.quad_gradient)

lattice = TEAPOT_Lattice()
lattice.addNode(quad_node_1)
lattice.addNode(drift_node_1)
lattice.addNode(quad_node_2)
lattice.addNode(drift_node_2)
lattice.addNode(quad_node_3)
lattice.initialize()

for node in lattice.getNodes():
    node.setnParts(1 + int(node.getLength() / cfg.lattice.ds))

sc_path_length_min = 1.00e-06
sc_calc = SpaceChargeCalc3D(cfg.spacecharge.grid.x, cfg.spacecharge.grid.y, cfg.spacecharge.grid.z)
sc_nodes = setSC3DAccNodes(lattice, sc_path_length_min, sc_calc)

if not cfg.lattice.spacecharge:
    for sc_node in sc_nodes:
        sc_node.switcher = False


# Bunch
# --------------------------------------------------------------------------------------

bunch = Bunch()
bunch.mass(cfg.bunch.mass)
bunch.getSyncParticle().kinEnergy(cfg.bunch.kin_energy)

alpha_x = cfg.bunch.alpha_x
alpha_y = cfg.bunch.alpha_y
beta_x = cfg.bunch.beta_x
beta_y = cfg.bunch.beta_y
eps_x = cfg.bunch.eps_x
eps_y = cfg.bunch.eps_y
twiss_x = TwissContainer(alpha_x, beta_x, eps_x)
twiss_y = TwissContainer(alpha_y, beta_y, eps_y)
dist = GaussDist2D(twiss_x, twiss_y)

data_type = orbit_mpi.mpi_datatype.MPI_DOUBLE
main_rank = 0

for i in range(cfg.bunch.size):
    x, xp, y, yp = dist.getCoordinates()
    z = rng.normal(scale=cfg.bunch.sigma_z)
    de = 0.0

    (x, xp, y, yp, z, de) = orbit_mpi.MPI_Bcast(
        (x, xp, y, yp, z, de), data_type, main_rank, _mpi_comm
    )
    if i % _mpi_size == _mpi_rank:
        bunch.addParticle(x, xp, y, yp, z, de)

if cfg.bunch.intensity > 0:
    bunch.macroSize(cfg.bunch.intensity / cfg.bunch.size)


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

        if _mpi_rank == 0:
            message = ""
            message += "s={:0.3f} ".format(distance)
            message += "xrms={:0.3f} ".format(sigma_x * 1000.0)
            message += "yrms={:0.3f} ".format(sigma_y * 1000.0)
            message += "zrms={:0.3f} ".format(sigma_z * 1000.0)
            print(message)
            sys.stdout.flush()


monitor = Monitor()
action_container = AccActionsContainer()
action_container.addAction(monitor, AccActionsContainer.ENTRANCE)
action_container.addAction(monitor, AccActionsContainer.EXIT)

bunch.dumpBunch(os.path.join(output_dir, "bunch_00.dat"))

for period in range(cfg.lattice.periods):
    lattice.trackBunch(bunch, actionContainer=action_container)

bunch.dumpBunch(os.path.join(output_dir, "bunch_01.dat"))

history = pd.DataFrame(monitor.history)
history.to_csv(os.path.join(output_dir, "history.csv"))
