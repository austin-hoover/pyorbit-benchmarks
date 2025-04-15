import argparse
import os
import pathlib
import sys
import time

import numpy as np

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.spacecharge import SpaceChargeCalc3D
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import TwissContainer
from orbit.lattice import AccActionsContainer
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import DriftTEAPOT
from orbit.utils.consts import charge_electron
from orbit.utils.consts import mass_proton


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nparts", type=int, default=100_000)
parser.add_argument("--mass", type=float, default=mass_proton)
parser.add_argument("--current", type=float, default=0.050)
parser.add_argument("--kin-energy", type=float, default=0.0025)

parser.add_argument("--scale-x", type=float, default=0.001)
parser.add_argument("--scale-y", type=float, default=0.001)
parser.add_argument("--scale-z", type=float, default=0.001)

parser.add_argument("--sc-grid-x", type=int, default=64)
parser.add_argument("--sc-grid-y", type=int, default=64)
parser.add_argument("--sc-grid-z", type=int, default=64)

parser.add_argument("--distance", type=float, default=1.000)
parser.add_argument("--ds", type=float, default=0.01)

parser.add_argument("--seed", type=float, default=0)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


# Create output directory
path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
if _mpi_rank  == 0:
    os.makedirs(output_dir, exist_ok=True)
        

# Lattice
# --------------------------------------------------------------------------------------

# Create a drift lattice with required step size
lattice = TEAPOT_Lattice()
for _ in range(int(args.distance / args.ds)):
    node = DriftTEAPOT()
    node.setLength(args.ds)
    lattice.addNode(node)
lattice.initialize()

# Add 3D space charge nodes
sc_calc = SpaceChargeCalc3D(args.sc_grid_x, args.sc_grid_y, args.sc_grid_z)
sc_nodes = setSC3DAccNodes(lattice, args.ds, sc_calc)


# Bunch
# --------------------------------------------------------------------------------------

# Create empty bunch
bunch = Bunch()
bunch.mass(args.mass)
bunch.getSyncParticle().kinEnergy(args.kin_energy)

# Add particles to bunch
rng = np.random.default_rng(args.seed)
for index in range(args.nparts):    
    x = rng.normal(scale=args.scale_x)
    y = rng.normal(scale=args.scale_y)
    z = rng.normal(scale=args.scale_z)
    xp = 0.0
    yp = 0.0
    de = 0.0

    (x, xp, y, yp, z, de) = orbit_mpi.MPI_Bcast(
        (x, xp, y, yp, z, de), orbit_mpi.mpi_datatype.MPI_DOUBLE, 0, _mpi_comm
    )

    if index % _mpi_size == _mpi_rank:
        bunch.addParticle(x, xp, y, yp, z, de)
    
# Set macrosize
frequency = 402.5e6  # [Hz]
charge = args.current / frequency
intensity = charge / (abs(bunch.charge()) * charge_electron)

size_global = bunch.getSizeGlobal()
macro_size = intensity / size_global
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
    """Monitors bunch size during simulation."""
    def __init__(self) -> None:
        self.start_time = None

    def __call__(self, params_dict: dict) -> None:   
        # Update parameters
        if self.start_time is None:
            self.start_time = time.time()

        time_ellapsed = time.time() - self.start_time
        distance = params_dict["path_length"]

        # Calculate bunch statistics
        bunch = params_dict["bunch"]

        cov_matrix = get_bunch_cov(bunch)
        x_rms = np.sqrt(cov_matrix[0, 0]) * 1000.0
        y_rms = np.sqrt(cov_matrix[2, 2]) * 1000.0
        z_rms = np.sqrt(cov_matrix[4, 4]) * 1000.0

        # Print update
        if _mpi_rank == 0:
            message = ""
            message += "time={:0.3f} ".format(time_ellapsed) 
            message += "s={:0.3f} ".format(distance)
            message += "xrms={:0.3f} ".format(x_rms)
            message += "yrms={:0.3f} ".format(y_rms)
            message += "zrms={:0.3f} ".format(z_rms)
            print(message)
            sys.stdout.flush()

        
monitor = Monitor()
action_container = AccActionsContainer()
action_container.addAction(monitor, AccActionsContainer.EXIT)

bunch.dumpBunch(os.path.join(output_dir, "bunch_00.dat"))

lattice.trackBunch(bunch, actionContainer=action_container)

bunch.dumpBunch(os.path.join(output_dir, "bunch_01.dat"))