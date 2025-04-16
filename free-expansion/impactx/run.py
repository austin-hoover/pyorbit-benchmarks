import argparse
import impactx
from scipy.constants import speed_of_light

from omegaconf import DictConfig
from omegaconf import OmegaConf

import yaml


# Load config dict
cfg = OmegaConf.load("../config.yaml")


# Initialize simulation
sim = impactx.ImpactX()

# Set numerical parameters and IO control
sim.max_level = 1
sim.n_cell = [16, 16, 20]
sim.blocking_factor_x = [16]
sim.blocking_factor_y = [16]
sim.blocking_factor_z = [4]

sim.particle_shape = 2  # B-spline order
sim.space_charge = "3D"
sim.poisson_solver = "fft"
sim.dynamic_size = True
sim.prob_relative = [1.2, 1.1]

# Beam diagnostics
sim.slice_step_diagnostics = True

# Domain decomposition & space charge mesh
sim.init_grids()

# Beam parameters
kin_energy = cfg.kin_energy * 1000.0  # [MeV]
total_charge = cfg.intensity * 1.602176e-19  # [C]
nparts = cfg.nparts

# Reference particle
ref_particle = sim.particle_container().ref_particle()
ref_particle.set_charge_qe(1.0)
ref_particle.set_mass_MeV(938.272029)
ref_particle.set_kin_energy_MeV(kin_energy)

# Load particles
dist = impactx.distribution.Gaussian(
    lambdaX=0.010,
    lambdaY=0.010,
    lambdaT=0.010 / ref_particle.beta,
    lambdaPx=0.0,
    lambdaPy=0.0,
    lambdaPt=0.0,
)
sim.add_particles(total_charge, dist, nparts)

# Diagnostics
monitor = impactx.elements.BeamMonitor("monitor", backend="h5")

# Create accelerator lattice
sim.lattice.extend([
    monitor,
    impactx.elements.Drift(name="drift1", ds=5.0, nslice=40),
    monitor,
])

# Run simulation
sim.track_particles()

# Clean shutdown
sim.finalize()
