import numpy as np
import scipy
from omegaconf import DictConfig
from omegaconf import OmegaConf
from scipy.constants import speed_of_light

import impactx


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
sim.prob_relative = [1.1, 1.1]

# Beam diagnostics
sim.slice_step_diagnostics = True

# Domain decomposition & space charge mesh
sim.init_grids()

# Beam parameters
kin_energy = cfg.bunch.kin_energy * 1000.0  # [MeV]
mass = cfg.bunch.mass * 1000.0  # [MeV]
total_charge = cfg.bunch.intensity * 1.602176e-19  # [C]
nparts = cfg.bunch.size

# Reference particle
ref_particle = sim.particle_container().ref_particle()
ref_particle.set_charge_qe(cfg.bunch.charge)
ref_particle.set_mass_MeV(mass)
ref_particle.set_kin_energy_MeV(kin_energy)

# Add particles
alpha_x = cfg.bunch.alpha_x
alpha_y = cfg.bunch.alpha_y
beta_x = cfg.bunch.beta_x
beta_y = cfg.bunch.beta_y
eps_x = cfg.bunch.eps_x
eps_y = cfg.bunch.eps_y
gamma_x = (1.0 + alpha_x**2) / beta_x
gamma_y = (1.0 + alpha_y**2) / beta_y

dist = impactx.distribution.Gaussian(
    lambdaX=np.sqrt(eps_x / gamma_x),
    lambdaY=np.sqrt(eps_y / gamma_y),
    lambdaT=(cfg.bunch.sigma_z / ref_particle.beta),
    lambdaPx=np.sqrt(eps_x / beta_x),
    lambdaPy=np.sqrt(eps_y / beta_y),
    lambdaPt=0.0,
    muxpx=alpha_x / np.sqrt(beta_x * gamma_x),
    muypy=alpha_y / np.sqrt(beta_y * gamma_y),
    mutpt=0.0,
)

sim.add_particles(total_charge, dist, nparts)

# Create accelerator lattice
length = cfg.lattice.length
fill_fraction = cfg.lattice.fill_fraction
length_frac = length * fill_fraction * 0.50
kq = cfg.lattice.quad_gradient

# Try constant slice width
nslice = int(0.5 * length_frac / cfg.lattice.ds) + 1

monitor = impactx.elements.BeamMonitor("monitor", backend="h5")

elements = [
    impactx.elements.Quad(ds=(length_frac * 0.5), k=+kq, nslice=(nslice * 1)),
    impactx.elements.Drift(ds=(length_frac * 1.0), nslice=(nslice * 2)),
    impactx.elements.Quad(ds=(length_frac * 1.0), k=-kq, nslice=(nslice * 2)),
    impactx.elements.Drift(ds=(length_frac * 1.0), nslice=(nslice * 2)),
    impactx.elements.Quad(ds=(length_frac * 0.5), k=+kq, nslice=(nslice * 1)),
]
sim.lattice.append(monitor)
sim.lattice.extend(elements * 3)
sim.lattice.append(monitor)

# Run simulation
sim.track_particles()
sim.finalize()
