import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from scipy.constants import speed_of_light

import impactx
import amrex.space3d as amr


# Load config dict
cfg = OmegaConf.load("../config.yaml")


# Initialize simulation
sim = impactx.ImpactX()

# Set numerical parameters and IO control
sim.particle_shape = 2  # B-spline order
sim.space_charge = False
sim.slice_step_diagnostics = True

# Domain decomposition & space charge mesh
sim.init_grids()

# Beam parameters
kin_energy = cfg.kin_energy * 1000.0  # [MeV]
mass = cfg.mass * 1000.0  # [MeV]

# Reference particle
ref_particle = sim.particle_container().ref_particle()
ref_particle.set_charge_qe(cfg.charge)
ref_particle.set_mass_MeV(mass)
ref_particle.set_kin_energy_MeV(kin_energy)
ref_particle.z = 0

# Add single particle.
particle_container = sim.particle_container()

dx_podv = amr.PODVector_real_std()
dy_podv = amr.PODVector_real_std()
dt_podv = amr.PODVector_real_std()
dpx_podv = amr.PODVector_real_std()
dpy_podv = amr.PODVector_real_std()
dpt_podv = amr.PODVector_real_std()

coords = np.zeros((2, 6))
coords[0, :] = [cfg.x, cfg.xp, cfg.y, cfg.yp, 0.0, 0.0]

for p_dx in coords[:, 0]:
    dx_podv.push_back(p_dx)
for p_dy in coords[:, 2]:
    dy_podv.push_back(p_dy)
for p_dt in coords[:, 4]:
    dt_podv.push_back(p_dt)
for p_dpx in coords[:, 1]:
    dpx_podv.push_back(p_dpx)
for p_dpy in coords[:, 3]:
    dpy_podv.push_back(p_dpy)
for p_dpt in coords[:, 5]:
    dpt_podv.push_back(p_dpt)

charge_to_mass_ratio = 1.0 / mass / 1.00e+06  # proton charge/mass in e / eV
total_charge = 0.0
particle_container.add_n_particles(
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, charge_to_mass_ratio, total_charge
)

# Create accelerator lattice
quad_length = cfg.quad.length
quad_length = quad_length * 0.5  # ?
quad = impactx.elements.Quad(name="quad1", ds=quad_length, k=cfg.quad.gradient, nslice=cfg.nsteps)
sim.lattice.append(quad)

# Run simulation

# The monitor will give NaN for x_mean and y_mean. We can track the particle coordinates
# using x_max and x_min. Note that x_min is set to zero if no particles have negative x, 
# and x_max is set to zero if no particles have positive x. Thus to get the single 
# particle coordiantes: x = x_min + x_max.
monitor = impactx.elements.BeamMonitor("monitor", backend="h5")
sim.lattice.extend([
    monitor,
    quad,
    monitor,
])

sim.track_particles()
sim.finalize()

