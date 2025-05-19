import os
import math

import cheetah
import numpy as np
import scipy
import torch
from omegaconf import OmegaConf
from scipy import constants
from scipy.constants import physical_constants


# Setup
# --------------------------------------------------------------------------------------

# Load config dict
cfg = OmegaConf.load("../config.yaml")

# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# Beam
# --------------------------------------------------------------------------------------

species = cheetah.particles.Species("proton")
kin_energy = torch.tensor(cfg.kin_energy) * 1.00e+09  # [eV]
rest_energy = torch.tensor(cfg.mass * 1.00e+09)  # [eV]
elementary_charge = torch.tensor(constants.elementary_charge)

particles = torch.zeros((cfg.nparts, 7))
particles[:, 0] = torch.randn(particles.shape[0]) * cfg.xrms
particles[:, 2] = torch.randn(particles.shape[0]) * cfg.yrms
particles[:, 4] = torch.randn(particles.shape[0]) * cfg.zrms
particles[:, -1] = torch.ones(particles.shape[0])

particle_charges = (cfg.intensity / cfg.nparts) * elementary_charge

beam = cheetah.ParticleBeam(
    particles,
    energy=(kin_energy + rest_energy),
    species=species,
    particle_charges=particle_charges,
)


# Lattice
# --------------------------------------------------------------------------------------

length = torch.tensor(cfg.distance)
n_slice = 50
slice_length = length / n_slice
    
elements = []
for index in range(n_slice):
    elements.append(
        cheetah.SpaceChargeKick(
            slice_length,
            num_grid_points_x=torch.tensor(cfg.grid.x),
            num_grid_points_y=torch.tensor(cfg.grid.y),
            num_grid_points_tau=torch.tensor(cfg.grid.z),
            grid_extend_x=torch.tensor(3.0),
            grid_extend_y=torch.tensor(3.0),
            grid_extend_tau=torch.tensor(3.0),
        )
    )
    elements.append(cheetah.Drift(slice_length))

segment = cheetah.Segment(elements)


# Track
# --------------------------------------------------------------------------------------

particles = beam.particles.numpy()
particles = particles[:, :6]
np.savetxt(os.path.join(output_dir, "beam_00.dat"), particles)

print(torch.std(beam.particles, axis=0))

beam = segment.track(beam)

print(torch.std(beam.particles, axis=0))

particles = beam.particles.numpy()
particles = particles[:, :6]
np.savetxt(os.path.join(output_dir, "beam_01.dat"), particles)