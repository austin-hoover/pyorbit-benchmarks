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

R0 = torch.tensor(0.001)
energy = torch.tensor(2.50e+08)
rest_energy = torch.tensor(constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge)
elementary_charge = torch.tensor(constants.elementary_charge)
electron_radius = torch.tensor(physical_constants["classical electron radius"][0])
gamma = energy / rest_energy
beta = torch.sqrt(1.0 - 1.0 / gamma**2)

beam = cheetah.ParticleBeam.uniform_3d_ellipsoid(
    num_particles=torch.tensor(100_000),
    total_charge=torch.tensor(1e-8),
    energy=energy,
    radius_x=R0,
    radius_y=R0,
    radius_tau=R0 / gamma,  # Radius of the beam in s direction in the lab frame
    sigma_px=torch.tensor(1e-15),
    sigma_py=torch.tensor(1e-15),
    sigma_p=torch.tensor(1e-15),
)

# Lattice
# --------------------------------------------------------------------------------------

# Compute section length
kappa = 1.0 + (torch.sqrt(torch.tensor(2.0)) / 4.0) * torch.log(3.0 + 2.0 * torch.sqrt(torch.tensor(2.0)))
intensity = beam.total_charge / elementary_charge
section_length = beta * gamma * kappa * torch.sqrt(R0**3 / (intensity * electron_radius))

nslice = 10
slice_length = section_length / nslice


elements = []
elements.append(cheetah.Drift(slice_length * 0.5))
for index in range(nslice):
    elements.append(
        cheetah.SpaceChargeKick(
            effect_length=slice_length,
            num_grid_points_x=64,
            num_grid_points_y=64,
            num_grid_points_tau=64,
            grid_extend_x=3,  # TODO: Simplify these to a single tensor?
            grid_extend_y=3,
            grid_extend_tau=3,
        )
    )
    elements.append(cheetah.Drift(slice_length))
elements.append(cheetah.Drift(slice_length * 0.5))

segment = cheetah.Segment(elements)


# Track
# --------------------------------------------------------------------------------------

particles = beam.particles.numpy()
particles = particles[:, :6]
np.savetxt(os.path.join(output_dir, "beam_00.dat"), particles)

beam = segment.track(beam)

particles = beam.particles.numpy()
particles = particles[:, :6]
np.savetxt(os.path.join(output_dir, "beam_01.dat"), particles)