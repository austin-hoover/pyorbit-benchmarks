import math
import numpy as np


def pyorbit_to_impactx(particles: np.ndarray, kin_energy: float, mass: float) -> np.ndarray:
    gamma0 = (mass + kin_energy) / mass
    beta0 = np.sqrt(gamma0 * gamma0 - 1.0) / gamma0
    
    # Convert spatial coordinates
    dx = particles[:, 0]  # [m]
    dy = particles[:, 2]  # [m]
    dt = particles[:, 4] / beta0  # ct [m]
    
    # Convert momentum coordinates
    dgamma = particles[:, 5] / mass
    gamma = gamma0 + dgamma
    dbetaz = 0.25 * gamma0**(-1.5) * beta0**(-1.0) * dgamma
    betax = beta0 * particles[:, 1]
    betay = beta0 * particles[:, 3]
    gammax = 1.0 / np.sqrt(1.0 - betax**2)  # ~1
    gammay = 1.0 / np.sqrt(1.0 - betay**2)  # ~1
    dpx = (betax * gammax) / (beta0 * gamma0)
    dpy = (betay * gammay) / (beta0 * gamma0)
    dpt = (dgamma) / (beta0 * gamma0)

    return np.stack([dx, dpx, dy, dpy, dt, dpt], axis=-1)