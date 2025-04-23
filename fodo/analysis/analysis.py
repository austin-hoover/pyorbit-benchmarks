def pyorbit_to_impactx(bunch: np.ndarray, kin_energy: float, mass: float) -> np.ndarray:
    """
    Args:
        kin_energy: kinetic energy [MeV]
        mass: rest mass in [MeV / c^2]
    """
    gamma0 = (mass + kin_energy) / mass
    beta0 = np.sqrt(gamma0**2 - 1.0) / gamma0

    # GeV to MeV
    bunch[:, 5] *= 1.00e+03
    
    # x -> x
    # y -> y
    # z -> ct
    dx = bunch[:, 0]
    dy = bunch[:, 2]
    dt = bunch[:, 4] / beta0
    
    # Unitless momentum
    dgamma = bunch[:, 5] / mass
    gamma = gamma0 + dgamma
    
    betax = beta0 * bunch[:, 1]
    betay = beta0 * bunch[:, 3]
    gammax = 1.0 / np.sqrt(1.0 - betax**2)  # basically 1
    gammay = 1.0 / np.sqrt(1.0 - betay**2)  # basically 1
    dpx = (betax * gammax) / (beta0 * gamma0)
    dpy = (betay * gammay) / (beta0 * gamma0)
    dpt = (dgamma) / (beta0 * gamma0)

    return np.stack([dx, dpx, dy, dpy, dt, dpt], axis=-1) 
