from impactx import ImpactX
from impactx import elements
import amrex.space3d as amr

import pandas as pd
import numpy as np
import glob
import xml.etree.ElementTree as ET
import collections
import os
import time

import magnet_utilities
import bunch_utilities
import lattice_utilities


###########
# constants
###########
energy_MeV = 2.5
mass_MeV = 938.79
freq = 402.5e6 # Hz
current = 0.040 # amps
bunch_charge_C = 0.  # used with space charge

speed_of_light = 2.99792458e+8
gamma0 = energy_MeV/mass_MeV + 1
beta0 = np.sqrt(1 - 1/(gamma0**2))
P0 = gamma0 * mass_MeV * beta0
brho = gamma0*beta0*mass_MeV*1e6/speed_of_light 

########
# config
########
lattice_file = '../common-inputs/xml/btf_lattice_straight.xml'
mstate_file = '../common-inputs/mstate/settings_bend2_45mA_mismatch2_250109.mstate'

input_dir = 'inputs/'
output_dir = 'outputs/'
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


#######################
# Initialize simulation
#######################
sim = ImpactX()

# make lattice file, load
new_lattice_file = lattice_utilities.xml_to_madx(lattice_file,save_loc=input_dir)
time.sleep(1)
sim.lattice.load_file(new_lattice_file,nslice=10)

# set numerical parameters and IO control
sim.particle_shape = 1  # B-spline order (1,2 or 3) higher is better but slower
sim.space_charge = False
# sim.diagnostics = False  # benchmarking
sim.slice_step_diagnostics = True

# domain decomposition & space charge mesh
sim.init_grids()

##   reference particle
ref = sim.particle_container().ref_particle()
ref.set_charge_qe(-1.0).set_mass_MeV(mass_MeV).set_kin_energy_MeV(energy_MeV)
qm_eev = -1.0 / mass_MeV / 1e6  # electron charge/mass in e / eV
ref.z = 0

### load and set magnet parameters
quad_setpoints = magnet_utilities.quad_params_from_mstate(mstate_file)
mc = magnet_utilities.magConvert()

quadlist = [item for item in list(sim.lattice) if type(item)==elements.Quad]
for quad in quadlist:
    if quad.name in quad_setpoints.keys():
        setpoint = quad_setpoints[quad.name]
        field =  mc.c2gl(quad.name,setpoint)
        k = field / quad.ds / brho
        print(f"{quad.name}: changed from k={quad.k:.3f} to k={k:.3f}")
        quad.k = k
    else:
        print(f"{quad.name}: k={quad.k:.3f}")

### single bunch coordinates
bunch = {'x':0.,'y':0.,'t':0.,'px':.001,'py':0.001,'pt':0.0001}

pc = sim.particle_container()

# no gpu option for placing particles on mesh
dx_podv = amr.PODVector_real_std()
dy_podv = amr.PODVector_real_std()
dt_podv = amr.PODVector_real_std()
dpx_podv = amr.PODVector_real_std()
dpy_podv = amr.PODVector_real_std()
dpt_podv = amr.PODVector_real_std()

# place particles
for p_dx in bunch['x']:
    dx_podv.push_back(p_dx)
for p_dy in bunch['y']:
    dy_podv.push_back(p_dy)
for p_dt in bunch['t']:
    dt_podv.push_back(p_dt)
for p_dpx in bunch['px']:
    dpx_podv.push_back(p_dpx)
for p_dpy in bunch['py']:
    dpy_podv.push_back(p_dpy)
for p_dpt in bunch['pt']:
    dpt_podv.push_back(p_dpt)

pc.add_n_particles(
    dx_podv, dy_podv, dt_podv, dpx_podv, dpy_podv, dpt_podv, qm_eev, bunch_charge_C
)


#######
# Track
#######

sim.track_particles()

# clean shutdown
sim.finalize()