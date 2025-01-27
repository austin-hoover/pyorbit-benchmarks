from sns_linac_bunch_generator import SNS_Linac_BunchGenerator as LBG
import numpy as np

n_particles = 20000
beam_current = 40 # mA
distribution = 'waterbag'
cut_off = -1 # only used for Gaussian, units=sigmas
filename = f'initial_bunch_{n_particles*1e-3:.0f}k_{beam_current:.0f}mA_{distribution}.dat'

ax = -1.9899
bx = 0.19636
ex = 0.160372
ay = 1.92893
by = 0.17778
ey = 0.16362
az = 0.
bz = 0.6
ez = 0.2

ekin = 0.0025 # in [GeV]
mass0 = 0.939294 # in [GeV]
gamma = (mass0 + ekin)/mass0
beta = np.sqrt(gamma*gamma - 1.0)/gamma


#---make emittances un-normalized XAL units [m*rad]
ex = 1.0e-6*ex/(gamma*beta)
ey = 1.0e-6*ey/(gamma*beta)
ez = 1.0e-6*ez/(gamma**3*beta)

#---- transform to pyORBIT emittance[GeV*m]
ez = ez*gamma**3*beta**2*mass0
bz = bz/(gamma**3*beta**2*mass0)


twissx = (ax,bx,ex)
twissy = (ay,by,ey)
twissz = (az,bz,ez)

bunchgen = LBG(twissX = twissx, twissY = twissy, twissZ = twissz)
bunchgen.setBeamCurrent(beam_current)

bunch = bunchgen.getBunch(nParticles=n_particles,distribution=distribution,cut_off=cut_off)

bunch.dumpBunch(filename)


