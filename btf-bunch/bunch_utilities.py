import pandas as pd
import numpy as np

def pyorbit_to_impactx(filename):
    """
    units_in  = ['m','rad','m','rad','m','GeV'] # units for pyorbit
    units_out  = ['m','unitless momentum','m','unitless momentum','m','unitless momentum'] 
    """
    names=['x','xp','y','yp','z','de','id','phi']
    skiprows = 15
    
    thisbunch=pd.read_csv(filename, names=names,\
                              delimiter='\s+',
                              index_col=None,\
                              skiprows=skiprows)      
    
    
    # -- GeV to MeV
    thisbunch.loc[:,'de'] *= 1e3 # 
    
    print("N particles in bunch: %i"%(len(thisbunch)))

    # to impactx spatial units
    dx = thisbunch['x'].values # meters
    dy = thisbunch['y'].values # meters
    dt = thisbunch['z'].values / beta0 # time of flight, ct
    
    # unitless momenta
    dgamma = thisbunch['de'].values / m0
    gamma = gamma0 + dgamma
    dbetaz = 1/4*gamma0**(-3/2)*beta0**-1 * dgamma
    
    betax = beta0*thisbunch['xp'].values
    gammax = 1/np.sqrt(1-betax**2) # basically 1
    betay = beta0*thisbunch['yp'].values
    gammay = 1/np.sqrt(1-betay**2) # basically 1
    dpx = (betax*gammax)/(beta0*gamma0)
    dpy = (betay*gammay)/(beta0*gamma0)
    dpt = (dgamma)/(beta0*gamma0)

    bunch = {'x':dx,'y':dy,'t':dt,'px':dpx,'py':dpy,'pt':dpt}
    return bunch