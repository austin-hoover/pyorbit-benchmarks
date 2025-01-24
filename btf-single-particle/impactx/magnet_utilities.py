import sys
import numpy as np
from collections import OrderedDict


def quad_params_from_mstate(filename: str, param_name: str = "setpoint") -> dict:
    """Load quadrupole parameters from .mstate file."""
    tree = ET.parse(mstate)
    root = tree.getroot()
    setpoints = collections.OrderedDict()
    for item in root[0]:
        pv_name = item.attrib["setpoint_pv"].split(":")
        ps_name = pv_name[1].split("_")
        mag_name = ps_name[1].lower()
        setpoints[mag_name] = float(item.attrib[param_name])
    return setpoints
    

def file2dict(filename):
	#
	# writes text file with two columns to dictionary
	#
	filestring = open(filename,'r').read().split('\n')
	thisdict = OrderedDict()
	for item in filestring:
		if item: # if this isn't an empty line
			if item[0] != '#': # and if not a comment
				try: # added "try" layer because last item might be a single-space string
					items = item.split(', ')
					itemkey = items[0].lower() # first piece is item "key"
					itemvalue = items[1:]
					if len(itemvalue)==1: # if length is 1, save as string not list
						itemvalue = items[1]
					thisdict[itemkey] = itemvalue # other pieces are stored in dictionary
				except:
					print("Skipped line '%s'"%item)
					pass

	return thisdict

class magConvert(object):
    # converst gradients to currents and vice versa, based on cofficients in file
    def __init__(self,file = None):
        if type(file) == type(None):
            print('no coeffilename specified')
            filename = '../pyorbit/inputs/magnets/magnet_coefficients.csv'
        else: 
            filename = file
        print('using file %s'%filename)
        self.coeff = file2dict(filename)

    def c2gl(self,quadname,scaledAI):
        """
        arguments: 
        1 - Name of quad (ie, QH01)
        2 - Current setpoint (corresponds with scaled AI in IOC), [A]
        """
        scaledAI = float(scaledAI)
        try:
            A = float(self.coeff[quadname][0])
            B = float(self.coeff[quadname][1])
            GL = A*scaledAI + B*scaledAI**2 
        except KeyError: 
            print("Do not know conversion factor for element %s, gradient value not assigned"%quadname) 
            GL = []
        return GL

    def gl2c(self,quadname,GL):
        """
        arguments: 
        1 - Name of quad (ie, QH01)
        2 - Integrated gradient (GL), [T]
        """	
        GL = float(GL) 
        try:
            A = float(self.coeff[quadname][0])
            B = float(self.coeff[quadname][1])
            if B==0 and A==0 : # handle case of 0 coefficients
                scaledAI = 0
            elif B==0 and A!=0 : # avoid division by 0 for quads with 0 offset
                scaledAI = GL/A
            else:
                scaledAI = 0.5*(A/B) * (-1 + np.sqrt(1 + 4*GL*B/A**2))			
        except KeyError: 
            print("Do not know conversion factor for element %s, current set to 0"%quadname) 
            scaledAI = 0
        return scaledAI

    def igrad2current(self,inputdict):
        """
        Input is dictionary where key is name of magnet, and value is integrated gradient GL [T]
        """
        outputdict = OrderedDict.fromkeys(self.coeff.keys(),[])
        for name in inputdict:
            try: outputdict[name] = self.gl2c(name,inputdict[name])
            except: print("something went wrong on element %s"%name)
        return outputdict			

    def current2igrad(self,inputdict):
        """
        Input is dictionary where key is name of magnet, and value is current setpoint [A]
        """
        outputdict = OrderedDict.fromkeys(self.coeff.keys(),[])
        for name in inputdict:
            try: outputdict[name] = self.c2gl(name,inputdict[name])
            except: print("something went wrong on element %s"%name)
        return outputdict


