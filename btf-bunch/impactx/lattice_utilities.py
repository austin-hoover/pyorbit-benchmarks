import xml.etree.ElementTree as ET
import numpy as np
import magnet_utilities

E0 = 2.5
m0 = 938.79
speed_of_light = 2.99792458e+8
gamma = E0/m0 + 1
beta = np.sqrt(1 - 1/(gamma**2))
P0 = gamma * m0 * beta
brho = gamma*beta*m0*1e6/speed_of_light   

# some default settings for matched design beam
qdict = {}
qdict['QH01'] = 219.061
qdict['QV02'] = 285.240
qdict['QH03'] = 6.879
qdict['QV04'] = -4.475
qdict['QH05'] = 0
qdict['QH06'] = 3.828
qdict['QV07'] = -3.782
qdict['QH08'] = 3.573
qdict['QV09'] = -3.218
qdict['QH10'] = 4.935
qdict['QH30'] =  5.753
qdict['QV31'] =  -4.355
qdict['QH32'] =   6.155
qdict['QV33'] =  -4.237

def xml_to_madx(xmlfile,mstate=None,save_loc=''):
    mc =  magnet_utilities.magConvert(mstate)
    if type(mstate) == type(None):
        quad_settings = qdict
    else:
        quad_settings = magnet_utilities.quad_params_from_mstate(mstate)

    tree = ET.parse(xmlfile)
    root = tree.getroot()
    abc = list(map(chr, range(ord('a'), ord('z')+1)))

    counter = '00'
    beamlines = {}
    elements = []
    for beamline in root:
        s = 0
        lines = {}
        print(beamline.tag, beamline.attrib); print('\n')
        beamline_name = beamline.attrib['name']
        beamline_length = float(beamline.attrib['length'])
        
        # reset counter for stub, which turns off after quad QV06
        if beamline_name == 'STUB':
            counter = '06'
        
        for element in beamline:
            #print(element.tag, element.attrib)
            length = float(element.attrib['length'])
            pos = float(element.attrib['pos'])
            name = element.attrib['name'].split(':')[1]
            eletype = element.attrib['type']
            elements.append(name)
            
            # first, add preceding drift
            # need to pick name for drift. Should be # of previous quad, but with diagnostics there may be multiple drifts segments in-between quads
            
            driftname = f'DR{counter}a'
            while driftname in elements:
                segment = driftname[-1]
                idx = abc.index(segment)
                driftname = driftname.replace(segment,abc[idx+1])
            
            lastdriftlength = pos - 0.5*length - s
            if lastdriftlength > 0.:
                elements.append(driftname)
                lines[driftname] = f'DRIFT, L={lastdriftlength:.3f};\n'
            
            # now add element
            for param in element:
                #print(param.tag,param.attrib)
                #print('\n')
                continue
            if eletype == 'QUAD':
                k1 = -float(param.attrib['field'])/brho
                ## Check if in qdict defined above; if so, set K1 according to qdict
                if name in qdict.keys():
                    k1new = mc.c2gl(name.lower(),qdict[name])/length/brho
                    print(f'{name} setting changed from {k1} to {k1new} m^-2')
                    k1 = k1new
    
                line2write = f'{name}: QUADRUPOLE, L={length:.3f}, K1={k1:.3f};\n'
                counter = name[2:]
            elif eletype=='MARKER':
                line2write = f'{name}: MONITOR, L=0.0;\n'
            elif eletype=='BEND':     
                theta = float(param.attrib['theta'])
                ea1 = float(param.attrib['ea1'])
                ea2 = float(param.attrib['ea2'])
                k1 = float(param.attrib['kls'])
                line2write = f'{name}: SBEND, L={length:.3f}, ANGLE={theta:.3f}, K1={k1:.3f}, E1={ea1:.3f}, E2={ea2:.3f};\n'
            
            lines[name] = line2write.split(':')[1]
                
            # add to length:
            s += lastdriftlength + length
            
        # add end drift
        lastdriftlength = beamline_length - s
        if lastdriftlength > 0.:
            s = beamline_length
            driftname = f'DR{counter}a'
            while driftname in elements:
                segment = driftname[-1]
                idx = abc.index(segment)
                driftname = driftname.replace(segment,abc[idx+1])
    
            elements.append(driftname)
            lines[driftname] = f'DRIFT, L={lastdriftlength:.3f};\n'
            
        beamlines[beamline_name] = lines

    ### Save to file
    savefilename = xmlfile.split("/")[-1].replace('.xml','.madx')
    savefilename = save_loc + savefilename
    with open(savefilename,'w') as f:
        # define all elements
        for beamline in beamlines.keys():
            for element in beamlines[beamline].keys():
                f.write(f'{element}: {beamlines[beamline][element]}')
    
        # define beamlines
        for beamline in beamlines.keys():
            line2write = f'\n{beamline}: LINE=('
            for element in beamlines[beamline].keys():
                line2write += f'{element},'
            line2write = line2write[:-1] + ');\n'
            f.write(line2write)
    
        # define composite beamlines, set USE
        f.write("\nBEND2: LINE=(MEBT1,MEBT2);")
        f.write("\nBEND1: LINE=(MEBT1,STUB);")
        f.write("\nUSE,SEQUENCE=BEND2;")
        
        
    return savefilename