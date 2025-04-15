import argparse
from pprint import pprint

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import TEAPOT_Ring
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils.consts import mass_proton


parser = argparse.ArgumentParser()
parser.add_argument("--mass", type=float, default=mass_proton)
parser.add_argument("--energy", type=float, default=1.300)
parser.add_argument("--fringe", type=int, default=0)
args = parser.parse_args()

pprint(args)


# Build lattice from MADX lattice file
lattice = TEAPOT_Ring()
lattice.readMADX("../common-inputs/sns_ring.lattice", "rnginj")
lattice.initialize()

# Toggle fringe fields
for node in lattice.getNodes():
    try:
        node.setUsageFringeFieldIN(args.fringe)
        node.setUsageFringeFieldOUT(args.fringe)
    except:
        pass
        

# Extract linear transfer matrix and print parameters
bunch = Bunch()
bunch.mass(args.mass)
bunch.getSyncParticle().kinEnergy(args.energy)

matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
lattice_params = matrix_lattice.getRingParametersDict()
pprint(lattice_params)

