# import pyg4ometry.convert as convert
# import pyg4ometry.visualisation as vi
import numpy as np
from pyg4ometry.fluka import RPP, Region, Zone, Material, FlukaRegistry, Writer

beam_str = """
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
TITLE
Neutron fluence after a proton-irradiated Be target
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
BEAM         50.E-03                                                  PROTON
BEAMPOS          0.0       0.0     -50.0
*
GEOBEGIN                                                              COMBNAME
  0 0                       Be target inside vacuum
RPP body1 -5000000.0 +5000000.0 -5000000.0 +5000000.0 -5000000.0 +5000000.0
RPP body2 -1000000.0 +1000000.0 -1000000.0 +1000000.0     -100.0 +1000000.0
RPP body3      -10.0      +10.0      -10.0      +10.0        0.0      +20.0
RPP body4      -10.0      +10.0      -10.0      +10.0       +20.0     +40.0
* plane to separate the upstream and downstream part of the detector
XYP body5      30.0
"""

material_str = \
"""
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
MATERIAL         4.0               1.848       5.0                    BERYLLIU
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
"""

detector_str = """
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
* e+e- and gamma production threshold set at 10 MeV
EMFCUT        -0.010     0.010       1.0  BERYLLIU                    PROD-CUT
* score in each region energy deposition and stars produced by primaries
SCORE       ENERGY    BEAMPART 
* Boundary crossing fluence in the middle of the target (log intervals, one-way)
USRBDX          99.0   NEUTRON     -47.0    det1    det2        400.  NeuFlu
USRBDX         1.0     0.00001     +50.0                   0.0       10.0  &
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
RANDOMIZE        1.0
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
START       100000.0
STOP
"""


if __name__ == '__main__':
    # freg = FlukaRegistry()
    #
    # rpp = RPP("RPP_BODY", 0, 10, 0, 10, 0, 10, flukaregistry=freg)
    # ma = Material("BERYLLIU", 4, 1.848)
    # z = Zone()
    # z.addIntersection(rpp)
    # region = Region("RPP_REG")
    # region.addZone(z)
    # freg.addMaterial(ma)
    # freg.assignma(ma, region)
    # freg.addRegion(region)
    #
    # # greg = convert.fluka2Geant4(freg)
    # # greg.getWorldVolume().clipSolid()
    # #
    # # v = vi.VtkViewer()
    # # v.addAxes(length=20)
    # # v.addLogicalVolume(greg.getWorldVolume())
    # # v.view()
    #
    # f = Writer()
    # f.addDetector(freg)
    # f.write('/root/model.inp')

    y_range = (-10, 10)
    z_range = (0, 20)
    y_dim = 8
    z_dim = 8
    delta_y = (y_range[1] - y_range[0]) / y_dim
    delta_z = (z_range[1] - z_range[0]) / z_dim
    yy = np.linspace(y_range[0], y_range[1] - delta_y, y_dim)
    zz = np.linspace(z_range[0], z_range[1] - delta_z, z_dim)
    yy, zz = np.meshgrid(yy, zz)

    delta_y = (y_range[1] - y_range[0]) / y_dim
    delta_z = (z_range[1] - z_range[0]) / z_dim
    design_mask = np.random.uniform(size=(y_dim, z_dim)) > 0.5

    yy = yy.reshape(-1)
    zz = zz.reshape(-1)
    design_mask = design_mask.reshape(-1)

    body_str = ""

    region_str = "reg1  5  +body1  -body2\n" + \
                 "reg2  5  +body2  -body3\n" + \
                 "det1  5  +body4  +body5\n" +\
                 "det2  5  +body4  -body5\n"
    for i in range(len(design_mask)):
        body_str += f"RPP  b{i+1}  -10.0  +10.0  {yy[i]}  {yy[i] + delta_y}  {zz[i]}  {zz[i] + delta_z}\n"
        region_str += f"vr{i+1}  5  +body3  +b{i+1}\n"

        if design_mask[i] == True:
            material_str += f"ASSIGNMAT  BERYLLIU  vr{i+1}\n"
        else:
            material_str += f"ASSIGNMAT  VACUUM    vr{i+1}\n"

    body_str += "END\n"
    region_str += "END\n"
    geom_str = body_str + region_str + "GEOEND\n"
    material_str += "ASSIGNMAT  BLCKHOLE  reg1\n"
    material_str += "ASSIGNMAT  VACUUM    reg2\n"
    material_str += "ASSIGNMAT  VACUUM    det1\n"
    material_str += "ASSIGNMAT  VACUUM    det2\n"

    f_name = "/root/neutron-source.inp"
    f = open(f_name, "w")

    f.write(beam_str + geom_str + material_str + detector_str)

    f.close()
