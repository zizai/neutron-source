import os
import subprocess
import uuid
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sim.float_box import FloatBox
from sim.int_box import IntBox

BEAM_PART = \
"""
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
TITLE
Neutron fluence after a proton-irradiated Be target
*....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8....
GLOBAL       10000.
BEAM           -0.1       0.2       0.0     -2.36     -1.18       1.0 PROTON
BEAMPOS         0.0       0.0     -50.0
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

MATERIAL_PART = \
"""
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
MATERIAL         4.0               1.848       5.0                    BERYLLIU
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
"""

DETECTOR_PART = \
"""
*....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
* score in each region energy deposition and stars produced by primaries
SCORE       NEUTRON 
* Boundary crossing fluence in the middle of the target (log intervals, one-way)
USRBIN         10.0   NEUTRON      25.0      10.0      10.0      30.2 NeuFlu
USRBIN        -10.0     -10.0      30.0     100.0     100.0       1.0 &
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
RANDOMIZE        1.0
*...+....1....+....2....+....3....+....4....+....5....+....6....+....7....+....8
START       100000.0
STOP
"""

WORK_DIR = '/root/flukawork/'


def create_fluka_inp(f_name, design_mask, plane='xy'):
    beam_str = BEAM_PART
    material_str = MATERIAL_PART
    detector_str = DETECTOR_PART

    if plane == 'xy':
        x_range = (-10, 10)
        y_range = (-10, 10)
        x_dim = design_mask.shape[0]
        y_dim = design_mask.shape[1]
        delta_x = (x_range[1] - x_range[0]) / x_dim
        delta_y = (y_range[1] - y_range[0]) / y_dim
        xx = np.linspace(x_range[0], x_range[1] - delta_x, x_dim)
        yy = np.linspace(y_range[0], y_range[1] - delta_y, y_dim)
        xx, yy = np.meshgrid(xx, yy)

        delta_x = (x_range[1] - x_range[0]) / x_dim
        delta_y = (y_range[1] - y_range[0]) / y_dim

        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        design_mask = design_mask.reshape(-1)

        body_str = ""

        region_str = "reg1  5  +body1  -body2\n" + \
                     "reg2  5  +body2  -body3\n" + \
                     "det1  5  +body4  +body5\n" + \
                     "det2  5  +body4  -body5\n"
        for i in range(len(design_mask)):
            body_str += f"RPP  b{i+1}  {xx[i]}  {xx[i] + delta_x}  {yy[i]}  {yy[i] + delta_y}  0.0  20.0\n"
            region_str += f"vr{i+1}  5  +body3  +b{i+1}\n"
            if design_mask[i] == True:
                material_str += f"ASSIGNMAT  BERYLLIU  vr{i+1}\n"
            else:
                material_str += f"ASSIGNMAT  VACUUM    vr{i+1}\n"
    elif plane == 'yz':
        y_range = (-10, 10)
        z_range = (0, 20)
        y_dim = design_mask.shape[0]
        z_dim = design_mask.shape[1]
        delta_y = (y_range[1] - y_range[0]) / y_dim
        delta_z = (z_range[1] - z_range[0]) / z_dim
        yy = np.linspace(y_range[0], y_range[1] - delta_y, y_dim)
        zz = np.linspace(z_range[0], z_range[1] - delta_z, z_dim)
        yy, zz = np.meshgrid(yy, zz)

        delta_y = (y_range[1] - y_range[0]) / y_dim
        delta_z = (z_range[1] - z_range[0]) / z_dim

        yy = yy.reshape(-1)
        zz = zz.reshape(-1)
        design_mask = design_mask.reshape(-1)

        body_str = ""

        region_str = "reg1  5  +body1  -body2\n" + \
                     "reg2  5  +body2  -body3\n" + \
                     "det1  5  +body4  +body5\n" + \
                     "det2  5  +body4  -body5\n"
        for i in range(len(design_mask)):
            body_str += f"RPP  b{i + 1}  -10.0  +10.0  {yy[i]}  {yy[i] + delta_y}  {zz[i]}  {zz[i] + delta_z}\n"
            region_str += f"vr{i + 1}  5  +body3  +b{i + 1}\n"
            if design_mask[i] == True:
                material_str += f"ASSIGNMAT  BERYLLIU  vr{i+1}\n"
            else:
                material_str += f"ASSIGNMAT  VACUUM    vr{i+1}\n"
    else:
        raise NotImplementedError

    body_str += "END\n"
    region_str += "END\n"
    geom_str = body_str + region_str + "GEOEND\n"
    material_str += "ASSIGNMAT  BLCKHOLE  reg1\n"
    material_str += "ASSIGNMAT  VACUUM    reg2\n"
    material_str += "ASSIGNMAT  VACUUM    det1\n"
    material_str += "ASSIGNMAT  VACUUM    det2\n"

    f = open(f_name, "w")

    f.write(beam_str + geom_str + material_str + detector_str)

    f.close()


class NeutronSourceEnv(object):
    def __init__(self, action_dim=1, max_steps=64, rew_scale=50., work_dir=None):
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.action_space = IntBox(0, 1, shape=(action_dim,))
        # self.observation_space = IntBox(0, 1, shape=(max_steps,))
        self.observation_space = FloatBox(-1, 1, shape=(max_steps * 5,))
        self.geometry = np.ones((max_steps, max_steps))
        self.steps = 0
        self.rew_scale = rew_scale
        self.work_dir = WORK_DIR + uuid.uuid4().__str__() if work_dir is None else work_dir

    def get_obs(self, action):
        d = self.max_steps * 2
        k = 2 * np.pi * np.random.normal(0, 1, (2, d))
        pos = np.stack(np.meshgrid(np.linspace(0, 1, num=self.max_steps, endpoint=False), np.linspace(0, 1, num=self.max_steps + 1, endpoint=True)), -1)
        pos = pos[self.steps]
        action = np.tile(action.T, (self.max_steps, 1))
        return np.concatenate([action, np.sin(pos @ k), np.cos(pos @ k)], -1)

    def step(self, action):
        self.geometry[self.steps] = np.squeeze(action, -1)
        self.steps += 1

        rew = - np.sum(1 - action) * 0.01
        if self.steps == self.max_steps:
            self.run_sim()
            data = self.read_data()
            in_count = np.sum(data[20:-20, 20:-20])
            out_count = np.sum(data) - np.sum(data[20:-20, 20:-20])
            rew += (in_count - out_count * 0.1) * self.rew_scale

        obs = self.get_obs(action)
        done = self.steps == self.max_steps

        return obs, rew, done, {}

    def reset(self):
        action = np.zeros((self.max_steps, 1))
        self.steps = 0
        obs = self.get_obs(action)
        return obs

    def run_sim(self):
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)

        os.chdir(self.work_dir)

        f_name = self.work_dir + '/neutron-source.inp'
        create_fluka_inp(f_name, self.geometry)

        return subprocess.run(['/usr/local/fluka/bin/rfluka', '-M 1', f_name], capture_output=True)

    def read_data(self):
        f_name = self.work_dir + '/neutron-source001_fort.25'
        f = open(f_name, 'r')

        data = []
        for i, line in enumerate(f):
            if i >= 16:
                data.append(np.fromstring(line, sep='  '))

        data = np.concatenate(data)
        data = data.reshape(100, 100)

        plt.imshow(self.geometry)
        plt.savefig(self.work_dir + '/neutron-source.png')

        plt.imshow(data)
        plt.savefig(self.work_dir + '/neutron-fluence.png')
        return data


if __name__ == '__main__':
    env = NeutronSourceEnv()
    np.random.seed(47)

    done = False
    while not done:
        action = np.random.uniform(size=(64, 1)) > 0.5
        obs, rew, done, _ = env.step(action)

    print(obs)
    print(rew)
