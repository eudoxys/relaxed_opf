# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Same as L{t_case9_opfv2} with addition of DC line data.
Please see L{caseformat} for details on the case file format.

@return: Power flow data for 9 bus, 3 generator case, with OPF
and DC line data.
@see: L{toggle_dcline}, L{idx_dcline}.
"""

from numpy import array

case = {
    "version": 2,
    "baseMVA": 100.0,
    "bus": array([
        # BUS_I BUS_TYPE    PD  PQ GS BS BUS_AREA VM VA BASE_KV ZONE VMAX VMIN LAM_P LAM_Q MU_VMAX MU_VMIN
        [     1,       3,    0,  0, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
        [     2,       2,    0,  0, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
        [    30,       2,    0,  0, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
        [     4,       1,    0,  0, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
        [     5,       1,   90, 30, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
        [     6,       1,    0,  0, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
        [     7,       1,  100, 35, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
        [     8,       1,    0,  0, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
        [     9,       1,  125, 50, 0, 0,       1, 1, 0,    345,   1, 1.1, 0.9],
    ]),
    "gen": array([
        # GEN_BUS   PG QG QMAX  QMIN VG MBASE GEN_STATUS PMAX PMIN  PC1  PC2 QC1MAX QC1MIN QC2MAX QC2MIN RAMP_AGC RAMP_10 RAMP_30 RAMP_Q APF MU_PMAX MU_PMIN MU_QMAX MU_QMIN
        [       1,   0, 0, 300, -300, 1,  100,         1, 250,   0,   0,   0,     0,     0,     0,     0,       0,      0,      0,     0,  0],
        [       2, 163, 0, 300, -300, 1,  100,         1, 300,   0,   0, 200,   -20,    20,   -10,    10,       0,      0,      0,     0,  0],
        [      30,  85, 0, 300, -300, 1,  100,         1, 270,   0,   0, 200,   -30,    30,   -15,    15,       0,      0,      0,     0,  0],
    ]),
    "branch": array([
        # F_BUS T_BUS    BR_R    BR_X   BR_B RATE_A RATE_B RATE_C TAP SHIFT BR_STATUS ANGMIN ANGMAX PF QF PT QT MU_SF MU_ST MU_ANGMAX MU_ANGMIN
        [     1,    4,      0, 0.0576,     0,     1,   250,   250,  0,    0,        1,  -360,  2.48],
        [     4,    5, 0.0170, 0.0920, 0.158,     1,   250,   250,  0,    0,        1,  -360,   360],
        [     5,    6, 0.0390, 0.1700, 0.358,   150,   150,   150,  0,    0,        1,  -360,   360],
        [    30,    6,      0, 0.0586,     0,     1,   300,   300,  0,    0,        1,  -360,   360],
        [     6,    7, 0.0119, 0.1008, 0.209,    40,   150,   150,  0,    0,        1,  -360,   360],
        [     7,    8, 0.0085, 0.0720, 0.149,   250,   250,   250,  0,    0,        1,  -360,   360],
        [     8,    2,      0, 0.0625,     0,   250,   250,   250,  0,    0,        1,  -360,   360],
        [     8,    9, 0.0320, 0.1610, 0.306,   250,   250,   250,  0,    0,        1,  -360,   360],
        [     9,    4, 0.0100, 0.0850, 0.176,   250,   250,   250,  0,    0,        1,    -2,   360],
    ]),
    "gencost": array([
        # MODEL STARTUP SHUTDOWN  N   COST0   COST1 COST2 COST3 COST4 COST5 COST6 COST7
        [     1,      0,       0, 4,      0,      0,  100, 2500, 200,  5500,  250, 7250],
        [     2,      0,       0, 2, 24.035, -403.5,    0,    0,   0,     0,    0,    0],
        [     1,      0,       0, 3,      0,      0,  200, 3000, 300,  5000,    0,    0],
    ]),
    "dcline": array([
        # F_BUS T_BUS BR_STATUS PF PT QF QT    VF    VT PMIN PMAX QMINF QMAXF QMINT QMAXT LOSS0 LOSS1
        [    30,    4,        1, 0, 0, 0, 0, 1.01,    1,   0,  10,  -10,   10,  -10,   10,    1, 0.01],
        [     7,    9,        0, 0, 0, 0, 0,    1,    1,   0,  10,    0,    0,    0,    0,    0,    0],
        [     5,    8,        1, 0, 0, 0, 0,    1,    1,   0,  10,  -10,   10,  -10,   10,    0,    0],
        [     5,    9,        1, 0, 0, 0, 0,    1, 0.98,   0,  10,  -10,   10,  -10,   10,    0, 0.05],
    ]),
    "dclinecost": array([
        # MODEL STARTUP SHUTDOWN  N COST0 COST1 COST2 COST3 COST4 COST5 COST6 COST7
        [     2,      0,       0, 2,    0,    0,    0,    0,    0,    0,    0,    0],
        [     2,      0,       0, 2,    0,    0,    0,    0,    0,    0,    0,    0],
        [     2,      0,       0, 2,    0,    0,    0,    0,    0,    0,    0,    0],
        [     2,      0,       0, 2,  7.3,    0,    0,    0,    0,    0,    0,    0],
    ])
}

def case9dc():
    return case
