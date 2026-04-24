import numpy as np
case = {
    "version": '2',
    "baseMVA": 100.0,
    "bus": np.array([
        # BUS_I BUS_TYPE    PD    PQ GS BS BUS_AREA VM VA BASE_KV ZONE VMAX VMIN LAM_P LAM_Q MU_VMAX MU_VMIN
        [     1,       3,    0,    0, 0, 0,       1, 1, 0,    230,   1, 1.1, 0.9],
        [     2,       1,    0,    0, 0, 0,       1, 1, 0,    230,   1, 1.1, 0.9],
        [     3,       1, 50.0,  5.0, 0, 0,       1, 1, 0,    230,   1, 1.1, 0.9],
        [     4,       2, 50.0,  5.0, 0, 0,       1, 1, 0,    230,   1, 1.1, 0.9]
        ]),
    "branch": np.array([
        # F_BUS T_BUS     BR_R    BR_X    BR_B RATE_A RATE_B RATE_C TAP SHIFT BR_STATUS ANGMIN ANGMAX PF QF PT QT MU_SF MU_ST MU_ANGMAX MU_ANGMIN
        [     1,    2, 0.01008, 0.0504, 0.1025,   25,   250,   250,  0,    0,        1,  -360,   360, 0, 0, 0, 0,    0,    0,        0,        0],
        [     1,    3, 0.00744, 0.0372, 0.0775,   25,   250,   250,  0,    0,        1,  -360,   360, 0, 0, 0, 0,    0,    0,        0,        0],
        [     2,    3, 0.00744, 0.0372, 0.0775,   25,   250,   250,  0,    0,        0,  -360,   360, 0, 0, 0, 0,    0,    0,        0,        0],
        [     2,    4, 0.00744, 0.0372, 0.0775,   25,   250,   250,  0,    0,        1,  -360,   360, 0, 0, 0, 0,    0,    0,        0,        0],
        [     3,    4, 0.01272, 0.0636, 0.1275,   25,   250,   250,  1,    0,        1,  -360,   360, 0, 0, 0, 0,    0,    0,        0,        0]
        ]),
    "gen": np.array([
        # GEN_BUS   PG QG QMAX  QMIN VG MBASE GEN_STATUS PMAX PMIN PC1 PC2 QC1MAX QC1MIN QC2MAX QC2MIN RAMP_AGC RAMP_10 RAMP_30 RAMP_Q APF MU_PMAX MU_PMIN MU_QMAX MU_QMIN
        [       4,   0, 0,   5,   -5, 1,  100,         1, 100,   0,  0,  0,     0,     0,     0,     0,       0,      0,      0,     0,  0],
    ]),
    "gencost": np.array([
        # MODEL STARTUP SHUTDOWN  N COST0 COST1 COST2
        [     2,    0.0,     0.0, 3, 0.04, 20.0,  0.0],
        ]),
    }