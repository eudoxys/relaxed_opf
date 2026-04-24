"""Powerflow solvers"""

from copy import deepcopy as copy

import numpy as np
import pandas as pd
import cvxpy as cp

from pypower.ppoption import ppoption
from pypower.runpf import runpf
from pypower.runopf import runopf
from pypower import idx_bus as bus
from pypower import idx_brch as branch
from pypower import idx_gen as gen
from pypower import idx_cost as cost

def full_acpf(case:dict,**kwargs) -> dict:
    """Solve full AC powerflow"""
    solution = runpf(case,ppoption(**kwargs))[0]["order"]["int"]
    ref = [n for n, bt in enumerate(solution["bus"][:, bus.BUS_TYPE]) if bt == 3]
    solution["bus"][:,bus.VA] = (solution["bus"][:,bus.VA] - solution["bus"][:,bus.VA][ref[0]])
    return {
        "case": copy(case),
        "ok": True,
        "status": "solved",
        "solution": solution
    }

def full_acopf(case:dict,**kwargs) -> dict:
    """Solve full AC optimal powerflow"""
    result = runopf(case,ppoption(**kwargs))
    solution = copy(case)
    data = result["var"]["val"]
    solution["gen"][:,gen.PG] = data["Pg"]
    solution["gen"][:,gen.QG] = data["Qg"][0]
    solution = runpf(solution,ppoption(**kwargs))[0]["order"]["int"]
    ref = [n for n, bt in enumerate(solution["bus"][:, bus.BUS_TYPE]) if bt == 3]
    solution["bus"][:,bus.VA] = (data["Va"] - data["Va"][ref[0]])*180/np.pi
    return {
        "case": copy(case),
        "ok": True,
        "status": "solved",
        "solution": solution
    }

def decoupled_acopf(
    data:dict,
    # costs:dict[str,float]=None,
    **options,
    ) -> dict:
    """Solve decoupled optimal powerflow problem
    
    Arguments
    ---------

    - `data`: `pypower` case data

    - `**options`: `cvxpy` solver options

    Returns
    -------

    - `dict`: solution results include the following:

      - `case`: a copy of the original problem data (see `pypower.casedata`)
      - `status`: status of the solution (see `cvxpy.Solve`)
      - `value`: value of the objection function (see `cvxpy.Solve`)
      - `objective`: objective function
      - `constraints`: constraints list
      - `problem`: cvxpy problem data (see `cvxpy.Problem`)
      - `solution`: solved case data (see `pypower.casedata`)
      - `parameters`: problem parameters (dict)
      - `variables`: problem variables (dict)
      - `ok`: valid solution obtained flag (boolean)

      In addition the following are included is the problem is feasible:
      
      - `pf`: real power flow on branches
      - `qf`: reactive power flow on branches
      - `vm`: bus voltage magnitudes
      - `va`: bus voltage angles
      - `pg`: real power generation dispatch
      - `qg`: reactive power generation dispatch
    """
    
    # default options
    if "canon_backend" not in options:
        options["canon_backend"] = "SCIPY"
    # if costs is None:
    #     costs = { # all costs per-unit generation cost $/MWh
    #         "curtailment": 10.0, # $/MVA
    #     }

    # dimensions
    N = len(data["bus"])
    M = len(data["branch"])
    K = len(data["gen"])

    # per-unit system
    puS = data["baseMVA"]

    # branch parameters
    s = cp.Parameter(shape=(M,1),value=data["branch"][:,[branch.RATE_A]]/puS,name="s",nonneg=True) # line flow limits
    b = cp.Parameter(shape=(M,N),value=np.zeros((M,N)), name="b") # line susceptances
    f = cp.Parameter(shape=(N,M),value=np.zeros((N,M)), name="f") # bus line flow injections
    bi = {i: n for n, i in enumerate(data["bus"][:, bus.BUS_I])}  # bus index
    # TODO: vectorize these transformations using sparce matrices
    for n,br in enumerate(data["branch"]):
        i = bi[br[branch.F_BUS]]
        j = bi[br[branch.T_BUS]]
        tap = br[branch.TAP]
        bx = br[branch.BR_STATUS] / br[branch.BR_X] / (tap if tap > 0 else 1)
        shift = br[branch.SHIFT]*np.pi/180
        b.value[n,i] = bx
        b.value[n,j] = - ( bx + shift )
        f.value[i,n] = 1
        f.value[j,n] = -1

    # bus parameters
    vl = cp.Parameter(shape=(N,1), value=data["bus"][:, [bus.VMIN]], name="vl", nonneg=True) # voltage lower limit
    vu = cp.Parameter(shape=(N,1), value=data["bus"][:, [bus.VMAX]], name="vu", nonneg=True) # voltage upper limit
    pd = cp.Parameter(shape=(N,1), value=data["bus"][:, [bus.PD]] / puS, name="pd") # load real power
    qd = cp.Parameter(shape=(N,1), value=data["bus"][:, [bus.QD]] / puS, name="qd") # load reactive power

    # gen parameters
    pl = cp.Parameter(shape=(N,1), value=np.zeros((N,1)), name="pl", nonneg=True) # real power lower limit
    pu = cp.Parameter(shape=(N,1), value=np.zeros((N,1)), name="ph", nonneg=True) # real power upper limit
    ql = cp.Parameter(shape=(N,1), value=np.zeros((N,1)), name="ql") # reactive power lower limit
    qu = cp.Parameter(shape=(N,1), value=np.zeros((N,1)), name="qh") # reactive power upper limit
    vg = cp.Parameter(shape=(K,1), value=data["gen"][:,[gen.VG]], name="vg") # bus voltage setpoints

    # variables
    pf = cp.Variable((M,1), name="p")  # line real power flows
    qf = cp.Variable((M,1), name="q")  # line reactive power flows
    vm = cp.Variable((N,1), name="|v|", nonneg=True)  # voltage magnitudes
    va = cp.Variable((N,1), name="𝞱")  # voltage angles
    pg = cp.Variable((N,1), name="pg", nonneg=True)  # generator real power dispatch
    qg = cp.Variable((N,1), name="qg")  # generator reactive power dispatch
    # pc = cp.Variable(shape=(N,1), name="pc", nonneg=True) # real power demand curtailment
    # qc = cp.Variable(shape=(N,1), name="qc", nonneg=True) # reactive power demand curtailment

    # gen parameters
    gi = [bi[n] for n in data["gen"][:,gen.GEN_BUS]]
    pl.value[gi] = data["gen"][:,gen.GEN_STATUS] * data["gen"][:,gen.PMIN].T[0] / puS
    pu.value[gi] = data["gen"][:,gen.GEN_STATUS] * data["gen"][:,gen.PMAX].T[0] / puS
    ql.value[gi] = data["gen"][:,gen.GEN_STATUS] * data["gen"][:,gen.QMIN].T[0] / puS
    qu.value[gi] = data["gen"][:,gen.GEN_STATUS] * data["gen"][:,gen.QMAX].T[0] / puS

    # setup Feasible Sets
    ref = [n for n, bt in enumerate(data["bus"][:, bus.BUS_TYPE]) if bt == 3]

    # cost function
    cost = cp.sum(pg**2+qg**2)
    # cost += costs["curtailment"] * cp.sum ( pc**2 + qc**2 ) # curtailment cost

    # constraints
    constraints = [

        # Feasible Set 2
        pf == b @ va, # Equation (1a)
        qf == b @ vm, # Equation (1b)
        f @ pf + pd == pg, # Equation (2a)
        f @ qf + qd == qg, # Equation (2b)
        # f @ pf + pd - pc == pg, # Equation (2a) with load curtailment
        # f @ qf + qd - qc == qg, # Equation (2b) with load curtailment

        # Feasible Set 4
        pl <= pg, pg <= pu, # Equation (3a)
        ql <= qg, qg <= qu, # Equation (3b)
        cp.abs(pf) <= s, # Equation (4a)
        vl <= vm, vm <= vu, # Equation (5a)

        # practical constraints not specified in the mathematical model
        va[ref] == 0,  # reference bus angle is always 0
        vm[gi] == vg,  # bus voltage setpoints
        cp.abs(va) <= 0.175,  # +/- 10 degrees for decoupling assumptions to be valid

        # curtailment constraints
        # pc <= pd, # real power curtailment
        # qd <= qd, # reactive power curtailment
    ]

    # problem statement
    objective = cp.Minimize(cost)
    problem = cp.Problem(objective,constraints)
    problem.solve(**options)

    # solution results
    result = {
        "ok": False,
        "case": copy(data),
        "status": problem.status,
        "value": np.round(problem.value,4),
        "problem": problem,
        "objective": objective,
        "constraints": constraints,
        "parameters": {
            "s (pu.MVA)": s.value.T[0],
            "b (pm.S)": b.value,
            "f (pu)": f.value,
            "vl (pu.kV)": vl.value.T[0],
            "vu (pu.kV)": vu.value.T[0],
            "pd (pu.MW)": pd.value.T[0],
            "qd (pu.MVAr)": qd.value.T[0],
            "pl (pu.MW)": pl.value.T[0],
            "pu (pu.MW)": pu.value.T[0],
            "ql (pu.MVAr)": ql.value.T[0],
            "qu (pu.MVAr)": qu.value.T[0],
            "vg (pu.kV)": vg.value.T[0],
        },
        "solution": {},
        "violations": {},
    }
    if va.value is not None:
        result["variables"] = {
            "pf (pu.MW)": (pf.value).round(4).T[0],
            "qf (pu.MVAr)": (qf.value).round(4).T[0],
            "vm (pu.kV)": (vm.value).round(4).T[0],
            "va (deg)": (va.value*180/np.pi).round(4).T[0],
            "pg (pu.MW)": (pg.value).round(4).T[0],
            "qg (pu.MVAr)": (qg.value).round(4).T[0],
        }

        solution = copy(data)
        
        solution["bus"][:,bus.VA] = (va.value.T[0] - va.value.T[0][ref[0]]) * 180 / np.pi
        solution["bus"][:,bus.VM] = vm.value.T[0]
        # solution["bus"][:,bus.PD] = (pd.value-pc.value).T[0] * puS
        # solution["bus"][:,bus.QD] = (qd.value-qc.value).T[0] * puS

        solution["branch"][:,branch.PF] = pf.value.T[0] * puS
        solution["branch"][:,branch.QF] = qf.value.T[0] * puS

        n = [bi[x] for x in solution["gen"][:,gen.GEN_BUS]]
        solution["gen"][:,gen.PG] = pg.value[n,0] * puS
        solution["gen"][:,gen.QG] = qg.value[n,0] * puS

        result["solution"] = solution
        checks = violations(solution,formatter=dict)
        if checks:
            result["violations"] = checks
        result["ok"] = True

    return result

def decoupled_acosp(
    data:dict,
    costs:dict[str,float]=None,
    margin:float=0.15,
    **options) -> dict:
    """Solve decoupled optimal sizing/placement problem
    
    Arguments
    ---------

    - `data`: `pypower` case data

    - `costs`: capacity addition costs (per-unit generation cost)
      Valid costs are
      - `"capacitor"`: cost of adding a capacitor (default is 0.1)
      - `"condensor"`: cost of adding a condensor (default is 1.0)
      - `"transformer"`: cost of increasing transformer capacity (default is 2.0)
      - `"powerline"`: cost of increasing powerline capacity (default is 10.0)

    - `margin`: load margin for sizing

    - `**options`: `cvxpy` solver options

    Returns
    -------

    - `dict`: solution results include the following:

      - `case`: a copy of the original problem data (see `pypower.casedata`)
      - `status`: status of the solution (see `cvxpy.Solve`)
      - `value`: value of the objection function (see `cvxpy.Solve`)
      - `objective`: objective function
      - `constraints`: constraints list
      - `problem`: cvxpy problem data (see `cvxpy.Problem`)
      - `solution`: solved case data (see `pypower.casedata`)
      - `parameters`: problem parameters (dict)
      - `variables`: problem variables (dict)
      - `ok`: valid solution obtained flag (boolean)

      In addition the following are included is the problem is feasible:
      
      - `pf`: real power flow on branches
      - `qf`: reactive power flow on branches
      - `vm`: bus voltage magnitudes
      - `va`: bus voltage angles
      - `pg`: real power generation dispatch
      - `qg`: reactive power generation dispatch
      - `ac`: capacitors/condensors additions
      - `ap`: real power generation capacity expansions
      - `aq`: reactive power generation capacity expansions
      - `al`: transformer and powerline capacity expansions
    """
    
    # default options
    if "canon_backend" not in options:
        options["canon_backend"] = "SCIPY"
    default_costs = { 
            # all costs per-unit generation cost $/MW
            "capacitor": 0.1, # $/MVAr
            "condensor": 1.0, # $/MVAr
            "transformer": 2.0, # $/MVA
            "powerline": 5.0, # $/MVA
        }
    if costs is None:
        costs = default_costs
    for key,value in default_costs.items():
        if key not in costs:
            costs[key] = value

    # dimensions
    N = len(data["bus"])
    M = len(data["branch"])
    K = len(data["gen"])

    # per-unit system
    puS = data["baseMVA"]

    # branch parameters
    s = cp.Parameter(shape=(M,1),value=data["branch"][:,[branch.RATE_A]]/puS,name="s",nonneg=True) # line flow limits
    b = cp.Parameter(shape=(M,N),value=np.zeros((M,N)), name="b") # line susceptances
    f = cp.Parameter(shape=(N,M),value=np.zeros((N,M)), name="f") # bus line flow injections
    bi = {i: n for n, i in enumerate(data["bus"][:, bus.BUS_I])}  # bus index
    # TODO: vectorize these transformations using sparce matrices
    for n,br in enumerate(data["branch"]):
        i = bi[br[branch.F_BUS]]
        j = bi[br[branch.T_BUS]]
        tap = br[branch.TAP]
        bx = 1 / br[branch.BR_X] / (tap if tap > 0 else 1)
        shift = br[branch.SHIFT]*np.pi/180
        b.value[n,i] = bx
        b.value[n,j] = - ( bx + shift )
        f.value[i,n] = 1
        f.value[j,n] = -1

    # bus parameters
    vl = cp.Parameter(shape=(N,1), value=data["bus"][:, [bus.VMIN]], name="vl", nonneg=True) # voltage lower limit
    vu = cp.Parameter(shape=(N,1), value=data["bus"][:, [bus.VMAX]], name="vu", nonneg=True) # voltage upper limit
    pd = cp.Parameter(shape=(N,1), value=data["bus"][:, [bus.PD]] / puS, name="pd") # load real power
    qd = cp.Parameter(shape=(N,1), value=data["bus"][:, [bus.QD]] / puS, name="qd") # load reactive power

    # gen parameters
    pl = cp.Parameter(shape=(N,1), value=np.zeros((N,1)), name="pl", nonneg=True) # real power lower limit
    pu = cp.Parameter(shape=(N,1), value=np.zeros((N,1)), name="ph", nonneg=True) # real power upper limit
    ql = cp.Parameter(shape=(N,1), value=np.zeros((N,1)), name="ql") # reactive power lower limit
    qu = cp.Parameter(shape=(N,1), value=np.zeros((N,1)), name="qh") # reactive power upper limit
    vg = cp.Parameter(shape=(K,1), value=data["gen"][:,[gen.VG]], name="vg") # bus voltage setpoints

    # variables
    pf = cp.Variable((M,1), name="p")  # line real power flows
    qf = cp.Variable((M,1), name="q")  # line reactive power flows
    vm = cp.Variable((N,1), name="|v|", nonneg=True)  # voltage magnitudes
    va = cp.Variable((N,1), name="𝞱")  # voltage angles
    pg = cp.Variable((N,1), name="pg", nonneg=True)  # generator real power dispatch
    qg = cp.Variable((N,1), name="qg")  # generator reactive power dispatch

    # softening constraints
    ac = cp.Variable(shape=(N,1), name="ac") # capacitor/condensor additions
    ap = cp.Variable(shape=(N,1), name="ap", nonneg=True) # generator real power additions
    aq = cp.Variable(shape=(N,1), name="aq", nonneg=True) # generator reactive power additions
    al = cp.Variable(shape=(M,1), name="al", nonneg=True) # powerline/transformer capacity additions

    # gen parameters
    gi = [bi[n] for n in data["gen"][:,gen.GEN_BUS]]
    pl.value[gi] = data["gen"][:,gen.PMIN].T[0] / puS
    pu.value[gi] = data["gen"][:,gen.PMAX].T[0] / puS
    ql.value[gi] = data["gen"][:,gen.QMIN].T[0] / puS
    qu.value[gi] = data["gen"][:,gen.QMAX].T[0] / puS

    # setup Feasible Sets
    ref = [n for n, x in enumerate(data["bus"][:, bus.BUS_TYPE]) if x == 3] # reference bus(ses)
    nongen = list(set(range(N)) - set(gi)) # non-generation busses
    powerlines = [n for n,x in enumerate(data["branch"][:,branch.TAP]) if x == 0]
    transformers = list(set(range(M))- set(powerlines))

    # cost function
    cost = cp.sum(ap) # + cp.sum(aq)/10 # generation capacity costs
    cost += cp.sum( # capacity/condensor costs
            ( costs["capacitor"] - costs["condensor"] ) * ac / 2
            + ( costs["capacitor"] + costs["condensor"] ) * cp.abs(ac) / 2
            )
    cost += costs["powerline"] * cp.sum(al[powerlines]) # powerline costs
    cost += costs["transformer"] * cp.sum(al[transformers]) # transformer costs

    # constraints
    constraints = [

        # Feasible Set 2
        pf == b @ va, # Equation (1a)
        qf == b @ vm, # Equation (1b)
        f @ pf + pd*(1+margin) == pg, # Equation (2c)
        f @ qf + qd*(1+margin) + ac == qg, # Equation (2d)

        # Feasible Set 4
        pl <= pg, pg <= pu + ap, # Equation (3c)
        ql - aq <= qg, qg <= qu + aq, # Equation (3d)
        cp.abs(pf) <= s + al, # Equation (4b)
        vl <= vm, vm <= vu, # Equation (5b)

        # practical constraints not specified in the mathematical model
        va[ref] == 0,  # reference bus angle is always 0
        vm[gi] == vg,  # bus voltage setpoints
        cp.abs(va) <= 0.175,  # +/- 10 degrees for decoupling assumptions to be valid

        # constraints on addition placements
        ac[gi] == 0, # no capacitors/condensors at generation busses
        ap[nongen] == 0, # no new real power generation at non-generation busses
        aq[nongen] == 0, # no new reactive power generation at non-generation busses
        
        # limits on reactive power additions relative to real power additions
        ql - aq >= - ( pg + ap ), qu + aq <= pg + ap, # lower and upper bounds on generation additions
    ]

    # problem statement
    objective = cp.Minimize(cost)
    problem = cp.Problem(objective,constraints)
    problem.solve(**options)

    # solution results
    result = {
        "ok": False,
        "case": copy(data),
        "status": problem.status,
        "value": np.round(problem.value,4),
        "problem": problem,
        "objective": objective,
        "constraints": constraints,
        "parameters": {
            "s (pu.MVA)": s.value.T[0],
            "b (pm.S)": b.value,
            "f (pu)": f.value,
            "vl (pu.kV)": vl.value.T[0],
            "vu (pu.kV)": vu.value.T[0],
            "pd (pu.MW)": pd.value.T[0],
            "qd (pu.MVAr)": qd.value.T[0],
            "pl (pu.MW)": pl.value.T[0],
            "pu (pu.MW)": pu.value.T[0],
            "ql (pu.MVAr)": ql.value.T[0],
            "qu (pu.MVAr)": qu.value.T[0],
            "vg (pu.kV)": vg.value.T[0],
        },
        "solution": {},
        "violations": {},
    }
    if va.value is not None:
        result["variables"] = {
            "pf (pu.MW)": (pf.value).round(4).T[0],
            "qf (pu.MVAr)": (qf.value).round(4).T[0],
            "vm (pu.kV)": (vm.value).round(4).T[0],
            "va (deg)": (va.value*180/np.pi).round(4).T[0],
            "pg (pu.MW)": (pg.value).round(4).T[0],
            "qg (pu.MVAr)": (qg.value).round(4).T[0],
            "ac (pu.MVAr)": (ac.value).round(4).T[0],
            "ap (pu.MVAr)": (ap.value).round(4).T[0],
            "aq (pu.MVAr)": (aq.value).round(4).T[0],
            "al (pu.MVAr)": (al.value).round(4).T[0],
        }

        solution = copy(data)
        
        solution["bus"][:,bus.VA] = va.value.T[0] * 180 / np.pi
        solution["bus"][:,bus.VM] = vm.value.T[0]
        solution["bus"][:,bus.BS] = solution["bus"][:,bus.BS] + ac.value.T[0]

        solution["branch"][:,branch.BR_STATUS] = np.ones(M)
        solution["branch"][:,branch.PF] = pf.value.T[0] * puS
        solution["branch"][:,branch.QF] = qf.value.T[0] * puS
        solution["branch"][:,branch.RATE_A] = solution["branch"][:,branch.RATE_A] + al.value.T[0] * puS

        n = [bi[x] for x in solution["gen"][:,gen.GEN_BUS]]
        solution["gen"][:,gen.GEN_STATUS] = np.ones(K)
        solution["gen"][:,gen.PG] = pg.value[n,0] * puS
        solution["gen"][:,gen.QG] = qg.value[n,0] * puS
        solution["gen"][:,gen.PMAX] = solution["gen"][:,gen.PMAX] + ap.value[n,0] * puS
        solution["gen"][:,gen.QMIN] = solution["gen"][:,gen.QMIN] - aq.value[n,0] * puS
        solution["gen"][:,gen.QMAX] = solution["gen"][:,gen.QMAX] + aq.value[n,0] * puS

        result["solution"] = solution
        checks = violations(solution,formatter=dict)
        if checks:
            result["violations"] = checks
        result["ok"] = True

    return result

def violations(data, precision=3, formatter=None):
    """Enumerate violations in case"""
    if formatter is None:
        formatter = as_table
    result = {"bus": [], "gen": [], "branch": []}
    if "bus" in data:
        for n, v in enumerate(
            data["bus"][:, (bus.VM, bus.VA, bus.VMIN, bus.VMAX)].round(precision)
        ):
            VM, VA, VMIN, VMAX = map(float, v)
            if not VMIN <= VM <= VMAX:
                result["bus"].append((n, f"{VM=} pu.V outside ({VMIN=},{VMAX=})"))
    if "gen" in data:
        for n, g in enumerate(
            data["gen"][
                :, (gen.PG, gen.QG, gen.PMIN, gen.PMAX, gen.QMIN, gen.QMAX)
            ].round(precision)
        ):
            PG, QG, PMIN, PMAX, QMIN, QMAX = map(float, g)
            if PMIN < PMAX and not PMIN <= PG <= PMAX:
                result["gen"].append((n, f"{PG=} MW outside ({PMIN=},{PMAX=})"))
            if QMIN < QMAX and not QMIN <= QG <= QMAX:
                result["gen"].append((n, f"{QG=} MVAr outside ({QMIN=},{QMAX=})"))
    if "branch" in data and data["branch"].shape[1] >= branch.PF:
        for n, b in enumerate(data["branch"][:,[branch.PF,branch.RATE_A]]):
            PF, RATE_A = map(float, np.abs(b))
            if RATE_A > 0 and PF > RATE_A*1.001:
                result["branch"].append((n, f"|PF|={PF:.1f} MVA outside (0,{RATE_A=:.1f})"))
    if formatter:
        return formatter(result)
    return result

def as_frames(data,showall=False,**kwargs):
    """Return case data as dataframes"""
    if not kwargs and showall is False:
        kwargs = dict(bus="BUS_I,PD,QD,BS,VM,VA,VMAX,VMIN",
                      branch="F_BUS,T_BUS,BR_STATUS,BR_X,RATE_A,PF,QF",
                      gen="GEN_BUS,GEN_STATUS,PG,QG,PMIN,PMAX,QMIN,QMAX",
                     )
    columns = {
        "bus":"BUS_I,BUS_TYPE,PD,QD,GS,BS,BUS_AREA,VM,VA,BASE_KV,ZONE,VMAX,VMIN,LAM_P,LAM_Q,MU_VMAX,MY_VMIN",
        "branch": "F_BUS,T_BUS,BR_R,BR_X,BR_B,RATE_A,RATE_B,RATE_C,TAP,SHIFT,BR_STATUS,ANGMIN,ANGMAX,PF,QF,PT,QT,MU_SF,MU_ST,MU_ANGMAX,MU_ANGMIN",
        "gen":"GEN_BUS,PG,QG,QMAX,QMIN,VG,MBASE,GEN_STATUS,PMAX,PMIN,PC1,PC2,QC1MIN,QC1MAX,QC2MIN,QC2MAX,RAMP_AGC,RAMP_10,RAMP_30,RAMP_Q,APF,MU_PMAX,MU_PMIN,MU_QMAX,MU_QMIN",
        "gencost":"MODEL,STARTUP,SHUTDOWN,N,COST0,COST1,COST2",
    }
    return {x:pd.DataFrame(
            data[x].round(3),
            columns=y.split(",")[:data[x].shape[1]],
        )[kwargs[x].split(",") if x in kwargs else y.split(",")[:data[x].shape[1]]] for x,y in columns.items() if x in data}

def as_mdtable(violations):
    """Format violation results as a table"""
    result = []
    for key, values in violations.items():
        result.append(f"| **{key.title()}** | Violation(s) |")
        result.append("| ---: | :--- |")
        if values:
            for n, m in values:
                result.append(f"| {n}| {m} |")
        else:
            result.append(f"| None | |")
        result.append("")
    return "\n".join(result)

def as_table(violations):
    """Format violation results as a table"""
    data = []
    colwidth = [7,0]
    for key, values in violations.items():
        if values:
            for n, m in values:
                p = f"{key}[{n}]"
                colwidth = [max(len(p),colwidth[0]),max(len(m),colwidth[1])]
                data.append([p,m])
    if not data:
        return "No violations"
    result = [
        "ELEMENT" + " "*(colwidth[0]-7) + "  VIOLATION",
        # " ".join(['-'*x for x in colwidth])
    ]
    for key,value in data:
        result.append(key + "  "*(colwidth[0]-len(key)) + " " + value)
    return "\n".join(result)

def internals(case):
    """Generation solver internals in a readable format"""
    def dump(x):
        if isinstance(x,dict):
            return("\n\n  ".join([f"{x}:\n    {str(y).replace('\n','\n    ')}" for x,y in x.items()]))
        else:
            return str(x).replace("\n","  \n")

    return "\n".join([f"\n{x}\n{'-'*len(x)}\n\n  {dump(y)}" for x,y in case.items() if x in ["problem","parameters","variables"]])
    
if __name__ == '__main__':
    
    from case4r import case
    
    pd.options.display.max_columns = None
    pd.options.display.width = None

    ppoptions = dict(VERBOSE=0,OUT_ALL=0)
    
    print("*****************")
    print("*** BASE CASE ***")
    print("*****************\n")
    print(*[f"{x}:\n{y}\n" for x,y in as_frames(case).items()],sep="\n")
    print(violations(case))

    print("\n*************************")
    print("*** FULL AC POWERFLOW ***")
    print("*************************\n")
    initial_acpf = full_acpf(case,**ppoptions)
    print("STATUS:",initial_acpf["status"])
    print(*[f"{x}:\n{y}\n" for x,y in as_frames(initial_acpf["solution"]).items()],sep="\n")
    if initial_acpf["ok"]: print(violations(initial_acpf["solution"]))

    print("\n*********************************")
    print("*** FULL AC OPTIMAL POWERFLOW ***")
    print("*********************************\n")
    initial_acopf = full_acopf(case,**ppoptions)
    print("STATUS:",initial_acopf["status"])
    print(*[f"{x}:\n{y}\n" for x,y in as_frames(initial_acopf["solution"]).items()],sep="\n")
    if initial_acopf["ok"]: print(violations(initial_acopf["solution"]))

    print("\n************************************")
    print("*** DECOUPLED AC OPTIMAL POWERFLOW ***")
    print("**************************************\n")
    fast_acopf = decoupled_acopf(case)
    print("STATUS:",fast_acopf["status"])
    print(*[f"{x}:\n{y}\n" for x,y in as_frames(fast_acopf["solution"]).items()],sep="\n")
    if fast_acopf["ok"]: 
        print(violations(fast_acopf["solution"]))
    
    else:

        print("\n***********************************")
        print("*** DECOUPLED AC OPTIMAL SIZING ***")
        print("***********************************\n")
        fast_acosp = decoupled_acosp(case)
        print("STATUS:",fast_acosp["status"],f"(cost={fast_acosp["value"]})")
        print(*[f"{x}:\n{y}\n" for x,y in as_frames(fast_acosp["solution"]).items()],sep="\n")
        print(violations(fast_acosp["solution"]))
        if not fast_acosp["ok"]:
            print(internals(fast_acosp))
        
        else:

            # print(internals(fast_acosp))
            print("\n**********************************")
            print("*** FINAL AC OPTIMAL POWERFLOW ***")
            print("**********************************\n")
            final_acopf = full_acopf(fast_acosp["solution"],**ppoptions)
            print("STATUS:",final_acopf["status"])
            print(*[f"{x}:\n{y}\n" for x,y in as_frames(final_acopf["solution"]).items()],sep="\n")
            if final_acopf["ok"]: print(violations(final_acopf["solution"]))

