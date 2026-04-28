"""Powerflow solvers"""

import os
from copy import deepcopy as copy
import importlib
import warnings
from time import time

import numpy as np
import scipy as sp
import pandas as pd
import cvxpy as cp

from pypower.ppoption import ppoption
from pypower.runpf import runpf
from pypower.runopf import runopf
from pypower import idx_bus as bus
from pypower import idx_brch as branch
from pypower import idx_gen as gen
from pypower import idx_cost as cost

def load(case:dict,name:str=None) -> dict:
    """Fix case to include branch line flows"""
    module = importlib.import_module(case)
    if name is None:
        name = os.path.splitext(os.path.basename(case))[0]
        if not hasattr(module,name) and hasattr(module,"case"):
            name = "case"
    if callable(getattr(module,name)):
        case = getattr(module,name)()
    else:
        case = getattr(module,name)

    N,M = case["bus"].shape

    N,M = case["branch"].shape
    if M < branch.QT:
        case["branch"] = np.hstack([case["branch"],np.zeros((N,branch.QT-M+1))])

    return case

def full_acpf(case:dict,**kwargs) -> dict:
    """Solve full AC powerflow"""
    tic = time()
    result,ok = runpf(case,ppoption(**kwargs))
    if ok:
        solution = result["order"]["int"]
        ref = [n for n, bt in enumerate(result["bus"][:, bus.BUS_TYPE]) if bt == 3]
        solution["bus"][:,bus.VA] = (solution["bus"][:,bus.VA] - solution["bus"][:,bus.VA][ref[0]])
        result = {
            "case": copy(case),
            "ok": True,
            "status": "solved",
            "solution": result,
        }
    else:
        result = {
            "case": copy(case),
            "ok": False,
            "status": "failed",
            "result": result,
            "time": toc-tic,
        }
    toc = time()
    result["time"] = toc-tic
    return result

def full_acopf(case:dict,**kwargs) -> dict:
    """Solve full AC optimal powerflow"""
    tic = time()
    result = runopf(case,ppoption(**kwargs))
    if result["success"]:
        solution = copy(case)
        data = result["var"]["val"]
        solution["gen"][:,gen.PG] = data["Pg"]
        solution["gen"][:,gen.QG] = data["Qg"][0]
        solution = runpf(solution,ppoption(**kwargs))[0]["order"]["int"]
        ref = [n for n, bt in enumerate(solution["bus"][:, bus.BUS_TYPE]) if bt == 3]
        solution["bus"][:,bus.VA] = (data["Va"] - data["Va"][ref[0]])*180/np.pi
        result = {
            "case": copy(case),
            "ok": True,
            "status": "solved",
            "solution": solution,
        }
    else:
        result = {
            "case": copy(case),
            "ok": False,
            "status": result["raw"]["output"]["message"],
            "result": result,
        }
    toc = time()
    result["time"] = toc-tic
    return result

def decoupled_acopf(
    data:dict,
    curtailment:float=None,
    **options,
    ) -> dict:
    """Solve decoupled optimal powerflow problem
    
    Arguments
    ---------

    - `data`: `pypower` case data

    - `curtailment`: cost of curtailment per-unit generation cost
      (None disables curtailment)

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
      - `pc`: real power curtailment (if any)
      - `qc`: reactive power curtailment (if any)

    Caveats
    -------

    Placing a generator at a reference bus (`BUS_TYPE==3`) may result in OPF
    solution that violate generation limits at the bus. A warning is
    emitted by OSP when this is possible.
    """
    
    tic = time()

    # default options
    if "canon_backend" not in options:
        options["canon_backend"] = "SCIPY"

    # dimensions
    N = len(data["bus"])
    M = len(data["branch"])
    K = len(data["gen"])

    # per-unit system
    puS = data["baseMVA"]

    # bus parameters
    bb = data["bus"]
    vl = cp.Parameter(shape=(N,1), value=bb[:, [bus.VMIN]], name="vl", nonneg=True) # voltage lower limit
    vu = cp.Parameter(shape=(N,1), value=bb[:, [bus.VMAX]], name="vu", nonneg=True) # voltage upper limit
    pd = cp.Parameter(shape=(N,1), value=bb[:, [bus.PD]] / puS, name="pd") # load real power
    qd = cp.Parameter(shape=(N,1), value=bb[:, [bus.QD]] / puS, name="qd") # load reactive power

    # branch parameters
    br = data["branch"] # branch data
    bi = {i: n for n, i in enumerate(bb[:, bus.BUS_I])}  # bus index (i is not necessarily reasonable)
    f_bus = [bi[x] for x in br[:,branch.F_BUS]]
    t_bus = [bi[x] for x in br[:,branch.T_BUS]]
    tap = br[:,branch.TAP]
    tap[np.where(tap==0)] = 1.0
    shift = br[:,branch.SHIFT] * np.pi / 180
    br_status = br[:,branch.BR_STATUS]
    br_x = br[:,branch.BR_X]
    b = sp.sparse.coo_matrix((br_status/br_x/tap,(range(M),f_bus)),shape=(M,N)).todense() \
        - sp.sparse.coo_matrix((br_status/br_x/tap+shift,(range(M),t_bus)),shape=(M,N)).todense() 
    b = cp.Parameter(shape=(M,N),value=b, name="b") # line susceptances
    f = sp.sparse.coo_matrix((br_status,(range(M),f_bus)),shape=(M,N)).todense().T \
        - sp.sparse.coo_matrix((br_status,(range(M),t_bus)),shape=(M,N)).todense().T 
    f = cp.Parameter(shape=(N,M),value=f, name="f") # bus line flow injections
    s = cp.Parameter(shape=(M,1),value=br[:,[branch.RATE_A]]/puS,name="s",nonneg=True) # line flow limits

    # gen parameters
    gg = data["gen"]
    gi = np.array([bi[n] for n in gg[:,gen.GEN_BUS]])
    vg = cp.Parameter(shape=(K,1), value=gg[:,[gen.VG]], name="vg") # bus voltage setpoints
    pl = cp.Parameter(shape=(K,1), value=gg[:,[gen.PMIN]], name="pl") # real power minimum
    pu = cp.Parameter(shape=(K,1), value=gg[:,[gen.PMAX]], name="pu") # real power maximum
    ql = cp.Parameter(shape=(K,1), value=gg[:,[gen.QMIN]], name="ql") # reactive power minimum
    qu = cp.Parameter(shape=(K,1), value=gg[:,[gen.QMAX]], name="qu") # reactive power maximum
    g = cp.Parameter(shape=(N,K),value=sp.sparse.coo_matrix((np.ones(K),(list(range(K)),gi)),shape=(K,N)).todense().T,name="g") # sum generators to busses

    # variables
    pf = cp.Variable((M,1), name="p")  # line real power flows
    qf = cp.Variable((M,1), name="q")  # line reactive power flows
    vm = cp.Variable((N,1), name="|v|", nonneg=True)  # voltage magnitudes
    va = cp.Variable((N,1), name="𝞱")  # voltage angles
    pg = cp.Variable((K,1), name="pg", nonneg=True)  # generator real power dispatch
    qg = cp.Variable((K,1), name="qg")  # generator reactive power dispatch
    if not curtailment is None:
        pc = cp.Variable(shape=(N,1), name="pc", nonneg=True) # real power demand curtailment
        qc = cp.Variable(shape=(N,1), name="qc") # reactive power demand curtailment

    # reference busses
    ref = [n for n, bt in enumerate(data["bus"][:, bus.BUS_TYPE]) if bt == 3]

    # cost function
    # cost = cp.sum(pg**2+qg**2) # TODO: replace with generation costs
    # if curtailment:
    #     cost += curtailment * cp.sum ( pc**2 + qc**2 ) # curtailment cost
    cost = 0

    # constraints
    constraints = [

        # Feasible Set 2
        pf == b @ va, # Equation (1a)
        qf == b @ vm, # Equation (1b)

        # Feasible Set 4
        pl <= pg, pg <= pu, # Equation (3a)
        ql <= qg, qg <= qu, # Equation (3b)
        cp.abs(pf) <= s, # Equation (4a)
        vl <= vm, vm <= vu, # Equation (5a)

        # practical constraints not specified in the mathematical model
        va[ref] == 0,  # reference bus angles are always 0
        vm[gi] == vg,  # bus voltage setpoints
        cp.abs(va) <= 0.175,  # +/- 10 degrees for decoupling assumptions to be valid
    ]
    if curtailment is None:
        constraints += [
            f @ pf + pd == g @ pg, # Equation (2a)
            f @ qf + qd == g @ qg, # Equation (2b)
        ]
    else:
        constraints += [
            f @ pf + pd - pc == g @ pg, # Equation (2a) with load curtailment
            f @ qf + qd - qc == g @ qg, # Equation (2b) with load curtailment

            pc <= pd, # real power curtailment limits
            cp.minimum(qc,0) >= cp.minimum(qd,0), # reactive power curtailment lower limits
            cp.maximum(qc,0) <= cp.maximum(qd,0), # reactive power curtailment upper limits
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
            "g (pu)": g.value,
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
    if problem.status == "optimal":
        result["variables"] = {
            "pf (pu.MW)": pf.value.T[0],
            "qf (pu.MVAr)": qf.value.T[0],
            "vm (pu.kV)": vm.value.T[0],
            "va (deg)": (va.value*180/np.pi).T[0],
            "pg (pu.MW)": pg.value.T[0],
            "qg (pu.MVAr)": qg.value.T[0],
        }
        if not curtailment is None:
            result["variables"]["pc (pu.MW)"] = pc.value.round(4).T[0]
            result["variables"]["qc (pu.MVAr)"] = qc.value.round(4).T[0]

        solution = copy(data)
        
        # update bus data
        bb = solution["bus"]
        bb[:,[bus.VA]] = ( va.value - va.value.T[0][ref[0]]) * 180 / np.pi
        bb[:,[bus.VM]] = vm.value
        if not curtailment is None:
            bb[:,[bus.PD]] = ( pd.value - pc.value ) * puS
            bb[:,[bus.QD]] = ( qd.value - qc.value ) * puS

        # update branch data
        bb = solution["branch"]
        bb[:,[branch.PF]] = pf.value * puS
        bb[:,[branch.QF]] = qf.value * puS
        bb[:,[branch.PT]] = 0
        bb[:,[branch.QT]] = 0

        # update gen
        gg = solution["gen"]
        gg[:,[gen.PG]] = pg.value * puS
        gg[:,[gen.QG]] = qg.value * puS

        result["solution"] = solution
        checks = violations(solution,formatter=dict)
        if checks:
            result["violations"] = checks
        result["ok"] = True

    toc = time()
    result["time"] = toc-tic
    return result

def decoupled_acosp(
    data:dict,
    costs:dict[str,float]=None,
    margin:float=0.15,
    allin:bool=True,
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

    - `allin`: enable use of all available resources

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
    
    tic = time()

    # default options
    if "canon_backend" not in options:
        options["canon_backend"] = "SCIPY"

    # default costs, if needed
    default_costs = { 
            # all costs per-unit generation cost $/MW
            "capacitor": 0.1, # $/MVAr
            "condensor": 1.0, # $/MVAr
            "transformer": 2.0, # $/MVA
            "powerline": 5.0, # $/MVA
        }
    if costs is None:
        costs = default_costs
    else:
        for key,value in default_costs.items():
            if key not in costs:
                costs[key] = value

    # dimensions
    N = len(data["bus"])
    M = len(data["branch"])
    K = len(data["gen"])

    # per-unit system
    puS = data["baseMVA"]

   # bus parameters
    bb = data["bus"]
    vl = cp.Parameter(shape=(N,1), value=bb[:, [bus.VMIN]], name="vl", nonneg=True) # voltage lower limit
    vu = cp.Parameter(shape=(N,1), value=bb[:, [bus.VMAX]], name="vu", nonneg=True) # voltage upper limit
    pd = cp.Parameter(shape=(N,1), value=bb[:, [bus.PD]]/puS, name="pd") # load real power
    qd = cp.Parameter(shape=(N,1), value=bb[:, [bus.QD]]/puS, name="qd") # load reactive power

    # branch parameters
    br = data["branch"] # branch data
    bi = {i: n for n, i in enumerate(bb[:, bus.BUS_I])}  # bus index (i is not necessarily reasonable)
    f_bus = [bi[x] for x in br[:,branch.F_BUS]]
    t_bus = [bi[x] for x in br[:,branch.T_BUS]]
    tap = br[:,branch.TAP]
    tap[np.where(tap==0)] = 1.0
    shift = br[:,branch.SHIFT] * np.pi / 180
    br_status = np.ones(M) if allin else br[:,branch.BR_STATUS] 
    br_x = br[:,branch.BR_X]
    b = sp.sparse.coo_matrix((br_status/br_x/tap,(range(M),f_bus)),shape=(M,N)).todense() \
        - sp.sparse.coo_matrix((br_status/br_x/tap+shift,(range(M),t_bus)),shape=(M,N)).todense() 
    b = cp.Parameter(shape=(M,N),value=b, name="b") # line susceptances
    f = sp.sparse.coo_matrix((br_status,(range(M),f_bus)),shape=(M,N)).todense().T \
        - sp.sparse.coo_matrix((br_status,(range(M),t_bus)),shape=(M,N)).todense().T 
    f = cp.Parameter(shape=(N,M),value=f, name="f") # bus line flow injections
    s = cp.Parameter(shape=(M,1),value=br[:,[branch.RATE_A]]/puS,name="s",nonneg=True) # line flow limits

    # gen parameters
    gg = data["gen"]
    gi = np.array([bi[n] for n in gg[:,gen.GEN_BUS]])
    gs = 1 if allin else gg[:,gen.GEN_STATUS]
    vg = cp.Parameter(shape=(K,1), value=gg[:,[gen.VG]], name="vg") # bus voltage setpoints
    pl = cp.Parameter(shape=(K,1), value=gs*gg[:,[gen.PMIN]]/puS, name="pl") # real power minimum
    pu = cp.Parameter(shape=(K,1), value=gs*gg[:,[gen.PMAX]]/puS, name="pu") # real power maximum
    ql = cp.Parameter(shape=(K,1), value=gs*gg[:,[gen.QMIN]]/puS, name="ql") # reactive power minimum
    qu = cp.Parameter(shape=(K,1), value=gs*gg[:,[gen.QMAX]]/puS, name="qu") # reactive power maximum
    g = cp.Parameter(shape=(N,K),value=sp.sparse.coo_matrix((np.ones(K),(list(range(K)),gi)),shape=(K,N)).todense().T,name="g") # sum generators to busses

    # variables
    pf = cp.Variable((M,1), name="p")  # line real power flows
    qf = cp.Variable((M,1), name="q")  # line reactive power flows
    vm = cp.Variable((N,1), name="|v|", nonneg=True)  # voltage magnitudes
    va = cp.Variable((N,1), name="𝞱")  # voltage angles
    pg = cp.Variable((K,1), name="pg", nonneg=True)  # generator real power dispatch
    qg = cp.Variable((K,1), name="qg")  # generator reactive power dispatch

    # softened constraint variables
    ac = cp.Variable(shape=(N,1), name="ac") # capacitor/condensor additions
    ap = cp.Variable(shape=(K,1), name="ap", nonneg=True) # generator real power additions
    aq = cp.Variable(shape=(K,1), name="aq", nonneg=True) # generator reactive power additions
    al = cp.Variable(shape=(M,1), name="al", nonneg=True) # powerline/transformer capacity additions

    # setup Feasible Sets
    ref = [n for n, x in enumerate(data["bus"][:, bus.BUS_TYPE]) if x == 3] # reference bus(ses)
    nongen = list(set(range(N)) - set(gi)) # non-generation busses
    powerlines = [n for n,x in enumerate(data["branch"][:,branch.TAP]) if x == 0]
    transformers = list(set(range(M))- set(powerlines))
    for gr in list(set(gi) & set(ref)):
        warnings.warn(f"generator at reference bus[{gr}] may result in unexpected violations")

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
        f @ pf + pd*(1+margin) == g @ pg, # Equation (2c)
        f @ qf + qd*(1+margin) + ac == g @ qg, # Equation (2d)

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
        
        # limits on reactive power additions relative to real power additions
        # ql - aq >= - ( pg + ap ), qu + aq <= pg + ap, # lower and upper bounds on generation additions
        qu + aq <= pu + ap,
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
    if problem.status == "optimal":

        result["variables"] = {
            "pf (pu.MW)": pf.value.T[0],
            "qf (pu.MVAr)": qf.value.T[0],
            "vm (pu.kV)": vm.value.T[0],
            "va (deg)": (va.value*180/np.pi).T[0],
            "pg (pu.MW)": pg.value.T[0],
            "qg (pu.MVAr)": qg.value.T[0],
            "ac (pu.MVAr)": ac.value.T[0],
            "ap (pu.MVAr)": ap.value.T[0],
            "aq (pu.MVAr)": aq.value.T[0],
            "al (pu.MVAr)": al.value.T[0],
        }

        # creates solution
        solution = copy(data)
        
        # bus updates
        solution["bus"][:,bus.VA] = va.value.T[0] * 180 / np.pi
        solution["bus"][:,bus.VM] = vm.value.T[0]
        solution["bus"][:,bus.BS] = solution["bus"][:,bus.BS] + ac.value.T[0]

        # branch updates
        if allin:
            solution["branch"][:,branch.BR_STATUS] = np.ones(M)
        solution["branch"][:,[branch.PF]] = pf.value * puS
        solution["branch"][:,[branch.QF]] = qf.value * puS
        ratio = al.value * puS/solution["branch"][:,[branch.RATE_A]] + 1
        columns = [branch.RATE_A,branch.RATE_B,branch.RATE_C] # raised values
        solution["branch"][:,columns] = solution["branch"][:,columns] * ratio
        columns = [branch.BR_R,branch.BR_X,branch.BR_B] # lowered values
        solution["branch"][:,columns] = solution["branch"][:,columns] / ratio
        
        # generator updates
        if allin:
            solution["gen"][:,gen.GEN_STATUS] = np.ones(K)
        solution["gen"][:,[gen.PG]] = pg.value * puS
        solution["gen"][:,[gen.QG]] = qg.value * puS
        solution["gen"][:,[gen.PMAX]] = solution["gen"][:,[gen.PMAX]] + ap.value * puS
        solution["gen"][:,[gen.QMIN]] = solution["gen"][:,[gen.QMIN]] - aq.value * puS
        solution["gen"][:,[gen.QMAX]] = solution["gen"][:,[gen.QMAX]] + aq.value * puS

        result["solution"] = solution

        # TODO create update list
        updates = []
        # ac
        # ap/aq
        # al
        result["updates"] = updates

        # create violations list
        checks = violations(solution,formatter=dict)
        if checks:
            result["violations"] = checks
        
        result["ok"] = True

    toc = time()
    result["time"] = toc - tic
    return result

def violations(data, precision=3, formatter=None):
    """Enumerate violations in case"""
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
    match formatter:
        case None:
            return result
        case "counter":
            return sum([len(x) for x in result.values()])
        case "table":
            return as_table(result)
        case "_":
            return formatter(result)

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
        # "gencost":"MODEL,STARTUP,SHUTDOWN,N,COST0,COST1,COST2",
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
    
    import re

    results = {}

    testlist = [os.path.splitext(x)[0]  for x in os.listdir() if x.startswith("case") and x.endswith(".py")]
    for name in sorted(sorted(testlist),key=lambda x:int(re.match("[^0-9]*([0-9]+)",x).group(1))):

        tic = time()

        print(f"Processing {name}",end="...",flush=True)
        with open(f"{name}.txt","w") as fh:
            case = load(name)
            results[name] = {
                "Base case": "-",
                "Initial fast OPF": "-",
                "Initial full PF": "-",
                "Fast OSP": "-",
                "Final fast OPF": "-",
                "Final full PF": "-",
            }
            
            pd.options.display.max_columns = None
            pd.options.display.width = None
            pd.options.display.max_rows = None

            ppoptions = dict(VERBOSE=0,OUT_ALL=0)
            
            print("*****************",file=fh)
            print("*** BASE CASE ***",file=fh)
            print("*****************\n",file=fh)
            initial_case = full_acpf(case,**ppoptions)
            print("STATUS:",initial_case["status"],file=fh)
            print(f"TIME: {initial_case['time']:.3f} s",file=fh)
            results[name]["Base case"] = initial_case["status"]
            print(*[f"{x}:\n{y}\n" for x,y in as_frames(initial_case["solution"]).items()],sep="\n",file=fh)
            if initial_case["ok"]: 
                print(violations(initial_case["solution"],formatter="table"),file=fh)
                results[name]["Base case"] = "violations" if violations(initial_case["solution"],formatter="counter") else "ok"

            print("\n**************************************",file=fh)
            print("*** DECOUPLED AC OPTIMAL POWERFLOW ***",file=fh)
            print("**************************************\n",file=fh)
            fast_acopf = decoupled_acopf(case)
            print("STATUS:",fast_acopf["status"],file=fh)
            print(f"TIME: {fast_acopf['time']:.3f} s",file=fh)
            results[name]["Initial fast OPF"] = fast_acopf["status"]
            print(*[f"{x}:\n{y}\n" for x,y in as_frames(fast_acopf["solution"]).items()],sep="\n",file=fh)
            if fast_acopf["ok"]: 
                print(violations(fast_acopf["solution"],formatter="table"),file=fh)
                results[name]["Initial fast OPF"] = "violations" if violations(fast_acopf["solution"],formatter="counter") else "ok"

                print("\n*************************",file=fh)
                print("*** FULL AC POWERFLOW ***",file=fh)
                print("*************************\n",file=fh)
                initial_acpf = full_acpf(fast_acopf["solution"] if fast_acopf["ok"] else case,**ppoptions)
                print("STATUS:",initial_acpf["status"],file=fh)
                print(f"TIME: {initial_acpf['time']:.3f} s",file=fh)
                results[name]["Initial full PF"] = initial_acpf["status"]
                print(*[f"{x}:\n{y}\n" for x,y in as_frames(initial_acpf["solution"]).items()],sep="\n",file=fh)
                if initial_acpf["ok"]: 
                    print(violations(initial_acpf["solution"],formatter="table"),file=fh)
                    results[name]["Initial full PF"] = "violations" if violations(initial_acpf["solution"],formatter="counter") else "ok"

            print("\n***********************************",file=fh)
            print("*** DECOUPLED AC OPTIMAL SIZING ***",file=fh)
            print("***********************************\n",file=fh)
            fast_acosp = decoupled_acosp(case)
            print("STATUS:",fast_acosp["status"],f"(cost={fast_acosp["value"]})",file=fh)
            print(f"TIME: {fast_acosp['time']:.3f} s",file=fh)
            results[name]["Fast OSP"] = fast_acosp["status"]
            print(*[f"{x}:\n{y}\n" for x,y in as_frames(fast_acosp["solution"]).items()],sep="\n",file=fh)
            print(violations(fast_acosp["solution"],formatter="table"),file=fh)
            if not fast_acosp["ok"]:
                print(internals(fast_acosp),file=fh)
            
            else:

                results[name]["Fast OSP"] = "violations" if violations(fast_acosp["solution"],formatter="counter") else "ok"
                
                print("\n**************************",file=fh)
                print("*** FINAL DECOUPLED OPF ***",file=fh)
                print("**************************\n",file=fh)
                final_opf = decoupled_acopf(fast_acosp["solution"])
                print("STATUS:",final_opf["status"],file=fh)
                print(f"TIME: {final_opf['time']:.3f} s",file=fh)
                results[name]["Final fast OPF"] = final_opf["status"]
                if final_opf["ok"]:
                    print(*[f"{x}:\n{y}\n" for x,y in as_frames(final_opf["solution"]).items()],sep="\n",file=fh)
                    results[name]["Final fast OPF"] = "ok"
                if final_opf["ok"]: 
                    print(violations(final_opf["solution"],formatter="table"),file=fh)
                    results[name]["Final fast OPF"] = "violations" if violations(final_opf["solution"],formatter="counter") else "ok"

                print("\n**************************",file=fh)
                print("*** FINAL AC POWERFLOW ***",file=fh)
                print("**************************\n",file=fh)
                final_acpf = full_acpf(fast_acosp["solution"],**ppoptions)
                print("STATUS:",final_acpf["status"],file=fh)
                print(f"TIME: {final_acpf['time']:.3f} s",file=fh)
                results[name]["Final full PF"] = final_acpf["status"]
                if final_acpf["ok"]:
                    print(*[f"{x}:\n{y}\n" for x,y in as_frames(final_acpf["solution"]).items()],sep="\n",file=fh)
                    results[name]["Final full PF"] = "ok"
                if final_acpf["ok"]: 
                    print(violations(final_acpf["solution"],formatter="table"),file=fh)
                    results[name]["Final full PF"] = "violations" if violations(final_acpf["solution"],formatter="counter") else "ok"

        print(f"done in {time()-tic:.1f} s")
    print(pd.DataFrame(results).T)