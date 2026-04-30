"""Powerflow solvers"""

import os
from copy import deepcopy as copy
import importlib
from warnings import warn
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
    try:
        result,ok = runpf(copy(case),ppoption(**kwargs))
        if ok:
            solution = {x:y for x,y in result.items() if x in case}
            ref = [n for n, bt in enumerate(result["bus"][:, bus.BUS_TYPE]) if bt == 3]
            solution["bus"][:,bus.VA] = (solution["bus"][:,bus.VA] - solution["bus"][:,bus.VA][ref[0]])
            result = {
                "case": copy(case),
                "ok": True,
                "status": "solved",
                "warnings": [],
                "solution": result,
            }
        else:
            result = {
                "case": copy(case),
                "ok": False,
                "status": "failed",
                "warnings": [],
                "result": result,
            }
    except Exception as err:
        result = {
            "case": copy(case),
            "ok": False,
            "status": "exception",
            "warnings": [],
            "message": str(err),
        }

    toc = time()
    result["time"] = round(toc-tic,3)
    return result

def full_acopf(case:dict,**kwargs) -> dict:
    """Solve full AC optimal powerflow"""
    tic = time()
    try:
        result = runopf(copy(case),ppoption(**kwargs))
        if result["success"]:
            solution = {x:y for x,y in result.items() if x in case}
            result = {
                "case": copy(case),
                "ok": True,
                "status": "solved",
                "warnings": [],
                "solution": solution,
            }
        else:
            result = {
                "case": copy(case),
                "ok": False,
                "status": result["raw"]["output"]["message"],
                "warnings": [],
                "result": result,
            }
    except Exception as err:
        result = {
            "case": copy(case),
            "ok": False,
            "status": "exception",
            "warnings": [],
            "message": str(err),
        }

    toc = time()
    result["time"] = round(toc-tic,3)
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
    """
    
    tic = time()

    # default options
    if "canon_backend" not in options:
        options["canon_backend"] = "SCIPY"

    # model check
    assert "baseMVA" in data, "missing baseMVA value"
    assert "bus" in data, "missing bus array"
    assert "branch" in data, "missing branch array"
    assert "gen" in data, "missing gen array"

    # dimensions
    N = len(data["bus"])
    M = len(data["branch"])
    K = len(data["gen"])

    # per-unit system
    puS = data["baseMVA"]

    # bus parameters
    bb = np.array(data["bus"])
    vl = cp.Constant(value=bb[:, [bus.VMIN]], name="vl") # voltage lower limit
    vu = cp.Constant(value=bb[:, [bus.VMAX]], name="vu") # voltage upper limit
    pd = cp.Parameter(shape=(N,1), value=bb[:, [bus.PD]] / puS, name="pd") # load real power
    qd = cp.Parameter(shape=(N,1), value=bb[:, [bus.QD]] / puS, name="qd") # load reactive power
    bi = {i: n for n, i in enumerate(bb[:, bus.BUS_I])}  # bus index (i is not necessarily reasonable)

    # branch parameters
    br = data["branch"] # branch data

    f_bus = [bi[x] for x in br[:,branch.F_BUS]]
    t_bus = [bi[x] for x in br[:,branch.T_BUS]]
    
    tap = br[:,[branch.TAP]]
    tap[np.where(tap==0)] = 1.0 # non-zero is only for transformers, zero is powerline (unity tap)
    err = np.where(tap<0)[0]
    assert len(err) == 0, f"bus[{err},TAP] < 0"
    
    shift = br[:,[branch.SHIFT]] * np.pi / 180
    
    br_status = br[:,[branch.BR_STATUS]]
    err = np.where([x for x in br_status.flatten() if x not in [0,1]])[0]
    assert len(err)==0, f"bus[{err},BR_STATUS] value is not in [0,1]"
    
    br_x = br[:,[branch.BR_X]]
    err = np.where(br_x==0)[0]
    assert len(err) == 0, f"bus[{err},BR_X] <= 0"

    x = br_status/br_x/tap

    b = sp.sparse.coo_matrix((x.flatten(),(range(M),f_bus)),shape=(M,N)) \
        - sp.sparse.coo_matrix(((x+shift).flatten(),(range(M),t_bus)),shape=(M,N)) 
    b = cp.Constant(value=b, name="b") # line susceptances

    f = sp.sparse.coo_matrix((br_status.flatten(),(range(M),f_bus)),shape=(M,N)).T \
        - sp.sparse.coo_matrix((br_status.flatten(),(range(M),t_bus)),shape=(M,N)).T 
    f = cp.Constant(value=f, name="f") # line connections

    s = br[:,[branch.RATE_A]]/puS
    s = cp.Parameter(shape=(M,1),value=s,name="s") # line flow limits

    # gen parameters
    gg = np.array(data["gen"])
    gi = np.array([bi[n] for n in gg[:,gen.GEN_BUS]])
    vg = cp.Constant(value=gg[:,[gen.VG]], name="vg") # bus voltage setpoints
    pl = cp.Constant(value=gg[:,[gen.PMIN]]/puS, name="pl") # real power minimum
    pu = cp.Constant(value=gg[:,[gen.PMAX]]/puS, name="pu") # real power maximum
    ql = cp.Constant(value=gg[:,[gen.QMIN]]/puS, name="ql") # reactive power minimum
    qu = cp.Constant(value=gg[:,[gen.QMAX]]/puS, name="qu") # reactive power maximum
    g = cp.Constant(value=sp.sparse.coo_matrix((np.ones(K),(list(range(K)),gi)),shape=(K,N)).T,name="g") # sum generators to busses

    # variables
    pf = cp.Variable((M,1), name="pf")  # line real power flows
    qf = cp.Variable((M,1), name="qf")  # line reactive power flows
    vm = cp.Variable((N,1), name="|v|", nonneg=True)  # voltage magnitudes
    va = cp.Variable((N,1), name="𝞱")  # voltage angles
    pg = cp.Variable((K,1), name="pg", nonneg=True)  # generator real power dispatch
    qg = cp.Variable((K,1), name="qg")  # generator reactive power dispatch
    if not curtailment is None:
        pc = cp.Variable(shape=(N,1), name="pc", nonneg=True) # real power demand curtailment
        qc = cp.Variable(shape=(N,1), name="qc") # reactive power demand curtailment

    # sanity checks
    warnings = []

    # bus vl/vu range
    if min(vu.value - vl.value) < 0 or min(vl.value) < 0.8 or max(vu.value) > 1.2:
        for n in np.where((vu.value - vl.value) < 0 )[0]:
            warnings.append(f"bus[{n},VMAX] < bus[{n},VMIN]")
        for n in np.where(vl.value < 0.8)[0]:
            warnings.append(f"bus[{n},VMIN] < 0.8")
        for n in np.where(vu.value > 1.2)[0]:
            warnings.append(f"bus[{n},VMAX] > 1.2")

    # line ratings
    if min(s.value) < 0:
        for n in np.where(s.value < 0)[0]:
            warnings.append(f"line[{n},RATE_A] < 0")

    # gen pl/pu range
    if min(pu.value - pl.value) < 0 or min(pu.value) < 0 or min(pl.value) < 0 :
        for n in np.where((pu.value - pl.value) < 0)[0]:
            warnings.append(f"gen[{n},PMAX] < gen[{n},PMIN]")
        for n in np.where(pl.value < 0)[0]:
            warnings.append(f"gen[{n},PMIN] < 0")
        for n in np.where(pu.value < 0)[0]:
            warnings.append(f"gen[{n},PMAX] < 0")

    # gen vg range
    if min(vg.value) < 0.8 or max(vg.value) > 1.2:
        for n in np.where(vg.value < 0.8)[0]:
            warnings.append(f"gen[{n}].vg < 0.8")
        for n in np.where(vg.value > 1.2)[0]:
            warnings.append(f"gen[{n}].vg > 1.2")

    # warn of check failures
    if warnings:
        warn(f"{len(warnings)} model warnings (see 'warnings' for details)")

    # reference busses
    ref = [n for n, bt in enumerate(data["bus"][:, bus.BUS_TYPE]) if bt == 3]

    # cost function
    cost = 0
    # cost = cp.sum(pg**2+qg**2) # TODO: replace with generation costs from gencost
    if curtailment:
        cost += curtailment * cp.sum ( pc**2 + qc**2 ) # curtailment cost

    # constraints
    constraints = [

        # Feasible Set 2
        pf == b @ va, # Equation (1a)
        qf == b @ vm, # Equation (1b)

        # Feasible Set 4
        pl <= pg, pg <= pu, # Equation (3a)
        ql <= qg, qg <= qu, # Equation (3b)
        cp.abs(pf) <= s, # Equation (4a)
        cp.abs(qf) <= s,
        cp.abs(pf) + cp.abs(qf) <= 1.4 * s,
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
        "constraints": [str(x) for x in constraints],
        "constants": {
            "b (pm.S)": b.value.todense(),
            "f (pu)": f.value.todense(),
            "g (pu)": g.value.todense(),
            "vl (pu.kV)": vl.value.T[0],
            "vu (pu.kV)": vu.value.T[0],
            "pl (pu.MW)": pl.value.T[0],
            "pu (pu.MW)": pu.value.T[0],
            "ql (pu.MVAr)": ql.value.T[0],
            "qu (pu.MVAr)": qu.value.T[0],
        },
        "parameters": {
            "s (pu.MVA)": s.value.T[0],
            "pd (pu.MW)": pd.value.T[0],
            "qd (pu.MVAr)": qd.value.T[0],
            "vg (pu.kV)": vg.value.T[0],
        },
        "solution": {},
        "warnings": warnings,
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
    result["time"] = round(toc-tic,3)
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

    # model check
    assert "baseMVA" in data, "missing baseMVA value"
    assert "bus" in data, "missing bus array"
    assert "branch" in data, "missing branch array"
    assert "gen" in data, "missing gen array"

    # dimensions
    N = len(data["bus"])
    M = len(data["branch"])
    K = len(data["gen"])

    # per-unit system
    puS = data["baseMVA"]

   # bus parameters
    bb = data["bus"]
    vl = cp.Constant(value=bb[:, [bus.VMIN]], name="vl") # voltage lower limit
    vu = cp.Constant(value=bb[:, [bus.VMAX]], name="vu") # voltage upper limit
    pd = cp.Parameter(shape=(N,1), value=bb[:, [bus.PD]]/puS, name="pd") # load real power
    qd = cp.Parameter(shape=(N,1), value=bb[:, [bus.QD]]/puS, name="qd") # load reactive power
    bi = {i: n for n, i in enumerate(bb[:, bus.BUS_I])}  # bus index (i is not necessarily reasonable)

    # branch parameters
    br = data["branch"] # branch data

    f_bus = [bi[x] for x in br[:,branch.F_BUS]]
    t_bus = [bi[x] for x in br[:,branch.T_BUS]]

    tap = br[:,[branch.TAP]].flatten()
    tap[np.where(tap==0)] = 1.0 # non-zero is only for transformers, zero is powerline (unity tap)
    err = np.where(tap<0)[0]
    assert len(err) == 0, f"bus[{err},TAP] < 0"

    shift = br[:,[branch.SHIFT]].flatten() * np.pi / 180

    br_status = br[:,[branch.BR_STATUS]].flatten()
    if allin:
        br_status[br_status==0] = 1
    err = np.where([x for x in br_status if x not in [0,1]])[0]
    assert len(err)==0, f"bus[{err},BR_STATUS] value is not in [0,1]"

    br_x = br[:,[branch.BR_X]].flatten()
    err = np.where(br_x==0)[0]
    assert len(err) == 0, f"bus[{err},BR_X] <= 0"

    x = br_status/br_x/tap
    b = sp.sparse.coo_matrix((x,(range(M),f_bus)),shape=(M,N)) \
        - sp.sparse.coo_matrix(((x+shift),(range(M),t_bus)),shape=(M,N)) 
    b = cp.Constant(value=b, name="b") # line susceptances

    f = sp.sparse.coo_matrix((br_status,(range(M),f_bus)),shape=(M,N)).T \
        - sp.sparse.coo_matrix((br_status,(range(M),t_bus)),shape=(M,N)).T 
    f = cp.Constant(value=f, name="f") # line connections

    s = br[:,[branch.RATE_A]]/puS
    s = cp.Parameter(shape=(M,1),value=s,name="s") # line flow limits

    # gen parameters
    gg = data["gen"]
    gi = np.array([bi[n] for n in gg[:,gen.GEN_BUS]])
    gs = 1 if allin else gg[:,gen.GEN_STATUS]
    vg = cp.Parameter(shape=(K,1), value=gg[:,[gen.VG]], name="vg") # bus voltage setpoints
    pl = cp.Constant(value=gs*gg[:,[gen.PMIN]]/puS, name="pl") # real power minimum
    pu = cp.Constant(value=gs*gg[:,[gen.PMAX]]/puS, name="pu") # real power maximum
    ql = cp.Constant(value=gs*gg[:,[gen.QMIN]]/puS, name="ql") # reactive power minimum
    qu = cp.Constant(value=gs*gg[:,[gen.QMAX]]/puS, name="qu") # reactive power maximum
    g = sp.sparse.coo_matrix((np.ones(K),(list(range(K)),gi)),shape=(K,N)).T # sum generators to busses
    g = cp.Constant(value=g,name="g") # sum generators to busses

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

    # sanity checks
    warnings = []

    # bus vl/vu range
    if min(vu.value - vl.value) < 0 or min(vl.value) < 0.8 or max(vu.value) > 1.2:
        for n in np.where((vu.value - vl.value) < 0 )[0]:
            warnings.append(f"bus[{n},VMAX] < bus[{n},VMIN]")
        for n in np.where(vl.value < 0.8)[0]:
            warnings.append(f"bus[{n},VMIN] < 0.8")
        for n in np.where(vu.value > 1.2)[0]:
            warnings.append(f"bus[{n},VMAX] > 1.2")

    # line ratings
    if min(s.value) < 0:
        for n in np.where(s.value < 0)[0]:
            warnings.append(f"line[{n},RATE_A] < 0")

    # gen pl/pu range
    if min(pu.value - pl.value) < 0 or min(pu.value) < 0 or min(pl.value) < 0 :
        for n in np.where((pu.value - pl.value) < 0)[0]:
            warnings.append(f"gen[{n},PMAX] < gen[{n},PMIN]")
        for n in np.where(pl.value < 0)[0]:
            warnings.append(f"gen[{n},PMIN] < 0")
        for n in np.where(pu.value < 0)[0]:
            warnings.append(f"gen[{n},PMAX] < 0")

    # gen vg range
    if min(vg.value) < 0.8 or max(vg.value) > 1.2:
        for n in np.where(vg.value < 0.8)[0]:
            warnings.append(f"gen[{n}].vg < 0.8")
        for n in np.where(vg.value > 1.2)[0]:
            warnings.append(f"gen[{n}].vg > 1.2")

    # warn of check failures
    if warnings:
        warn(f"{len(warnings)} model warnings (see 'warnings' for details)")

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
        f @ pf + pd*(1+margin) == g @ pg, # Equation (2c)
        f @ qf + qd*(1+margin) + ac == g @ qg, # Equation (2d)

        # Feasible Set 4
        pl <= pg, pg <= pu + ap, # Equation (3c)
        ql - aq <= qg, qg <= qu + aq, # Equation (3d)
        cp.abs(pf) <= s + al, # Equation (4b)
        cp.abs(qf) <= s + al, 
        cp.abs(pf) + cp.abs(qf) <= 1.4 * ( s + al ),
        vl <= vm, vm <= vu, # Equation (5b)

        # practical constraints not specified in the mathematical model
        va[ref] == 0,  # reference bus angle is always 0
        vm[gi] == vg,  # bus voltage setpoints
        cp.abs(va) <= 0.175,  # +/- 10 degrees for decoupling assumptions to be valid

        # constraints on addition placements
        ac[gi] == 0, # no capacitors/condensors at generation busses
        
        # limits on reactive power additions relative to real power additions
        cp.abs(qu) + aq <= pu + ap,
        cp.abs(ql) + aq <= pu + ap,
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
        "constants": {
            "b (pm.S)": b.value,
            "f (pu)": f.value,
            "g (pu)": g.value,
            "vl (pu.kV)": vl.value.T[0],
            "vu (pu.kV)": vu.value.T[0],
            "pl (pu.MW)": pl.value.T[0],
            "pu (pu.MW)": pu.value.T[0],
            "ql (pu.MVAr)": ql.value.T[0],
            "qu (pu.MVAr)": qu.value.T[0],
        },
        "parameters": {
            "s (pu.MVA)": s.value.T[0],
            "pd (pu.MW)": pd.value.T[0],
            "qd (pu.MVAr)": qd.value.T[0],
            "vg (pu.kV)": vg.value.T[0],
        },
        "solution": {},
        "violations": {},
        "warnings": []
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
        rows = np.where((solution["branch"][:,[branch.RATE_A]]>0) & (al.value>0) )[0]
        ratio = 2 * al.value.flatten()[rows] * puS / solution["branch"][rows,branch.RATE_A].flatten() + 1
        for column in [branch.RATE_A,branch.RATE_B,branch.RATE_C]: # raised values
            solution["branch"][rows,column] = solution["branch"][rows,column] * ratio
        for column in [branch.BR_R,branch.BR_X,branch.BR_B]: # lowered values
            solution["branch"][rows,column] = solution["branch"][rows,column] / ratio
        
        # generator updates
        if allin:
            solution["gen"][:,gen.GEN_STATUS] = np.ones(K)
        solution["gen"][:,[gen.PG]] = pg.value * puS
        solution["gen"][:,[gen.QG]] = qg.value * puS
        solution["gen"][:,[gen.PMAX]] = solution["gen"][:,[gen.PMAX]] + ap.value * puS
        solution["gen"][:,[gen.QMIN]] = solution["gen"][:,[gen.QMIN]] - aq.value * puS
        solution["gen"][:,[gen.QMAX]] = solution["gen"][:,[gen.QMAX]] + aq.value * puS

        result["solution"] = solution

        # create update list
        updates = []
        for n,x in enumerate([x for x in ap.value[:,0]*puS]):
             if abs(x) > 1e-3:
                updates.append(f"add {x:.3f} MW gen[{n},PMAX]")
        for n,x in enumerate([x for x in aq.value[:,0]*puS]):
             if abs(x) > 1e-3:
                updates.append(f"add {x:.3f} MVAr gen[{n},QMAX]")
        for n,x in enumerate([x for x in ac.value[:,0]*puS]):
             if abs(x) > 1e-3:
                updates.append(f"add {x:.3f} MVAr to bus[{n},BS]")
        for n,x in enumerate([x for x in al.value[:,0]*puS]):
             if abs(x) > 1e-3:
                updates.append(f"add {x:.3f} MVA to branch[{n},RATE_A]")
        result["updates"] = updates

        # create violations list
        checks = violations(solution,formatter=dict)
        if checks:
            result["violations"] = checks
        
        result["ok"] = True

    toc = time()
    result["time"] = round(toc-tic,3)
    return result

def violations(data, 
    precision:float=4, # rounding on data before checking
    error:float=0.02, # error margin on tests
    formatter:str|Callable=None, # formatting call for results (or "counter","table",None)
    ):
    """Enumerate violations in case"""
    result = {"bus": [], "gen": [], "branch": []}

    if "bus" in data:
        for n, v in enumerate(
            data["bus"][:, (bus.VM, bus.VA, bus.VMIN, bus.VMAX)].round(precision)
        ):
            VM, VA, VMIN, VMAX = map(float, v)
            if not VMIN*(1-error) <= VM <= VMAX*(1+error):
                result["bus"].append((n, f"{VM=} pu.V outside ({VMIN=},{VMAX=})"))
    if "gen" in data:
        for n, g in enumerate(
            data["gen"][
                :, (gen.PG, gen.QG, gen.PMIN, gen.PMAX, gen.QMIN, gen.QMAX)
            ].round(precision)
        ):
            PG, QG, PMIN, PMAX, QMIN, QMAX = map(float, g)
            if PMIN < PMAX and not PMIN*(1-error) <= PG <= PMAX*(1+error):
                result["gen"].append((n, f"{PG=} MW outside ({PMIN=},{PMAX=})"))
            if QMIN < QMAX and not QMIN*(1-error) <= QG <= QMAX*(1+error):
                result["gen"].append((n, f"{QG=} MVAr outside ({QMIN=},{QMAX=})"))
    if "branch" in data and data["branch"].shape[1] >= branch.PF:
        for n, b in enumerate(data["branch"][:,[branch.PF,branch.RATE_A]]):
            PF, RATE_A = map(float, np.abs(b))
            if RATE_A > 0 and PF > RATE_A*(1+error):
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

def internals(case,all=False):
    """Generate solver internals in a readable format"""
    def dump(x):
        if isinstance(x,dict):
            return("\n\n  ".join([f"{x}:\n    {str(y).replace('\n','\n    ')}" for x,y in x.items()]))
        elif isinstance(x,list):
            return("\n  ".join((f"{y}" for y in x)))
        else:
            return str(x).replace("\n","  \n")

    keys = case.keys() if all else ["problem","constants","parameters","variables","warnings"]
    return "\n".join([f"\n{x}\n{'-'*len(x)}\n\n  {dump(y)}" for x,y in case.items() if x in keys])

def as_mermaid(case,bus_order=None,line_order=None):
    """Generate Mermaid network diagram""" 
    result = [
        "flowchart LR",
        "classDef bus fill:#000",
        "classDef bushot fill:#f00",
        "classDef genhot fill:#f00",
        "classDef bus color:white,font-weight:bold"]

    nodes = case["bus"][:,[bus.BUS_I,bus.PD,bus.QD,bus.VM,bus.VA,bus.VMIN,bus.VMAX]]
    if bus_order is not None:
        nodes = nodes[bus_order,:]
    result.extend([f"  {int(i)}[{vm:.3f} V\n{va:.3f}&deg;]:::bushot" 
        for i,pd,qd,vm,va,vmin,vmax in nodes if vm < vmin or vm > vmax])
    result.extend([f"  {int(i)}[{vm:.3f} V\n{va:.3f}&deg;]:::bus" 
        for i,pd,qd,vm,va,vmin,vmax in nodes if vmin <= vm <= vmax and va != 0.0])
    result.extend([f"  {int(i)}[{int(i)}]:::bus" 
        for i,pd,qd,vm,va,vmin,vmax in nodes if vm == 1.0 and va == 0.0])
    result.extend([f"""  {int(i)} ==>|{qd:+.1f}j MVAr| L{int(i)}@{{ shape: tri, label: "{pd:+.1f} MW"}} """ 
        for i,pd,qd,va,vm,vmin,vmax in nodes if pd**2 + qd**2 > 0])

    gens = case["gen"][:,[gen.GEN_BUS,gen.PG,gen.QG,gen.PMIN,gen.PMAX,gen.QMIN,gen.QMAX]]
    result.extend([f"  G{int(b)}(({pmax:.0f} MW)) ==>|{p:+.1f}{q:+.1f}j MVA| {int(b)} " 
        for b,p,q,pmin,pmax,qmin,qmax in gens if p**2+q**2 > 0 and pmin <= p <= pmax and qmin <= q <= qmax])
    result.extend([f"  G{int(b)}(({pmax:.0f} MW)):::genhot ==>|{p:+.1f}{q:+.1f}j MVA| {int(b)} " 
        for b,p,q,pmin,pmax,qmin,qmax in gens if p**2+q**2 > 0 and not ( pmin <= p <= pmax and qmin <= q <= qmax ) ])
    result.extend([f"  G{int(b)}(({pmax:.0f} MW)) --> {int(b)} " 
        for b,p,q,pmin,pmax,qmin,qmax in gens if p**2+q**2 == 0])
    
    lines = case["branch"][:,[branch.F_BUS,branch.T_BUS,branch.PF,branch.QF,branch.RATE_A]]
    if line_order is not None:
        lines = lines[line_order]
    result.extend([f"  {int(f)} --> {int(t)}" 
        for f,t,p,q,m in lines if p**2+q**2==0])
    result.extend([f"  {int(f)} ==>|<font color={'red' if abs(p)>m else 'black'}>{p:+.1f}{q:+.1f}j MVA</font>| {int(t)}" 
        if f > 0 else f"  {int(t)} ==>|<font color= {'red' if abs(p)>m else 'black'}>{-p:+.1f}{-q:+.1f}j MVA</font>| {int(f)}"
        for f,t,p,q,m in lines if p**2+q**2>0])

    return "\n".join(result)


if __name__ == '__main__':
    
    import re

    np.set_printoptions(linewidth=999999,precision=4)

    results = {}

    testlist = [os.path.splitext(x)[0]  for x in os.listdir() if x.startswith("case") and x.endswith(".py")]
    for name in sorted(sorted(testlist),key=lambda x:int(re.match("[^0-9]*([0-9]+)",x).group(1))):

        tic = time()

        print(f"Processing {name}",end="...",flush=True)
        with open(f"{name}.txt","w") as fh:
            case = load(name)
            results[name] = {
                "Base case": "-  ",
                "Initial AC OPF": "-  ",
                "Initial FD OPF": "-  ",
                "Initial AC PF": "-  ",
                "Fast OSP": "-  ",
                "Final FD OPF": "-  ",
                "Final AC OPF": "-  ",
                "Final AC PF": "-  ",
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
            results[name]["Base case"] = initial_case["status"] if not initial_case["ok"] else ("warning" if initial_case["warnings"] else "ok")
            print(*[f"{x}:\n{y}\n" for x,y in as_frames(initial_case["solution"]).items()],sep="\n",file=fh)
            if initial_case["ok"]: 
                print(violations(initial_case["solution"],formatter="table"),file=fh)
                results[name]["Base case"] = "violations" if violations(initial_case["solution"],formatter="counter") else "ok"

            print("\n*************************",file=fh)
            print("*** INITIAL FULL OPF ***",file=fh)
            print("*************************\n",file=fh)
            initial_acopf = full_acopf(case,**ppoptions)
            print("STATUS:",initial_acopf["status"],file=fh)
            print(f"TIME: {initial_acopf['time']:.3f} s",file=fh)
            results[name]["Initial AC OPF"] = initial_acopf["status"] if not initial_acopf["ok"] else ("warning" if initial_acopf["warnings"] else "ok")
            if initial_acopf["ok"]:
                print(*[f"{x}:\n{y}\n" for x,y in as_frames(initial_acopf["solution"]).items()],sep="\n",file=fh)
                results[name]["Initial AC OPF"] = "ok"
            if initial_acopf["ok"]: 
                print(violations(initial_acopf["solution"],formatter="table"),file=fh)
                results[name]["Initial AC OPF"] = "violations" if violations(initial_acopf["solution"],formatter="counter") else "ok"

            print("\n***********************************",file=fh)
            print("*** DECOUPLED OPTIMAL POWERFLOW ***",file=fh)
            print("***********************************\n",file=fh)
            fast_acopf = decoupled_acopf(case)
            print("STATUS:",fast_acopf["status"],file=fh)
            print(f"TIME: {fast_acopf['time']:.3f} s",file=fh)
            results[name]["Initial FD OPF"] = fast_acopf["status"] if not fast_acopf["ok"] else ("warning" if fast_acopf["warnings"] else "ok")
            print(*[f"{x}:\n{y}\n" for x,y in as_frames(fast_acopf["solution"]).items()],sep="\n",file=fh)
            if fast_acopf["ok"]: 
                print(violations(fast_acopf["solution"],formatter="table"),file=fh)
                results[name]["Initial FD OPF"] = "violations" if violations(fast_acopf["solution"],formatter="counter") else "ok"

            print("\n****************************",file=fh)
            print("*** INITIAL AC POWERFLOW ***",file=fh)
            print("****************************\n",file=fh)
            initial_acpf = full_acpf(fast_acopf["solution"] if fast_acopf["ok"] else case,**ppoptions)
            if initial_acpf["ok"]: 
                print("STATUS:",initial_acpf["status"],file=fh)
                print(f"TIME: {initial_acpf['time']:.3f} s",file=fh)
                print(*[f"{x}:\n{y}\n" for x,y in as_frames(initial_acpf["solution"]).items()],sep="\n",file=fh)
                print(violations(initial_acpf["solution"],formatter="table"),file=fh)
                if fast_acopf["ok"]:
                    results[name]["Initial AC PF"] = initial_acpf["status"] if not initial_acpf["ok"] else ("warning" if initial_acpf["warnings"] else "ok")
                    if violations(initial_acpf["solution"],formatter="counter"):
                        results[name]["Initial AC PF"] = "violations"

                if not fast_acopf["ok"] or violations(initial_acpf["solution"],formatter="counter") > 0:

                    print("\n***********************************",file=fh)
                    print("*** DECOUPLED AC OPTIMAL SIZING ***",file=fh)
                    print("***********************************\n",file=fh)
                    fast_acosp = decoupled_acosp(case)
                    print("STATUS:",fast_acosp["status"],f"(cost={fast_acosp["value"]})",file=fh)
                    print(f"TIME: {fast_acosp['time']:.3f} s",file=fh)
                    results[name]["Fast OSP"] = fast_acosp["status"] if not fast_acosp["ok"] else ("warning" if fast_acosp["warnings"] else "ok")
                    print(*[f"{x}:\n{y}\n" for x,y in as_frames(fast_acosp["solution"]).items()],sep="\n",file=fh)
                    if not fast_acosp["ok"]:

                        print(internals(fast_acosp),file=fh)
                    
                    else:

                        print(violations(fast_acosp["solution"],formatter="table"),file=fh)
                        results[name]["Fast OSP"] = "violations" if violations(fast_acosp["solution"],formatter="counter") else "ok"
                        print("\nADDITIONS\n=========",*fast_acosp["updates"],sep="\n - ",file=fh)
                        
                        print("\n***************************",file=fh)
                        print("*** FINAL DECOUPLED OPF ***",file=fh)
                        print("***************************\n",file=fh)
                        final_opf = decoupled_acopf(fast_acosp["solution"])
                        print("STATUS:",final_opf["status"],file=fh)
                        print(f"TIME: {final_opf['time']:.3f} s",file=fh)
                        results[name]["Final FD OPF"] = final_opf["status"] if not final_opf["ok"] else ("warning" if final_opf["warnings"] else "ok")
                        if final_opf["ok"]:
                            print(*[f"{x}:\n{y}\n" for x,y in as_frames(final_opf["solution"]).items()],sep="\n",file=fh)
                            results[name]["Final FD OPF"] = "ok"
                            print(violations(final_opf["solution"],formatter="table"),file=fh)
                            results[name]["Final FD OPF"] = "violations" if violations(final_opf["solution"],formatter="counter") else "ok"
                        else:
                            print(internals(final_opf),file=fh)

                        print("\n**************************",file=fh)
                        print("*** FINAL FULL OPF ***",file=fh)
                        print("**************************\n",file=fh)
                        final_acopf = full_acopf(fast_acosp["solution"],**ppoptions)
                        print("STATUS:",final_acopf["status"],file=fh)
                        print(f"TIME: {final_acopf['time']:.3f} s",file=fh)
                        results[name]["Final AC OPF"] = final_acopf["status"] if not final_acopf["ok"] else ("warning" if final_acopf["warnings"] else "ok")
                        if final_acopf["ok"]:
                            print(*[f"{x}:\n{y}\n" for x,y in as_frames(final_acopf["solution"]).items()],sep="\n",file=fh)
                            results[name]["Final AC OPF"] = "ok"
                        if final_acopf["ok"]: 
                            print(violations(final_acopf["solution"],formatter="table"),file=fh)
                            results[name]["Final AC OPF"] = "violations" if violations(final_acopf["solution"],formatter="counter") else "ok"

                        print("\n**************************",file=fh)
                        print("*** FINAL AC POWERFLOW ***",file=fh)
                        print("**************************\n",file=fh)
                        final_acpf = full_acpf(fast_acosp["solution"],**ppoptions)
                        print("STATUS:",final_acpf["status"],file=fh)
                        print(f"TIME: {final_acpf['time']:.3f} s",file=fh)
                        results[name]["Final AC PF"] = final_acpf["status"] if not final_acpf["ok"] else ("warning" if final_acpf["warnings"] else "ok")
                        if final_acpf["ok"]:
                            print(*[f"{x}:\n{y}\n" for x,y in as_frames(final_acpf["solution"]).items()],sep="\n",file=fh)
                            results[name]["Final AC PF"] = "ok"
                        if final_acpf["ok"]: 
                            print(violations(final_acpf["solution"],formatter="table"),file=fh)
                            results[name]["Final AC PF"] = "violations" if violations(final_acpf["solution"],formatter="counter") else "ok"

        print(f"done in {time()-tic:.1f} seconds")

    print(pd.DataFrame(results).T)
