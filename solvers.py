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
    result = runpf(case,ppoption(**kwargs))
    return {
        "case": copy(case),
        "status": "solved",
        "solution": result[0]["order"]["int"]
    }

def full_acopf(case:dict,**kwargs) -> dict:
    """Solve full AC optimal powerflow"""
    result = runopf(case,ppoption(**kwargs))
    solution = copy(case)
    data = result["var"]["val"]
    solution["bus"][:,bus.VA] = data["Va"]
    solution["bus"][:,bus.VM] = data["Vm"]
    solution["gen"][:,gen.PG] = data["Pg"]
    solution["gen"][:,gen.QG] = data["Qg"][0]
    solution = runpf(solution,ppoption(**kwargs))[0]["order"]["int"]
    return {
        "case": copy(case),
        "status": "solved",
        "solution": solution
    }

def decoupled_acopf(data:dict,**options) -> dict:
    """Solve decoupled OPF
    
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
      - `problem`: cvxpy problem data (see `cvxpy.Problem)
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

    Description
    -----------

    Solves the optimal power flow problem using the decoupled powerflow method
    in Taylor Chapter 3. If `softening` is specified it must include the following

    - `load`: a dict with the following:

      - `cost`: a cost of load curtailment per-unit of generation cost

      - `limit`: a maximum fraction of load that may be curtailed
    """
    
    # default options
    if "canon_backend" not in options:
        options["canon_backend"] = "SCIPY"

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
    for n,br in enumerate(data["branch"]):
        i = bi[br[branch.F_BUS]]
        j = bi[br[branch.T_BUS]]
        b.value[n,i] = br[branch.BR_STATUS] / complex(br[branch.BR_R], br[branch.BR_X]).imag
        b.value[n,j] = -b.value[n,i]
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
    pc = cp.Variable(shape=(N,1), name="pc", nonneg=True) # load real power curtailment
    qc = cp.Variable(shape=(N,1), name="qc", nonneg=True) # load reactive power curtailment

    # gen parameters
    gi = [bi[n] for n in data["gen"][:,gen.GEN_BUS]]
    pl.value[gi] = data["gen"][:,gen.PMIN].T[0] / puS
    pu.value[gi] = data["gen"][:,gen.PMAX].T[0] / puS
    ql.value[gi] = data["gen"][:,gen.QMIN].T[0] / puS
    qu.value[gi] = data["gen"][:,gen.QMAX].T[0] / puS

    # setup Feasible Sets
    ref = [n for n, bt in enumerate(data["bus"][:, bus.BUS_TYPE]) if bt == 3]

    # cost function
    cost = cp.sum(pg**2+qg**2)

    # constraints
    constraints = [

        # Feasible Set 2
        pf == b @ va, # Equation (1a)
        qf == b @ vm, # Equation (1b)
        f @ pf + pd == pg, # Equation (2a)
        f @ qf + qd == qg, # Equation (2b)

        # Feasible Set 4
        pl <= pg, pg <= pu, # Equation (3a)
        ql <= qg, qg <= qu, # Equation (3b)
        cp.abs(pf) <= s, # Equation (4a)
        vl <= vm, vm <= vu, # Equation (5a)

        # practical constraints not specified in the mathematical model
        va[ref] == 0,  # reference bus angle is always 0
        vm[gi] == vg,  # bus voltage setpoints
        cp.abs(va) <= 0.175,  # +/- 10 degrees for decoupling assumptions to be valid
    ]

    # problem statement
    problem = cp.Problem(cp.Minimize(cost),constraints)
    problem.solve(**options)

    # solution results
    result = {
        "ok": False,
        "case": copy(data),
        "status": problem.status,
        "value": np.round(problem.value,4),
        "problem": problem,
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
        }
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
        
        solution["bus"][:,bus.VA] = va.value.T[0]
        solution["bus"][:,bus.VM] = vm.value.T[0]

        solution["branch"][:,branch.PF] = pf.value.T[0]
        solution["branch"][:,branch.QF] = qf.value.T[0]

        n = [bi[x] for x in solution["gen"][:,gen.GEN_BUS]]
        solution["gen"][:,gen.PG] = pg.value[n,0] * puS
        solution["gen"][:,gen.QG] = qg.value[n,0] * puS

        result["solution"] = solution
        checks = violations(solution,formatter=dict)
        if checks:
            result["violations"] = checks
        result["ok"] = True
    else:
        result["solution"] = {}
        result["violations"] = {}

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
    if branch in data and data["branch"].shape[1] >= branch.QT:
        for n, b in enumerate(data["branch"][:,[branch.PF,branch.RATE_A]]):
            PF, RATE_A = map(float, b)
            if RATE_A > 0 and PF > RATE:
                result["branch"].append((n, f"|PF|={PF} MVA outside (0,{RATE_A=})"))
    if formatter:
        return formatter(result)
    return result

def as_frames(data,showall=False,**kwargs):
    """Return case data as dataframes"""
    if not kwargs and showall is False:
        kwargs = dict(bus="BUS_I,PD,QD,VM,VA,VMAX,VMIN",
                      branch="F_BUS,T_BUS,BR_X,RATE_A,ANGMIN,ANGMAX",
                      gen="GEN_BUS,PG,QG,PMIN,PMAX,QMIN,QMAX",
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
    def dump(x):
        if isinstance(x,dict):
            return("\n\n  ".join([f"{x}:\n    {str(y).replace('\n','\n    ')}" for x,y in x.items()]))
        else:
            return str(x).replace("\n","  \n")

    return "\n".join([f"\n{x}\n{'-'*len(x)}\n\n  {dump(y)}" for x,y in case.items() if x in ["problem","parameters","variables"]])
    
if __name__ == '__main__':
    
    from case4m import case
    
    pd.options.display.max_columns = None
    pd.options.display.width = None

    ppoptions = dict(VERBOSE=0,OUT_ALL=0)
    
    curtailment = None

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
    print(violations(initial_acpf["solution"]))

    print("\n*********************************")
    print("*** FULL AC OPTIMAL POWERFLOW ***")
    print("*********************************\n")
    initial_acopf = full_acopf(case,**ppoptions)
    print("STATUS:",initial_acopf["status"])
    print(*[f"{x}:\n{y}\n" for x,y in as_frames(initial_acopf["solution"]).items()],sep="\n")
    print(violations(initial_acopf["solution"]))

    print("\n**********************************")
    print("*** DECOUPLED AC OPTIMAL POWERFLOW ***")
    print("************************************\n")
    fast_acopf = decoupled_acopf(case)
    print("STATUS:",fast_acopf["status"])
    print(*[f"{x}:\n{y}\n" for x,y in as_frames(fast_acopf["solution"]).items()],sep="\n")
    print(violations(fast_acopf["solution"]))
    if not fast_acopf["status"]:
        print(internals(fast_acopf))
