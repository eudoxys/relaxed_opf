import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <center>
    <font size=6><b>Ensuring optimal powerflow feasibility using convex optimization</b></font>
        <br/>
    David P. Chassin, <i>Eudoxys Sciences LLC</i>
        <br/>
    April 2026
    </center>

    **Citation**: D.P. Chassin, "Ensuring optimal powerflow feasibility using convext optimization", April 2026, https://github.com/eudoxys/relaxed_opf.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook examines the problem of locating and sizing real and reactive power resources on an electric network as a relaxation of the optimal power flow problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An electric power system network with $N$ busses, $M$ branches, and $K$ generators can be described using the following variables.

    1. The complex voltage $v \in \mathbb C^N$ are per-unit base kV, with the magnitude $|v| \in \mathbb R^N$ per-unit base kV and the angles $\theta \in \mathbb R^N$ in radians.

    2. The real and reactive power injection $p$ and $q \in \mathbb R^N$, respectively, are per-unit base MVA.

    3. The complex line admittance $y \in \mathbb C^{N \times N}$ are per-unit base Siemens, with the conductance and susceptance $g$ and $b \in \mathbb R^{N \times N}$, respectively.

    4. The real and reactive line flow $p$ and $q \in \mathbb R^{N \times N}$ are per-unit base MVA.

    5. The line flow limit $\bar s \in \mathbb R^{N \times N}$ are given per unit MVA.

    6. The bus voltage lower and upper limit $\underline v$ and $\bar v \in \mathbb R^N$, respectively, are given per-unit base kV.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The powerflow problem can be stated as the non-convex feasibility problem [1]
    $$
    \begin{array}{rlr}
        \min & 0 \\
        \textrm{subject to} & p_{ij} + iq_{ij} = v_i ( v_i^* - v_j^* ) y_{ij}^* & (1)\\
        & \sum_j p_{ij} = p_i & (2a)\\
        & \sum_j q_{ij} = q_i & (2b)\\
    \end{array}
    $$
    where (1), (2), and (3) are the **Feasible Set 1**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In polar representation, the real and reactive parts of Equation (1) can be written separately as
    $$
        p_{ij} = g_{ij}|v_i|^2 - |v_i||v_j|(g_{ij}\cos(\theta_i-\theta_j)-b_{ij}\sin(\theta_i-\theta_j))
    \\
        q_{ij} = b_{ij}|v_i|^2 - |v_i||v_j|(g_{ij}\sin(\theta_i-\theta_j)+b_{ij}\cos(\theta_i-\theta_j))
    $$
    which is approximately linear around nominal operation conditions, i.e., small voltage angle differences and voltage magnitudes near one per-unit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Feasible Set 1 can be linearized by making the following assumptions.

    1. All voltage magnitudes are close to one per-unit: set $|v_i|^2=|v_i|$ and $|v_i||v_j|=|v_j|$.

    2. Conductances are small relative to susceptance: set $g_{ij}=0$.

    3. Voltage angle differences are small: replace $\sin(\theta_i-\theta_j)$ with $\theta_i-\theta_j$ and $\cos(\theta_i-\theta_j)$ with 1.0.

    This gives us **Feasible Set 2** for a decouple power flow
    $$
    \begin{array}{lr}
        p_{ij} = b_{ij}(\theta_i-\theta_j) & \qquad (1a)\\
        q_{ij} = b_{ij}(|v_i| - |v_j|) & (1b) \\
        \sum_j p_{ij} = p_i & (2a) \\
        \sum_j q_{ij} = q_i & (2b)
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The optimal power flow problem adds generation dispatch limits (4a) and (4b), line flow limits (5), and bus voltage limits (6) to the powerflow problem to define the **Feasible Set 3**
    $$
    \begin{array}{lr}
        \textrm{Feasible Set 1} & (1) - (2) \\
        \underline p_i \le p_i \le \bar q_i & (3a) \\
        \underline q_i \le q_i \le \bar q_i & (3b) \\
        p_{ij}^2 + q_{ij}^2 \le \bar s_{ij}^2 & (4) \\
        \underline v_{ij} \le v_{ij} \le \bar v_{ij} & (5)
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The linearized form of Feasible Set 3 is **Feasible Set 4**
    $$
    \begin{array}{lr}
        \textrm{Feasible Set 2} & (1) - (2) \\
        \underline p_i \le p_i \le \bar q_i & (3a) \\
        \underline q_i \le q_i \le \bar q_i & (3b) \\
        |p_{ij}| \le \bar s_{ij} & (4a) \\
        \underline v_{i} \le |v_{i}| \le \bar v_{i} & (5a)
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Study Case
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A modified IEEE 4-bus test system [2] illustrates the conditions for which optimal sizing and placement may be required, specifically there is insufficient generation capacity, voltage support, and line capacity for optimal powerflow feasibility.
    """)
    return


@app.cell
def _(np):
    case = {
        "version": '2',
        "baseMVA": 100.0,
        "bus": np.array([
            # BUS_I, BUS_TYPE,  PD, PQ, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN 
            [     1,        3,   0,  0,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            [     2,        1,   0,  0,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            [     3,        1, 200, 10,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            [     4,        2, 200, 10,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            ]),
        "gen": np.array([
            # GEN_BUS, PG, QG, QMAX, QMIN,   VG, MBASE, GEN_STATUS, PMAX, PMIN
            [       1,  0,  0,  100, -100, 1.00,   100,          1,  400,    0,],
        ]),
        "branch": np.array([
            # F_BUS, T_BUS,  BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX
            [     1,     2, 0.001, 0.02, 0.10,    400,    400,    500,   0,     0,         1,   -360,    360],
            [     2,     3, 0.002, 0.05, 0.08,    200,    200,    200,   0,     0,         1,   -360,    360],
            [     2,     4, 0.002, 0.05, 0.08,    200,    200,    200,   0,     0,         1,   -360,    360],
            [     3,     4, 0.003, 0.08, 0.12,     50,     50,     50,   0,     0,         1,   -360,    360],
            ]),
        "gencost": np.array([
            # MODEL, STARTUP, SHUTDOWN, N, COST0, COST1, COST2
            [     2,     0.0,      0.0, 3,  0.04,  20.0,   0.0],
            ]),
        }
    return (case,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Violation Tests
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The primary test of a power flow case is whether there are any voltage, generation, or line flow violations of limits. This includes testing the following

    1. Are the bus voltage magnitudes `VM` within the range of `(VMIN,VMAX)`?

    2. Are the generation powers `PG` and `QG` within the ranges of `(PMIN,PMAX)` and `(QMIN,QMAX)`, respectively?

    3. Are the line flows `PF`, `PT`, `QF`, and `QT` within the ranges of `RATE_A`, `RATE_B`, and `RATE_C`?

    In its unsolved state, the study case is presented with no violations.
    """)
    return


@app.cell(hide_code=True)
def _(branch, bus, gen, mo, np):
    def violations(case, precision=3, formatter=None):
        """Enumerate violations in case"""
        if formatter is None:
            formatter = _as_table
        result = {"bus": [], "gen": [], "branch": []}
        for n, v in enumerate(
            case["bus"][:, (bus.VM, bus.VA, bus.VMIN, bus.VMAX)].round(precision)
        ):
            vm, va, vmin, vmax = map(float, v)
            if not vmin <= vm <= vmax:
                result["bus"].append((n, f"{vm=} pu.V outside ({vmin},{vmax})"))
        for n, g in enumerate(
            case["gen"][
                :, (gen.PG, gen.QG, gen.PMIN, gen.PMAX, gen.QMIN, gen.QMAX)
            ].round(precision)
        ):
            p, q, pmin, pmax, qmin, qmax = map(float, g)
            if not pmin <= p <= pmax:
                result["gen"].append((n, f"{p=} MW outside ({pmin},{pmax})"))
            if not qmin <= q <= qmax:
                result["gen"].append((n, f"{q=} MVAr outside ({qmin},{qmax})"))
        for n, b in enumerate(
            case["branch"][
                :,
                (
                    branch.PF,
                    branch.PT,
                    branch.QF,
                    branch.QT,
                    branch.RATE_A,
                    branch.RATE_B,
                    branch.RATE_C,
                ),
            ]
        ):
            pf, pt, qf, qt, ratea, rateb, ratec = map(float, b)
            s = max(
                map(
                    float,
                    np.array(
                        [np.sqrt(pf**2 + qf**2), np.sqrt(pt**2 + qt**2)]
                    ).round(precision),
                )
            )
            rate = round(float(np.max([ratea, rateb, ratec])), precision)
            if rate > 0 and s > rate:
                result["branch"].append((n, f"{s=} MVA outside (0,{ratea})"))
        if formatter:
            return formatter(result)
        return result


    def _as_table(violations):
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
        return mo.md("\n".join(result))

    return (violations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial AC Powerflow Solution
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The initial full AC power flow solved using PyPower [3] converges. However, the output of generator unit 0 exceeds the maximum real power limit and the line flows on lines 1 and 2 exceeds the line ratings.
    """)
    return


@app.cell(hide_code=True)
def _(branch, bus, case, ppoption, runpf, violations):
    _solution = runpf(case, ppoption(VERBOSE=0, OUT_ALL=0))
    initial_solution = _solution[0]["order"]["int"]

    _pf, _pt, _qf, _qt = _solution[0]["order"]["int"]["branch"][
        :, [branch.PF, branch.PT, branch.QF, branch.QT]
    ].T
    # print("pf=", _pf.round(3), "\npt=", _pt.round(3), sep="\n")
    # print("qf=", _qf.round(3), "\nqt=", _qt.round(3), sep="\n")

    _m, _a = _solution[0]["order"]["int"]["bus"][:, [bus.VM, bus.VA]].T
    # print("\nm=", _m.round(3), "\na=", _a.round(3), sep="\n")

    violations(initial_solution, 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial AC Optimal Powerflow Solution
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The full AC OPF of the case fails because there is no feasible generation output to satisfy the loads.
    """)
    return


@app.cell(hide_code=True)
def _(case, mo, ppoption, runopf):
    _result = runopf(case,ppoption(VERBOSE=0,OUT_ALL=0))["raw"]["output"]["message"]
    mo.md(f"AC OPF {_result.lower()}.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decoupled Optimal Powerflow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The decoupled powerflow problem cannot be solved without solving the optimal powerflow problem first. This is because the generator outputs must be part of the solution if the solution is to be realizable.  Since the decouple powerflow problem in Feasible Set 3 is part of Feasible Set 4, the powerflow problem will be solved as part of the optimal powerflow problem. The decoupled optimal powerflow problem can be constructed in CVX as the following feasibility problem using Feasible Set 4.
    """)
    return


@app.cell
def _(branch, bus, case, cp, gen, np):
    # decoupled opf
    N = len(case["bus"])
    M = len(case["branch"])
    K = len(case["gen"])

    # parameters
    b = cp.Parameter(shape=(N, N), value=[[0]*N]*N, name="b",nonneg=True,symmetric=True)
    s = cp.Parameter(shape=(N, N), value=[[0]*N]*N, name="s",nonneg=True,symmetric=True)
    pl = cp.Parameter(shape=(N),value=[0]*N,name="pl",nonneg=True)
    pu = cp.Parameter(shape=(N),value=[0]*N,name="ph",nonneg=True)
    ql = cp.Parameter(shape=(N),value=[0]*N,name="ql")
    qu = cp.Parameter(shape=(N),value=[0]*N,name="qh",nonneg=True)
    pd = cp.Parameter(shape=(N),value=[0]*N,name='pd',nonneg=True)
    qd = cp.Parameter(shape=(N),value=[0]*N,name='qd')

    # variables
    p = cp.Variable((N, N), name="p",nonneg=True)  # line real power injections
    q = cp.Variable((N, N), name="q")  # line reactive power injections
    m = cp.Variable(N, name="|v|", nonneg=True)  # voltage magnitudes
    a = cp.Variable(N, name="𝞱")  # voltage angles
    pg = cp.Variable(N, name="pg", nonneg=True) # generator real power dispatch
    qg = cp.Variable(N, name="qg", nonneg=True) # generator reactive power dispatch

    # setup Feasible Sets
    ref = [n for n, bt in enumerate(case["bus"][:, bus.BUS_TYPE]) if bt == 3]
    constraints = [  
    
        # practical constraints not specified in the mathematical model
        a[ref] == 0,  # reference bus angle is always 0
        m[ref] == 1,  # reference bus magnitude is always 1
        cp.abs(a)
        <= np.round(np.pi / 18, 4),  # angles must be within +/- 10 degrees
    ]

    # line injections
    puS = case["baseMVA"]
    # puV = case["bus"][:,bus.BASE_KV] # only needed for manual checks
    # puZ = puS / puV**2 # only needed for manual checks
    bi = {i: n for n, i in enumerate(case["bus"][:, bus.BUS_I])}  # bus index
    for i, j, bij, sij in [
        (bi[f], bi[t], 1 / complex(x, y).imag, z / puS)
        for f, t, x, y, z in case["branch"][
            :,
            [branch.F_BUS, branch.T_BUS, branch.BR_R, branch.BR_X, branch.RATE_A],
        ]
    ]:

        b.value[i, j] = b.value[j, i] = bij
        constraints.append(p[i,j] == b[i,j] * (a[i] - a[j]))  # Equation (1a)
        constraints.append(p[i,j] == -p[j,i])  # Equation (1a) anti-symmmetry
        constraints.append(q[i,j] == b[i,j] * (m[i] - m[j]))  # Equation (1b)
        constraints.append(q[i,j] == -q[j,i])  # Equation (1b) anti-symmmetry

        s.value[i, j] = s.value[j, i] = sij
        constraints.append(cp.abs(p[i, j]) <= s[i, j])  # Equation (4a)

    # bus injections
    pd.value = case["bus"][:, bus.PD] / puS  # bus real power injections
    qd.value = case["bus"][:, bus.QD] / puS  # bus reactive power injections
    vmin, vmax = case["bus"][
        :, [bus.VMIN, bus.VMAX]
    ].T  # bus voltage magnitude limits
    for j in range(N):
        constraints.append(pd[j] == cp.sum(p[:, j] + pg[j]))  # Equation (2a)
        constraints.append(qd[j] == cp.sum(q[:, j] + qg[j]))  # Equation (2b)
        constraints.append(m[j] >= vmin[j])  # Equation (5a)
        constraints.append(m[j] <= vmax[j])  # Equation (5a)

    # generation dispatch
    for n, pmin, qmin, pmax, qmax in case["gen"][
        :, [gen.GEN_BUS, gen.PMIN, gen.QMIN, gen.PMAX, gen.QMAX]
    ]:
        i = bi[n]
        pl.value[i] = pmin / puS
        pu.value[i] = pmax / puS
        ql.value[i] = qmin / puS
        qu.value[i] = qmax / puS
    for i in range(N):
        if pl[i].value < pu[i].value:
            constraints.append(pl[i] <= pg[i])  # Equation (3a)
            constraints.append(pu[i] >= pg[i])  # Equation (3a)
        else:
            constraints.append(pg[i] == pl[i])
        if ql[i].value < qu[i].value:
            constraints.append(ql[i] <= qg[i])  # Equation (3b)
            constraints.append(qu[i] >= qg[i])  # Equation (3b)
        else:
            constraints.append(qg[i] == ql[i])

    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve()

    print(
        "PARAMETERS\n----------",
        f"b[pu.S]=\n{b.value.round(1)}",
        f"s[MVA]=\n{(s*puS).value.round(1)}",
        f"pl[MW]=\n{(pl*puS).value.round(1)}",
        f"pu[MW]=\n{(pu*puS).value.round(1)}",
        f"ql[MW]=\n{(ql*puS).value.round(1)}",
        f"qu[MW]=\n{(qu*puS).value.round(1)}",
        f"pd[MW]=\n{(pd*puS).value.round(1)}",
        f"qd[MW]=\n{(qd*puS).value.round(1)}",

        "VARIABLES\n---------",
        f"p[MW]=\n{(p.value*puS).round(1)}",
        f"q[MVAr]=\n{(q.value*puS).round(1)}",
        f"m[pu.V]=\n{m.value.round(3)}",
        f"a[deg]=\n{(a.value*180/np.pi).round(3)}",
        f"pg[MW]=\n{(pg*puS).value.round(1)}",
        f"qg[MW]=\n{(qg*puS).value.round(1)}",

        "CONSTRAINTS\n---------",
        "\n".join([f"{n:2d}. {str(x)}" for n,x in enumerate(constraints)]),

        sep="\n\n",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # References

    1. J.A. Taylor, "Convex Optimization of Power Systems", 2015.

    2. W. H. Kersting, "Radial distribution test feeders," 2001 IEEE Power Engineering Society Winter Meeting. Conference Proceedings, pp. 908-912 vol.2.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from pypower.ppoption import ppoption
    from pypower.runpf import runpf
    from pypower.runopf import runopf
    from pypower import idx_bus as bus
    from pypower import idx_brch as branch
    from pypower import idx_gen as gen
    from pypower import idx_cost as cost

    return branch, bus, cp, gen, mo, np, ppoption, runopf, runpf


if __name__ == "__main__":
    app.run()
