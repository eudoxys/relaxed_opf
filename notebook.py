import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <center>
    <font size=5><b>Optimal sizing and placement using softened relaxed optimal powerflow</b></font><br/>
        David P. Chassin, <i>Eudoxys Sciences LLC</i><br/>
        April 2026
    </center>

    **Citation**: D.P. Chassin, "Optimal sizing and placement using softened relaxed optimal powerflow", April 2026. URL: https://github.com/eudoxys/relaxed_opf.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook presents a method for solving the problem of locating and sizing real and reactive power resources on an electric network as a relaxation of the optimal power flow problem with softened constraints on loads (i.e., load curtailment and static VAR devices), generation capacities (i.e., generator and substation capacity expansion), and line flows (i.e., transmission line capacity expansion).

    The relaxations enable the use of convex optimization solvers as illustrated here using [`cvxpy`](https://www.cvxpy.org/). These are discussed in the first section.  The softening of constraints converts the problem from a classical optimal powerflow problem to an optimal sizing and placement problem, which is discussed in the second section.

    The method does not support adding generators to PQ busses nor does it support adding new transmission lines or transformers where none are already present.
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
    where (1), (2a), and (2b) are the **Feasible Set 1**.
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

    This gives us **Feasible Set 2** for a decoupled power flow
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
    The optimal power flow problem adds generation dispatch limits (3a) and (3b), line flow limits (4), and bus voltage limits (5) to the powerflow problem to define the **Feasible Set 3**
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


@app.cell
def _(mo):
    mo.md(r"""
    A modified IEEE 4-bus test system [2] illustrates the conditions for which optimal sizing and placement may be required, specifically there is insufficient generation capacity, voltage support, and line capacity for optimal powerflow feasibility. In addition, the generator cost function is reduced to a unity linear function.
    """)
    return


@app.cell
def _(case, show_data):
    show_data(case,"Table 1(a): Base case data")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Violation Tests
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The primary test of a power flow case is whether there are any voltage, generation, or line flow violations of limits. This includes testing the following

    1. Are the bus voltage magnitudes `VM` within the range of `(VMIN,VMAX)`?

    2. Are the generation powers `PG` and `QG` within the ranges of `(PMIN,PMAX)` and `(QMIN,QMAX)`, respectively?

    3. Are the line flows `PF`, `PT`, `QF`, and `QT` within the ranges of `RATE_A`, `RATE_B`, and `RATE_C`?

    In its unsolved state, the study case is presented with no violations.
    """)
    return


@app.cell
def _(case, show_violations):
    show_violations(case,caption="Table 1(b): Base case violations")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial AC Powerflow Solution
    """)
    return


@app.cell
def _(case, solvers):
    initial_powerflow = solvers.full_acpf(case, VERBOSE=0, OUT_ALL=0)
    return (initial_powerflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The initial full AC power flow solved using PyPower [3] converges with the results shown in Table 2(a). However, it does not satisfy Feasible Set 4, as shown in Table 2(b).
    """)
    return


@app.cell
def _(initial_powerflow, mo):
    mo.md(f"""
    Initial powerflow {initial_powerflow["status"].lower()}.
    """)
    return


@app.cell
def _(initial_powerflow, show_data):
    # mo.accordion(solvers.as_frames(initial_powerflow["solution"],True))
    show_data(initial_powerflow["solution"],caption="Table 2(a): Initial full AC powerflow solution")
    return


@app.cell(hide_code=True)
def _(initial_powerflow, show_violations):
    show_violations(initial_powerflow["solution"],caption="Table 2(b): Initial full AC powerflow violations")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial AC Optimal Powerflow Solution
    """)
    return


@app.cell
def _(case, solvers):
    initial_opf = solvers.full_acopf(case,VERBOSE=0, OUT_ALL=0)
    return (initial_opf,)


@app.cell
def _(mo):
    mo.md(f"""
 
    """)
    return


@app.cell
def _(initial_opf, mo):
    mo.md(rf"""
    The full AC OPF in `pypower` finds a solution that satisfies the voltage limits (5a) in Feasible Set 4 but it does not satisfy the generation and line flow constraints (3a), (3b), and (4a). Thus, the full AC OPF is  {initial_opf["status"].lower()} by `pypower` but it does not satisfy Feasible Set 4. The result is shown in Table 3(a) and the violations of Feasible Set 4 are shown in Table 3(b).
    """)
    return


@app.cell
def _(initial_opf, show_data):
    show_data(initial_opf["solution"],caption="Table 3(a): Full AC optimal powerflow solution")
    return


@app.cell
def _(initial_opf, show_violations):
    show_violations(initial_opf["solution"],caption="Table 3(b): Full AC optimal powerflow violations")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decoupled Optimal Powerflow
    """)
    return


@app.cell
def _(case, solvers):
    # Decoupled opf solution
    decoupled_opf = solvers.decoupled_acopf(case)
    return (decoupled_opf,)


@app.cell(hide_code=True)
def _(decoupled_opf, mo):
    mo.md(rf"""
    The decoupled optimal powerflow problem must not only satisfy the voltage constraints (5a) in Feasible Set 4, but it also must satisfy the generation constraints (3a) and (3b) as well as the line flow constraints (4a). The decoupled AC OPF is {decoupled_opf["status"].lower()} as presented in the base case.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal Sizing and Placement

    This section discusses how an infeasible optimal powerflow problem is converted to a feasible optimal sizing and placement (OSP) problem by following softening of constraints.

    The OSP problem assumes that all defined resources are in service. The following constraints are softening by convertings them to costs per-unit of generation capacity expansion cost.

    1. A capacitor or condensors of size $c_i$ can be added to a PQ bus $i$ at the cost $\alpha c_i$ to raise or lower, respectively, the voltage $v_i$ when $|v_i|$ is outside the range of $(\underline v_i,\bar v_i)$.

    2. Added generation real power $g_i$ and reactive power $h_i$ can be provided to the non-PQ bus $i$ at the cost $\beta g$ when $p_i$ and/or $q_i$ are outside the ranges $(0,\bar p_i)$ and $(\underline q_i,\bar q_i)$, respectively, provided that $p_i + g_i <= q_i + h_i$.

    3. Transformer and powerline capacity can be increased by $d_{ij}$ at the costs $\gamma_t d_{ij}$ and $\gamma_l d_{ij}$, respectively, on branches where $|p_{ij}|$ exceeds $|s_{ij}|$.

    4. The load on PQ busses is increased to $(p_i+jq_i)(1+e_i)$ to ensure that the added assets provide a sufficient safety margin.

    The costs and margins of these softened constraints as shown in Table 4.

    Table 4: Optimal Size and Placement Softening Parameters

    | Item  | Value        | Description
    | :---: | :----------: | :----------
    | 1.    | $\alpha=0.1$ | Capacitor addition cost (PQ bus only)
    |       | $\alpha=1.0$ | Condensor addition cost (PQ bus only)
    | 2.    | $\beta=1.0$  | General real-power expansion cost (non-PQ bus only)
    |       | $\beta=0.0$  | Generator reactive power expansion cost (non-PQ bus only)
    | 3.    | $\gamma=10$  | Transformer capacity expansions cost (branch tap is non-zero)
    |       | $\gamma=100$ | Powerline capacity expansion cost (branch taps is zero)
    | 4.    | $e=15$%      | Load margin
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The optimal sizing and placement problem is then stated as
    $$
    \begin{array}{rlr}
        \underset{v,\theta,p,q,c,g,h,f} \min & \alpha \sum_i p_i + \beta \sum_i c_i + \gamma \sum_{ij} d_{ij} \\
        \textrm{subject to} \\
        & \textrm{(Feasible Set 5)} \\
        & p_{ij} = b_{ij} (\theta_i - \theta_j) & (1a) \\
        & q_{ij} = b_{ij} (|v_i| - |v_j|) & (1b) \\
        & \sum_j p_{ij} = p_i~(1+e_i) & (2c) \\
        & \sum_j q_{ij} = q_i~(1+e_i) & (2d) \\
        \\
        & \textrm{(Feasible Set 6)} \\
        & \underline p_i \le p_i \le \bar p_i + g_i & (3c) \\
        & \underline q_i - h_i \le q_i \le \bar q_i + h_i & (3d) \\
        & |p_{ij}| \le \bar s_{ij} + d_{ij} & (4b)\\
        & \underline v_i \le |v_i| \le \bar v_i & (5b)
    \end{array}
    $$
    """)
    return


@app.cell
def _(case, solvers):
    decoupled_osp = solvers.decoupled_acosp(case)
    return (decoupled_osp,)


@app.cell(hide_code=True)
def _(decoupled_osp, mo):
    mo.md(rf"""
    Decoupled AC OSP is {decoupled_osp["status"].lower()}.
    """)
    return


@app.cell
def _(decoupled_osp, show_data):
    show_data(decoupled_osp["solution"],caption="Table 5(a): Decoupled AC optimal sizing and placement solution") if decoupled_osp["ok"] else None
    return


@app.cell
def _(decoupled_osp, show_violations):
    show_violations(decoupled_osp["solution"],caption="Table 5(b): Decoupled AC optimal sizing and placement violations")  if decoupled_osp["ok"] else None
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
def _(mo, solvers):
    def show_data(case,caption=""):
        data = solvers.as_frames(case)
        result = mo.vstack(
            [mo.md(caption)] +
            [
                mo.md(f"**{x}**\n~~~\n{repr(y)}\n~~~\n")
                for x, y in data.items()
            ]
        )
        return mo.md(f"{caption}\n~~~\nNo solution\n~~~") if not data else result

    def show_violations(case,caption=""):
        return mo.vstack([mo.md(caption),mo.md(f"~~~\n{solvers.violations(case,precision=1)}\n~~~")])

    return show_data, show_violations


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from copy import deepcopy as copy
    from pypower.ppoption import ppoption
    from pypower.runpf import runpf
    from pypower.runopf import runopf
    from pypower import idx_bus as bus
    from pypower import idx_brch as branch
    from pypower import idx_gen as gen
    from pypower import idx_cost as cost
    import solvers
    from case4r import case

    return case, mo, solvers


if __name__ == "__main__":
    app.run()
