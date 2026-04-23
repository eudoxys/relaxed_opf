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

    The initial full AC power flow solved using PyPower [3] converges. However, the output of generator unit 0 exceeds the maximum real power limit and the line flows on lines 1 and 2 exceeds the line ratings.
    """)
    return


@app.cell
def _(case, solvers):
    initial_powerflow = solvers.full_acpf(case, VERBOSE=0, OUT_ALL=0)
    return (initial_powerflow,)


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
def _(mo):
    mo.md(r"""
    The full AC OPF of the case fails because there is no feasible generation output to satisfy the loads.
    """)
    return


@app.cell
def _(case, solvers):
    initial_opf = solvers.full_acopf(case,VERBOSE=0, OUT_ALL=0)
    return (initial_opf,)


@app.cell
def _(initial_opf, mo):
    mo.md(f"""
    Full AC OPF {initial_opf["status"].lower()}.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The decoupled powerflow problem cannot be solved without solving the optimal powerflow problem first. This is because the generator outputs must be part of the solution if the solution is to be realizable.  Since the decouple powerflow problem in Feasible Set 3 is part of Feasible Set 4, the powerflow problem will be solved as part of the optimal powerflow problem. The decoupled optimal powerflow problem can be constructed in CVX as the following feasibility problem using Feasible Set 4.
    """)
    return


@app.cell
def _(case, solvers):
    # Decoupled opf solution
    decoupled_opf = solvers.decoupled_acopf(case)
    return (decoupled_opf,)


@app.cell
def _(decoupled_opf, mo):
    mo.md(f"""
    Decoupled AC OPF is {decoupled_opf["status"].lower()}.
    """)
    return


@app.cell
def _(decoupled_opf, show_data):
    show_data(decoupled_opf["solution"],caption="Table 4(a): Decoupled AC optimal powerflow solution")
    return


@app.cell
def _(decoupled_opf, show_violations):
    show_violations(decoupled_opf["solution"],caption="Table 4(b): Decoupled AC optimal powerflow violations")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal Sizing and Placement

    This section discusses how an infeasible optimal powerflow problem is converted to a feasible optimal placement problem using the following softening of constraints:

    1. Capacitors can be added to raise voltage on PQ busses where `VM` is below `VMIN` by decreasing `BS`.

    2. Condensors can be added to reduce voltage on PQ busses where `VM` is above `VMAX` by increasing `BS`.

    3. Generation reactive power capacity can be added to non-PQ busses where `QG` is outside `(QMIN,QMAX)` by decreasing `QMIN` and/or increasing `QMAX`.

    4. Generation real power capacity can be added to non-PQ busses where `PG` is above `PMAX` by increasing `PMAX`.

    5. Transformer/line capacity can be increased where `PF` exceeds `RATE_A` by increasing `RATE_A`.

    The costs of these capacity expansions are specified such that they are applied in that order of preference.
    """)
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
                mo.md(f"**{x}**\n\n~~~\n{repr(y)}\n~~~\n")
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
