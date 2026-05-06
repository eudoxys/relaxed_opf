import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <center>
    <font size=5><b>Optimal capacity expansion using softened relaxed optimal powerflow</b></font><br/>
        David P. Chassin, <i>Eudoxys Sciences LLC</i><br/>
        April 2026
    </center>

    **Citation**: D.P. Chassin, "Optimal sizing and placement using softened relaxed optimal powerflow", April 2026. URL: https://github.com/eudoxys/relaxed_opf.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook presents a method for solving the problem of expanding real and reactive power resources on an electric network as a relaxation of the optimal power flow problem with softened constraints on loads (i.e., addition of static VAR devices), generation capacities (i.e., generator and substation capacity expansion), and line flows (i.e., transmission line capacity expansion).

    The relaxations enable the use of convex optimization solvers as illustrated here using [`cvxpy`](https://www.cvxpy.org/). These are discussed in the first section.  The softening of constraints converts the problem from a canonical optimal powerflow problem to an optimal capacity expansion problem, which is discussed in the second section.

    The method does not support adding generators to PQ busses nor does it support adding new transmission lines or transformers where none are already present as one might find in an optimal sizing and placement problem.
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
    Conventional electricity infrastructure planning addresses problems of expanding the capacity of an electric power network to meet growing demands for energy. These problems can be very complex, involving large number of factors that frequently involve integer variables that represent that addition of nodes resources and transmission lines [1].

    From a computational perspective, insufficient generation resources, reactive power support, or transmission capacity is often discernable by an infeasible optimal powerflow problem (OPF). A convex relaxation of the OPF problem is presented in the first section entitle "Optimal Powerflow" and illustrated with a simple 4-bus case that is infeasible.

    The infeasibility of most OPF problems can usually be remedied by load curtailment. However for certain studies it is preferable to upgrade the network's load-carrying capacity so that the OPF problem becomes feasible without resorting to load curtailment. Network capacity expansion in this context can be thought of as a subset of the larger infrastructure planning problem and expressed as an OPF problem where the constraints on generation and powerline resources are softened using variables describing the generation, static VAr devices, and powerline upgrades necessary to make OPF feasible at a minimum cost. This softening of the OPF problem is presented in the second section entitled "Optimal Capacity Expansion".

    In the third section entitled "WECC 240 Study", the optimal capacity expansion (OCE) problem is applied to a 243-bus model of the western interconnection in North America.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal Powerflow
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
    In cases where $\underline p_i > 0$, it should be noted that generators must be constrained to $0 \le p_i \le \overline p_i$. Thus the minimum generation dispatch constraint is removed in order to allow zero dispatch of generators, if necessary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When DC lines are included in the network model, additional constraints must be provided relating only the DC lines to define **Feasible Set 5**.
    $$
    \begin{array}{lr}
        \textrm{Feasible Set 4:} & (1) - (5) \\
        p_i = p_j + a_{ij} p_{ij} + c_{ij} & (6)\\
    \end{array}
    $$
    where $a_{ij}$ are the losses proportional to power flow on the DC line from bus $i$ to bus $j$ and $c_{ij}$ are the constant losses.
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
    A modified IEEE test system [2] illustrates the conditions for which optimal sizing and placement may be required, specifically there is insufficient generation capacity, voltage support, and line capacity for optimal powerflow feasibility. In addition, the generator cost function is reduced to a unity linear function.
    """)
    return


@app.cell
def _(solvers):
    #name,line_order = "case4r",[0,2,1,3,4])
    #name,line_order = "case9m",[3,0,1,2,4,5,6,7,8])
    name,line_order = "case9dc",None
    case = solvers.load(name)
    return case, line_order, name


@app.cell
def _(case, show_data, solvers):
    base_solution = solvers.full_acpf(case)
    show_data(base_solution["solution"],"Table 1: Base case data")
    return (base_solution,)


@app.cell
def _(case, line_order, mo, solvers):
    mo.mermaid(solvers.as_mermaid(case,line_order=line_order))
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
def _(base_solution, show_violations):
    # assert not solvers.violations(case,), "base case has unexpected violations"
    show_violations(base_solution,caption="Table 2: Base case violations")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial AC Powerflow Solution
    """)
    return


@app.cell
def _(case, solvers):
    initial_powerflow = solvers.full_acpf(case)
    return (initial_powerflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The initial full AC power flow solved using PyPower [3] converges with the results shown in Table 2(a). It satisfies Feasible Set 1 but it does not satisfy Feasible Set 3, as shown in Table 2(b).
    """)
    return


@app.cell
def _(initial_powerflow, mo):
    mo.md(f"""
    Initial powerflow {initial_powerflow["status"].lower()}.
    """)
    return


@app.cell
def _(initial_powerflow, line_order, mo, name, solvers):
    mo.vstack([mo.mermaid(solvers.as_mermaid(initial_powerflow["solution"],line_order=line_order)),
               mo.md(f"Figure 1: Network diagram of `{name}` as solved by `pypower`")
              ])
    return


@app.cell
def _(initial_powerflow, name, show_data):
    show_data(
        initial_powerflow["solution"],
        caption=f"Table 2(a): Initial full AC powerflow solution of `{name}`",
    )
    return


@app.cell(hide_code=True)
def _(initial_powerflow, name, show_violations):
    show_violations(
        initial_powerflow["solution"],
        caption=f"Table 2(b): Constraint violations after full AC powerflow solution of `{name}`",
    )
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
def _(initial_opf, mo):
    mo.md(rf"""
    The full AC OPF in `pypower` reports that it "{initial_opf["status"].lower()}" as presented in the base case, and the solution's constraint violations are shown in Table 3.
    """)
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
    The decoupled optimal powerflow problem must also satisfy Feasible Set 4 and the solver reports that it is {decoupled_opf["status"].lower()} as presented in the base case.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal Sizing and Placement

    This section discusses how an infeasible optimal powerflow problem is converted to a feasible optimal sizing and placement (OSP) problem by following softening of constraints in Feasible Set 4. The capacity expansion strategy only allows for the expansion of existing capacity, and does not allow for the addition of new generators, the addition of capacitors or condensors at non-PQ busses, or the addition of new powerlines. The OSP problem also assumes that all defined resources are in service regardless of their status in the base case.

    The following constraints are softening by converting them to costs per-unit of generation capacity expansion cost.

    1. A capacitor or condensors of size $c_i$ can be added at the PQ bus $i$ at the cost $\alpha c_i$ to raise or lower, respectively, the voltage $v_i$ when $|v_i|$ is outside the range of $(\underline v_i,\bar v_i)$.

    2. The generation real power $g_i$ and reactive power $h_i$ can be added to the non-PQ bus $i$ at the cost $\beta g$ when $p_i$ and/or $q_i$ are outside the ranges $(0,\bar p_i)$ and $(\underline q_i,\bar q_i)$, respectively, provided that $p_i + g_i <= q_i + h_i$.

    3. Transformer and powerline capacity can be increased by $d_{ij}$ at the costs $\gamma_t d_{ij}$ and $\gamma_l d_{ij}$, respectively, on branches where $|p_{ij}|$ exceeds $\bar s_{ij}$.

    4. The load on PQ busses is increased to $(p_i+jq_i)(1+e_i)$ to ensure that the capacity expansions $c$, $d$, $g$ and $h$ provide a sufficient safety margin.

    The costs and margins of these softened constraints as shown in Table 3.

    Table 3: Optimal Size and Placement Softening Parameters

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
        & \underline v_i \le |v_i| \le \bar v_i & (5b) \\
        & p_i = p_j + a_{ij} p_{ij} + c_{ij} & (6)
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(case, mo, solvers):
    decoupled_osp = solvers.decoupled_acoce(case)
    mo.md(rf"""
    and the decoupled AC OSP reports the result is "{decoupled_osp["status"].lower()} with the results shown in Tables (4a) and (4b)".
    """)
    return (decoupled_osp,)


@app.cell
def _(decoupled_osp, show_data):
    show_data(decoupled_osp["solution"],caption="Table 4(a): Decoupled AC optimal sizing and placement solution") if decoupled_osp["ok"] else None
    return


@app.cell
def _(decoupled_osp, name, show_data, solvers):
    case_solution = solvers.full_acpf(decoupled_osp["solution"], VERBOSE=0, OUT_ALL=0)
    assert (
        solvers.violations(case_solution, formatter="counter") == 0
    ), f"unexpected violations in {name}"
    show_data(case_solution["solution"],caption="Table 4(b): Full AC powerflow solution of optimal sizing and placement solution")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the difference in the generation dispatch is due to the 15% load margin applied during the sizing and placement problem, which is not present in the powerflow problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # WECC 240 Model

    This section applies the OCE method described above to a modified WECC 240-bus model that cannot be otherwise solved.
    """)
    return


@app.cell
def _(solvers):
    wecc=solvers.load("case240_2011")
    return (wecc,)


@app.cell
def _(mo, solvers, wecc):
    optimized = solvers.decoupled_acoce(wecc)
    solution = solvers.full_acpf(optimized["solution"])
    mo.accordion(solvers.as_frames(solution["solution"]))
    return (solution,)


@app.cell
def _(solution, solvers):
    solvers.violations(solution["solution"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusions

    This notebook shows how the canonical optimal powerflow problem can be easily adapted to solve an optimal sizing and placement problem that is practical for ensuring that a large complex network will solve without violating operating limits for generators and powerlines.
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
    def show_data(case, caption=""):
        data = solvers.as_frames(case)
        result = mo.vstack(
            [mo.md(caption)]
            + [mo.md(f"**{x}**\n~~~\n{repr(y)}\n~~~\n") for x, y in data.items()]
        )
        return mo.md(f"{caption}\n~~~\nNo solution\n~~~") if not data else result


    def show_violations(case, caption=""):
        return mo.vstack(
            [
                mo.md(caption),
                mo.md(
                    f"~~~\n{solvers.violations(case,precision=1,formatter="table")}\n~~~"
                ),
            ]
        )

    return show_data, show_violations


@app.cell
def _():
    import marimo as mo
    import solvers
    import pandas as pd
    pd.options.display.width = None
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    return mo, solvers


if __name__ == "__main__":
    app.run()
