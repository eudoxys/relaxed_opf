Optimal capacity expansion using softened relaxed optimal powerflow
-------------------------------------------------------------------

David P. Chassin, Eudoxys Sciences LLC

April 2026

Citation: D.P. Chassin, "Ensuring optimal powerflow feasibility using convext optimization", April 2026, https://github.com/eudoxys/relaxed_opf.

----

This notebook presents a method for solving the problem of expanding real and
reactive power resources on an electric network as a relaxation of the
optimal power flow problem with softened constraints on loads (i.e., addition
of static VAR devices), generation capacities (i.e., generator and substation
capacity expansion), and line flows (i.e., transmission line capacity
expansion).

The relaxations enable the use of convex optimization solvers as illustrated
here using [`cvxpy`](https://www.cvxpy.org/). These are discussed in the
first section.  The softening of constraints converts the problem from a
canonical optimal powerflow problem to an optimal capacity expansion problem,
which is discussed in the second section.

The method does not support adding generators to PQ busses nor does it support
adding new transmission lines or transformers where none are already present
as one might find in an optimal sizing and placement problem.

Quick Start
-----------

    ./notebook.sh
