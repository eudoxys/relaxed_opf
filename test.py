"""Single case test script"""

test = "case300"

import pandas as pd
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.width = None

import numpy as np
np.set_printoptions(
    linewidth=9999999,
    formatter={'float':lambda x:f"{x: 8.4f}" if x else f"{0:8.0f}"}
    )

from solvers import *
case = load(test)

result = full_acpf(case,VERBOSE=0,OUT_ALL=0)["solution"]
print(f"Base {test} violations....",violations(result,formatter="counter"))
if violations(result,formatter="counter"):
    print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

result = full_acopf(case,VERBOSE=0,OUT_ALL=0)
print(f"""Initial AC OPF.......... {result["status"]} in {result["time"]:.2f} s""",flush=True)
if result["ok"] and violations(result,formatter="counter"):
    print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

result = decoupled_acopf(case)
print(f"""Initial FD OPF.......... {result["status"]} in {result["time"]:.2f} s""",flush=True)
if result["ok"] and violations(result,formatter="counter"):
    print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")
if result["warnings"]:
    print("WARNINGS:",*result["warnings"],sep="\n  - ")
if not result["ok"]:
    print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(result["case"]).items()],sep="\n")
    print(internals(result))
    quit()

if result["ok"]:
    result = full_acpf(case,VERBOSE=0,OUT_ALL=0)
    print(f"""Initial AC PF........... {result["status"]} in {result["time"]:.2f} s""",flush=True)
    if result["ok"] and violations(result,formatter="counter"):
        print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

if not result["ok"] or violations(result,formatter="counter") > 0:
    fast_osp = decoupled_acosp(case)
    print(f"""Fast AC OSP............. {fast_osp["status"]} in {fast_osp["time"]:.2f} s""",flush=True)
    if not fast_osp["ok"]:

        print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(case).items()],sep="\n")
        print(internals(fast_osp))

    else:

        if fast_osp["updates"]:
            print("  Updates:",*fast_osp["updates"],sep="\n  - ")

        # print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(fast_osp["solution"]).items()],sep="\n")

        result = full_acopf(fast_osp["solution"],VERBOSE=0,OUT_ALL=0)
        print(f"""Final AC OPF............ {result["status"]} in {result["time"]:.2f} s""",flush=True)
        if result["ok"] and violations(result,formatter="counter"):
            print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

        result = decoupled_acopf(fast_osp["solution"])
        print(f"""Final FD OPF............ {result["status"]} in {result["time"]:.2f} s""",flush=True)
        if result["ok"]:
            if violations(result,formatter="counter"):
                print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")
        else:
            print(internals(result,all=True))

        result = full_acpf(fast_osp["solution"],VERBOSE=0,OUT_ALL=0)
        print(f"""Final AC PF............. {result["status"]} in {result["time"]:.2f} s""",flush=True)
        if result["ok"] and violations(result,formatter="counter"):
            print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

        print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(result["solution"]).items()],sep="\n")

        # print(internals(fast_osp))
