"""Single case test script"""

import os
import sys

import pandas as pd
import numpy as np

from solvers import *

def main(*args,**kargs):

    test = args[0].replace(".py","")

    print("*"*(len(test)+8),f"*** {test} ***","*"*(len(test)+8),"",sep="\n",flush=True)
    
    case = load(test)

    result = full_acpf(case,VERBOSE=0,OUT_ALL=0)["solution"]
    print(f"Base {test} violations....",violations(result,formatter="counter"))
    if violations(result,formatter="counter"):
        print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

    result = full_acopf(case,VERBOSE=0,OUT_ALL=0)
    print(f"""Initial AC OPF.......... {result["status"]} in {result["time"]:.2f} s""",flush=True)
    if result["ok"] and violations(result,formatter="counter"):
        print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")
    # print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(result["solution"]).items()],sep="\n",flush=True)

    result = decoupled_acopf(case)
    print(f"""Initial FD OPF.......... {result["status"]} in {result["time"]:.2f} s""",flush=True)
    if result["ok"]:
        if violations(result,formatter="counter"):
            print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")
        # print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(result["solution"]).items()],sep="\n",flush=True)
    # else:
    #     print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(result["case"]).items()],sep="\n",flush=True)
    #     print(internals(result))
    if result["warnings"]:
        print("WARNINGS:",*result["warnings"],sep="\n  - ")

    if result["ok"]:
        result = full_acpf(case,VERBOSE=0,OUT_ALL=0)
        print(f"""Initial AC PF........... {result["status"]} in {result["time"]:.2f} s""",flush=True)
        if result["ok"] and violations(result,formatter="counter"):
            print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

    if not result["ok"] or violations(result,formatter="counter") > 0:
        fast_oce = decoupled_acoce(case)
        print(f"""Fast AC OCE............. {fast_oce["status"]} in {fast_oce["time"]:.2f} s""",flush=True)
        if not fast_oce["ok"]:

            print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(case).items()],sep="\n")
            print(internals(fast_oce))

        else:

            if fast_oce["updates"]:
                print("  Updates:",*fast_oce["updates"],sep="\n  - ")

            # print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(fast_oce["solution"]).items()],sep="\n")

            result = full_acopf(fast_oce["solution"],VERBOSE=0,OUT_ALL=0)
            print(f"""Final AC OPF............ {result["status"]} in {result["time"]:.2f} s""",flush=True)
            if result["ok"] and violations(result,formatter="counter"):
                print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

            result = decoupled_acopf(fast_oce["solution"])
            print(f"""Final FD OPF............ {result["status"]} in {result["time"]:.2f} s""",flush=True)
            if result["ok"]:
                if violations(result,formatter="counter"):
                    print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")
            else:
                print(internals(result,all=True))

            result = full_acpf(fast_oce["solution"],VERBOSE=0,OUT_ALL=0)
            print(f"""Final AC PF............. {result["status"]} in {result["time"]:.2f} s""",flush=True)
            if result["ok"] and violations(result,formatter="counter"):
                print("",*violations(result,formatter="table").split("\n"),"",sep="\n  ")

            print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(result["solution"]).items()],sep="\n")
    # else:
    #     print(*[f"\n*** {x} ***\n\n{y}" for x,y in as_frames(result["solution"]).items()],sep="\n",flush=True)

if __name__ == "__main__":

    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.width = None

    np.set_printoptions(
        linewidth=9999999,
        formatter={'float':lambda x:f"{x: 8.4f}" if x else f"{0:8.0f}"}
        )
    if len(sys.argv) > 1:
        files = sys.args[1:]
    else:
        files = sorted([x for x in os.listdir() if x.startswith("case") and x.endswith(".py")])
    for file in files:
        main(file)

