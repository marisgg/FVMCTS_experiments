# FVMCTS_experiments

## Setup
Ensure you have julia installed. Then clone this repository somewhere:

```bash
git clone https://github.com/rejuvyesh/FVMCTS_experiments
```

Ensure that you are in the experiment directory:
```bash
cd FVMCTS_experiments
```

Then run in the terminal:
```bash
julia --project 'using Pkg; Pkg.instantiate(); Pkg.resolve()'
```

Now you are all set to run the scripts for running the experiments.

## SysAdmin

For evaluating the `random` planner on the `ring` sysadmin domain with `global` rewards for `8` agents over `100` runs, and storing the results in `/tmp/logdir`:
```bash
julia --project run_sysadmin.jl ring_global 8 random 100 /tmp/logdir
```

Similarly, for evaluating `maxplus` planner, with tree search depth as `3`, number of iterations of tree search as `20`, exploration constant as `5`, and maximum number of message passing iterations as `10`:
```bash
julia --project run_sysadmin.jl ring_global 8 maxplus 100 /tmp/logdir 3 20 5 10
```

For evaluating `varel` planner with tree search depth as `3`, number of iterations of tree search as `20`, and exploration constant as `5`:
```bash
julia --project run_sysadmin.jl ring_global 8 varel 100 /tmp/logdir 3 20 5
```

## UAV Delivery

For evaluating the `random` planner on problem `1` (8 agents) over `100` runs, and stroing the results in `/tmp/logdir`:
```bash
julia --project run_uav.jl 1 random 100 /tmp/logdir
```

Similarly, for evaluating the `maxplus` planner with tree search hyperparameters as used in the paper:
```bash
julia --project run_uav.jl 1 maxplus 100 /tmp/logdir
```
and `varel` planner:
```bash
julia --project run_uav.jl 1 varel 100 /tmp/logdir
```

**Note**: `varel` memory requirements are quite high and may run out of memory on larger problems.