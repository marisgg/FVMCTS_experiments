#!/bin/bash

n_agents="4 10 16"

n_sims="10 25 50 100 1000 10000"

set -u

env=${1}

n_episodes=100
c=5
mp_iters=20
d=5

directory="./reproduce/${env}"

for n in "${n_agents[@]}"
do
    for s in "${n_sims[@]}"
    do
        echo $
        set -x
        julia --project run_sysadmin.jl ${env} random
        julia --project run_sysadmin.jl ${env} maxplus ${n_episodes} ${directory} ${d} ${s} ${c} ${mp_iters}
        julia --project run_sysadmin.jl ${env} varel ${n_episodes} ${directory} ${d} ${s} ${c} ${mp_iters}
        set +x
    done
done

echo "INFO: Finished running experiments for ${env}. Results are in ${directory}."
