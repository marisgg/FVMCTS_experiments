using Distributed
addprocs(4)

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

sleep(0.1)

@everywhere begin
    using CSV
    using POMDPs
    using POMDPPolicies
    using POMDPSimulators
    using MultiAgentPOMDPs
    using FactoredValueMCTS
    using MultiAgentSysAdmin

    using Statistics
    using Distributions
    include("pipeline.jl")
end

const MAX_STEPS = 100

function get_maxplus_solver(d, n, c, k, agent_utils, node_exp, edge_exp)
    solver = FVMCTSSolver(
        depth=d,
        n_iterations=n,
        exploration_constant=c,
        rng=MersenneTwister(8),
        coordination_strategy = MaxPlus(
            message_iters=k,
            message_norm=true,
            use_agent_utils=agent_utils,
            node_exploration=node_exp,
            edge_exploration=edge_exp,
        )
    )
    return solver
end

function get_varel_solver(d, n, c)
    solver = FVMCTSSolver(depth=d,
                             n_iterations=n,
                             exploration_constant=c,
                             rng=MersenneTwister(8),
                             coordination_strategy=VarEl())
    return solver
end

const PSET = Dict(
    "ring_global" => BiSysAdmin{true},
    "ring_local" => BiSysAdmin{false},
    "star_global" => StarSysAdmin{true},
    "star_local" => StarSysAdmin{false},
    "ringofring_global" => RingofRingSysAdmin{true},
    "ringofring_local" => RingofRingSysAdmin{false},
)

function run_experiment(prob, nagents, solver_name, nevals, logdir)
    if startswith(prob, "ringofring")
        # we build rings of rings where each ring is of size 3!
        @assert nagents % 3 == 0
        mdp = PSET[prob](nagents=nagents//3, nagents_per_ring=3)
    else
        mdp = PSET[prob](nagents=nagents)
    end
    
    exp_name = "SysAdmin_$(prob)_nagents=$(nagents)_$(solver_name)"
    if solver_name == :random
        solver = RandomSolver(rng=MersenneTwister(1999))
        @info exp_name
        df = online_evaluate(Dict(solver_name=>solver), mdp, nevals, MAX_STEPS)
        CSV.write(joinpath(logdir, "$(exp_name).csv"), df)
        for k in unique(df.policy)
            fdf = filter(r->r[:policy] == k, df)
            vals = fdf.discret
            uvals = fdf.undiscret
            @info k (mean=mean(vals), std=std(vals))
            @info k (mean=mean(uvals), std=std(uvals))
        end
        return nothing
    end

    exp_name *= "_d=$(d),n=$(n),c=$(c),k=$(k)"
    if solver_name == :maxplus
        solver = get_maxplus_solver(d, n, c, k, true, true, false)
    elseif solver_name == :varel
        solver = get_varel_solver(d, n, c)
    end
    @info exp_name    
    df = online_evaluate(Dict(solver_name=>solver), mdp, nevals, MAX_STEPS)
    CSV.write(joinpath(logdir, "$(exp_name).csv"), df)
    for k in unique(df.policy)
        fdf = filter(r->r[:policy] == k, df)
        vals = fdf.discret
        uvals = fdf.undiscret
        @info k (mean=mean(vals), std=std(vals))
        @info k (mean=mean(uvals), std=std(uvals))
    end
end

function main(args)
end