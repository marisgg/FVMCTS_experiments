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
    using MultiAgentSysAdmin: StarSysAdmin
    using Random
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

function run_experiment(prob, nagents, solver_name, nevals, d, n, c, k, logdir)
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
            @info "discounted" k mean=mean(vals) std=std(vals)
            @info "undiscounted" k mean=mean(uvals) std=std(uvals)
        end
        return nothing
    end

    if solver_name == :maxplus
        exp_name *= "_d=$(d),n=$(n),c=$(c),k=$(k)"
        solver = get_maxplus_solver(d, n, c, k, true, true, false)
    elseif solver_name == :varel
        exp_name *= "_d=$(d),n=$(n),c=$(c)"
        solver = get_varel_solver(d, n, c)
    end
    @info exp_name    
    df = online_evaluate(Dict(solver_name=>solver), mdp, nevals, MAX_STEPS)
    CSV.write(joinpath(logdir, "$(exp_name).csv"), df)
    for k in unique(df.policy)
        fdf = filter(r->r[:policy] == k, df)
        vals = fdf.discret
        uvals = fdf.undiscret
        @info "discounted" k mean=mean(vals) std=std(vals)
        @info "undiscounted" k mean=mean(uvals) std=std(uvals)
    end
end

function main(args)
    prob = args[1]

    nagents = parse(Int, args[2])

    solver = Symbol(args[3])

    nevals = parse(Int, args[4])
    
    logdir = args[5]

    !isdir(logdir) && mkpath(logdir)

    if solver == :random
        d = 0
        n = 0
        c = 0
        k = 0
    else
        d = parse(Int, args[6])

        n = parse(Int, args[7])

        c = parse(Int, args[8])
        
        k = parse(Int, args[9])    
    end
    run_experiment(prob, nagents, solver, nevals, d, n, c, k, logdir)
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)