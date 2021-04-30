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
    using MultiUAVDelivery
    using Random
    using Statistics
    using Distributions
    include("pipeline.jl")
end

const PSET = [(nagents=8, XY_AXIS_RES=0.2, XYDOT_LIM=0.2, XYDOT_STEP=0.2, NOISESTD=0.1),
               (nagents=16, XY_AXIS_RES=0.1, XYDOT_LIM=0.1, XYDOT_STEP=0.1, NOISESTD=0.05),
               (nagents=32, XY_AXIS_RES=0.08, XYDOT_LIM=0.08, XYDOT_STEP=0.08, NOISESTD=0.05),
               (nagents=48, XY_AXIS_RES=0.05, XYDOT_LIM=0.05, XYDOT_STEP=0.05, NOISESTD=0.02)]

const REWSET = (goal_bonus=1000.0, prox_scaling = 1.0, repul_pen=10.0, dynamics_scaling=10.0)

const MAX_STEPS = 200

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


function run_experiment(prob, solver_name, nevals, logdir)
    mdp = FirstOrderMultiUAVDelivery(pset=PSET[prob], rewset=REWSET)

    psetvals = join(("$p1=$p2" for (p1, p2) in pairs(PSET[prob])), ",")
    rsetvals = join(("$p1=$p2" for (p1, p2) in pairs(REWSET)), ",")
    exp_name = "UAV_delivery_$(psetvals)_$(rsetvals)_$(solver_name)"
    if solver_name == :random
        solver = RandomSolver(rng=MersenneTwister(1999))
        @info exp_name
        df = online_evaluate(Dict(solver_name=>solver), mdp, nevals, MAX_STEPS)
        CSV.write(joinpath(logdir, "$(exp_name).csv"), df)
        for k in unique(df.policy)
            fdf = filter(r->r[:policy] == k, df)
            vals = fdf.discret
            uvals = fdf.undiscret
            @info k mean=mean(vals) std=std(vals)
            @info k mean=mean(uvals) std=std(uvals)
        end
        return nothing
    end

    d = 10
    n = 5*PSET[prob].nagents

    if prob == 1
        c = 5
        k = 10
    elseif prob == 2
        c = 10
        k = 20
    elseif prob == 3
        c = 20
        k = 20
    else
        c = 30
        k = 20
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
    prob = parse(Int, args[1])
    solver = Symbol(args[2])
    nevals = parse(Int, args[3])

    if length(args) < 4
        logdir = "."
    else
        logdir = args[4]
    end
    !isdir(logdir) && mkpath(logdir)
    run_experiment(prob, solver, nevals, logdir)
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
