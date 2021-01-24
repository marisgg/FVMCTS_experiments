using POMDPs
using POMDPSimulators
using StructArrays

function discounted_avg_return(h::AbstractSimHistory)
    γ = discount(h)
    r_total = sum(r.*γ^(i-1) for (i, r) in enumerate(h[:r]))
    return sum(r_total)/length(r_total)
end

function undiscounted_avg_return(h::AbstractSimHistory)
    r_total = sum(r for r in h[:r])
    return sum(r_total) / length(r_total)
end

function online_evaluate(solvers, mdp, nevals, max_steps)
    sims = Sim[]
    for (sname, solver) in solvers
        for i in 1:nevals
            push!(sims, Sim(mdp, solve(solver, mdp);
                            rng=MersenneTwister(42 + i),
                            max_steps=max_steps,
                            metadata=Dict(:policy=>sname))
                  )
        end
    end

    # Note: some bug in progress_pmap
    df = run_parallel(sims; show_progress=false) do sim, hist
        result = (nsteps=length(hist), discret=discounted_avg_return(hist), undiscret=undiscounted_avg_return(hist))
        # Note: Because this is being done lazily on the backend
        if :info in keys(hist[1])
            return merge(result, map(sum, fieldarrays(StructArray(hist[:info]))))
        end
        return result
    end

    return df
end

function offline_evaluate(solvers, mdp, nsolves, nevals, max_steps)
    sims = Sim[]
    for (sname, solver) in solvers
        for j in 1:nsolves
            sol = solve(solver, mdp)
            for i in 1:nevals
                push!(sims, Sim(mdp, sol; rng=MersenneTwister(42 +i),
                                max_steps=max_steps,
                                metadata=Dict(:policy=>sname))
                      )
            end
        end
    end
    
    df = run_parallel(sims; show_progress=false) do sim, hist
        result = (nsteps=length(hist), discret=discounted_avg_return(hist), undiscret=undiscounted_avg_return(hist))
        # Note: Because this is being done lazily on the backend
        if :info in keys(hist[1])
            return merge(result, map(sum, fieldarrays(StructArray(hist[:info]))))
        end
        return result
    end

    return df
end
