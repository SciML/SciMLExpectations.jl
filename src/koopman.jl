# tuplejoin from https://discourse.julialang.org/t/efficient-tuple-concatenation/5398/8
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

_rand(x::T) where T <: Sampleable = rand(x)
_rand(x) = x

function koopman(g, prob, u0, p, args...; u0s_func=identity, kwargs...)
    g(solve(remake(prob, u0=u0s_func(u0), p=p), args...;kwargs...))
end

function montecarlo_expectation(g,u0s,ps,prob,args...;
                                trajectories,u0s_func=identity,kwargs...)

    prob_func = function (prob, i, repeat)
        remake(prob, u0=u0s_func(_rand.(u0s)), p=_rand.(ps))
    end
    output_func = (sol, i) -> (g(sol), false)
    monte_prob = EnsembleProblem(prob;
                                 output_func=output_func,
                                 prob_func=prob_func)
    sol = solve(monte_prob, args...;trajectories=trajectories,kwargs...)
    mean(sol.u)# , sol
end

function koopman_expectation(g,u0s,ps,prob,ADparams,args...;maxiters=0,
                      batch=0,
                      quadalg=HCubatureJL(),
                      ireltol=1e-2, iabstol=1e-2,
                      nout=1,
                      u0s_func=identity, ∂dist=false, kwargs...)

    n_states = length(u0s)

    # find indices corresponding to distributions, check if sampleable and has non-zero support.
    ext_state = tuplejoin(u0s, ps)
    ext_state_dist_bitmask = collect(isa.(ext_state, Sampleable) .& (minimum.(ext_state) .!= maximum.(ext_state)))
    ext_state_val_bitmask = .!(ext_state_dist_bitmask)
    @show ext_state_dist_bitmask
    # get distributions and indx in extended state space
    dist_idx =  (1:length(ext_state))[ext_state_dist_bitmask]
    dists = ext_state[dist_idx]

    # Define the integrand for expectation
    if batch == 0
      # create numerical state space values
        ext_state_val = vcat(zero.(eltype.(ext_state))...)
        ext_state_val[ext_state_val_bitmask] .= minimum.(ext_state[ext_state_val_bitmask])   # minimum used to extract value if Dirac or a number type
        @show ext_state_val
        integrand = function (x, p)
            ext_state_val[dist_idx] .= x        # set values for indices corresponding to random variables
            
            w = prod(pdf(a, b) for (a, b) in zip(dists, x))
            sol = solve(remake(prob,u0=u0s_func(@view(ext_state_val[1:n_states])),
                      p=@view(ext_state_val[(n_states + 1):end])),
                      args...;u0s_func=u0s_func, kwargs...)

            k = g(sol)
          # @show k*w
          # @show w, g(sol)
            return k * w
        end
    else
        integrand = function (dx, x, p)
            trajectories = size(x, 2)

            ext_state_val = zeros(length(ext_state), trajectories)
            ext_state_val[ext_state_val_bitmask,:] .= minimum.(ext_state[ext_state_val_bitmask])   # minimum used to extract value if Dirac or a number type
            ext_state_val[dist_idx,:] = x

            prob_func = (prob, i, repeat) -> remake(prob, u0=u0s_func(@view(ext_state_val[1:n_states,i])),
                                                       p=@view(ext_state_val[n_states + 1:end,i]))
            output_func = function (sol, i)
                w = prod(pdf(a, b) for (a, b) in zip(dists, x[:,i]))
                k = g(sol)
                return k * w, false
            end

            ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)
            sol = solve(ensembleprob, args...;trajectories=trajectories,kwargs...)


            dx .= hcat(sol.u...) # Why do I need to hcat???
        end
    end

    # solve
    @show typeof(minimum.(dists)), typeof(minimum.(dists))
    @show typeof(ADparams)
    intprob = QuadratureProblem(integrand, minimum.(dists), maximum.(dists), ADparams, batch=batch, nout=nout)
    sol = solve(intprob,quadalg,reltol=ireltol,
                abstol=iabstol,maxiters=maxiters)

    sol# ,sols,ks
end

using RecursiveArrayTools
function koopman_expectation2(g,u0s_f,ps_f, params,prob,args...;maxiters=0,
                              batch=0,
                              quadalg=HCubatureJL(),
                              ireltol=1e-2, iabstol=1e-2,
                              nout=1,
                              u0s_func=identity, kwargs...)

    # construct 
    u0s = u0s_f(params)
    ps = ps_f(params)
    n_states = length(u0s)

    # find indices corresponding to distributions, check if sampleable and has non-zero support.  
    ext_state = ArrayPartition(u0s, ps)
    ext_state_dist_bitmask = collect(isa.(ext_state, Sampleable) .& (minimum.(ext_state) .!= maximum.(ext_state)))

    # get distributions and indx in extended state space
    dist_idx =  (1:length(ext_state))[ext_state_dist_bitmask]
    dists = [ext_state[idx] for idx ∈ dist_idx ]# ext_state[dist_idx]

    # create numerical state space values
    ext_state_val = [minimum(es) for es ∈ ext_state]

    integrand = function (x, p)
        
        ## Hack to avoid mutating array replacing ext_state_val[ext_state_dist_bitmask] .= x
        x_it = Iterators.Stateful(deepcopy(x));
        esv = [ext_state_dist_bitmask[idx] == true ? popfirst!(x_it) : ext_state_val[idx] for idx ∈ 1:length(ext_state_val)]

        # set values for indices corresponding to random variables
        w = prod(pdf(a, b) for (a, b) in zip(dists, x))

        sol = solve(remake(prob,u0=u0s_func(@view(esv[1:n_states])),
                            p=@view(esv[n_states+1:end])),
                            args...; kwargs...)

        k = g(sol)
        return k*w
    end
  
    intprob = QuadratureProblem(integrand, minimum.(dists), maximum.(dists), params, batch=batch, nout=nout)
    sol = solve(intprob, quadalg, reltol=ireltol, abstol=iabstol, maxiters=maxiters)

    sol
end

