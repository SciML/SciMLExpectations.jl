abstract type AbstractExpectationAlgorithm <: DiffEqBase.DEAlgorithm end
struct Koopman <:AbstractExpectationAlgorithm end
struct MonteCarlo <: AbstractExpectationAlgorithm end

# tuplejoin from https://discourse.julialang.org/t/efficient-tuple-concatenation/5398/8
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

_rand(x::T) where T <: Sampleable = rand(x)
_rand(x) = x

function expectation(g::Function, prob::ODEProblem, u0, p, expalg::Koopman, args...;
                        u0_func=(u,p)->u, p_func=(u,p)->p,
                        maxiters=0,
                        batch=0,
                        quadalg=HCubatureJL(),
                        ireltol=1e-2, iabstol=1e-2,
                        nout=1,kwargs...)

    S = function (u,p)
        solve(remake(prob,u0=u,p=p),
                      args...; kwargs...)
    end

    expectation(g, S, u0, p, expalg, args...; u0_func=u0_func, p_func=p_func, kwargs...)

end

function expectation(g::Function, S::Function, u0, p, expalg::Koopman, args...;
                        u0_func=(u,p)->u, p_func=(u,p)->p,
                        maxiters=0,
                        batch=0,
                        quadalg=HCubatureJL(),
                        ireltol=1e-2, iabstol=1e-2,
                        nout=1,kwargs...)

     # construct extended state space
    n_states = length(u0)
    n_params = length(p)
    ext_state = [u0; p]

    # find indices corresponding to distributions, check if sampleable and has non-zero support.
    dist_mask = collect(isa.(ext_state, Sampleable) .& (minimum.(ext_state) .!= maximum.(ext_state)))
    val_mask = .!(dist_mask)

    # get distributions and indx in extended state space
    dists = ext_state[dist_mask]

    # create numerical state space values
    ext_state_val = minimum.(ext_state)
    state_view = @view ext_state_val[dist_mask]
    # param_view = @view ext_state_val[val_mask]

    integrand = function (x, p)
        ## Hack to avoid mutating array replacing ext_state_val[ext_state_dist_bitmask] .= x
        x_it = 0
        p_it = 0
        T = promote_type(eltype(x),eltype(p))
        esv = map(1:length(ext_state_val)) do idx
            dist_mask[idx] ? T(x[x_it+=1]) : T(p[p_it+=1])
        end

        _u0 = @view(esv[1:n_states])
        _p = @view(esv[n_states+1:end])

        # set values for indices corresponding to random variables
        # state_view .= x
        # _u0 = @view(ext_state_val[1:n_states])
        # _p = @view(ext_state_val[n_states+1:end])

        # Koopman
        w = prod(pdf(a, b) for (a, b) in zip(dists, x))
        Ug = g(S(u0_func(_u0,_p), p_func(_u0,_p)))

        return Ug*w
    end

    # TODO fix params usage
    lb = minimum.(dists)
    ub = maximum.(dists)
    T = promote_type(eltype(p),eltype(lb),eltype(ub))
    intprob = QuadratureProblem(integrand, T.(lb), T.(ub), T.(p), batch=batch, nout=nout)
    sol = solve(intprob, quadalg, reltol=ireltol, abstol=iabstol, maxiters=maxiters)

    sol
end

function expectation(g::Function, prob::ODEProblem, u0, p, expalg::MonteCarlo, args...;
                        trajectories,
                        u0_func=(u,p)->u, p_func=(u,p)->p,
                        kwargs...)

    prob_func = function (prob, i, repeat)
        _u0 = _rand.(u0)
        _p = _rand.(p)
        remake(prob, u0=u0_func(_u0,_p), p=p_func(_u0,_p))
    end

    output_func = (sol, i) -> (g(sol), false)

    monte_prob = EnsembleProblem(prob;
                                 output_func=output_func,
                                 prob_func=prob_func)
    sol = solve(monte_prob, args...;trajectories=trajectories,kwargs...)
    mean(sol.u)# , sol
end

function expectation(g::Function, S::Function, u0, p, expalg::MonteCarlo, args...;
                        trajectories,
                        u0_func=(u,p)->u, p_func=(u,p)->p,
                        kwargs...)

    function mc_run(u0,p)
        _u0 = _rand.(u0)
        _p = _rand.(p)
        g(S(u0_func(_u0,_p), p_func(_u0,_p)))
    end

    tot = mc_run(u0,p)
    for ii in 2:trajectories
        tot += mc_run(u0,p)
    end
    tot/trajectories
end

function koopman_expectation(g,u0s,ps,prob,ADparams,args...;maxiters=0,
                      batch=0,
                      quadalg=HCubatureJL(),
                      ireltol=1e-2, iabstol=1e-2,
                      nout=1,
                      u0s_func=identity, ∂dist=false, kwargs...)

    n_states = length(u0s)

    # find indices corresponding to distributions, check if sampleable and has non-zero support.
    # ext_state = vcat(u0s,ps) #tuplejoin(u0s, ps)
    ext_state = ArrayPartition(u0s, ps)
    ext_state_dist_bitmask = collect(isa.(ext_state, Sampleable) .& (minimum.(ext_state) .!= maximum.(ext_state)))
    ext_state_val_bitmask = .!(ext_state_dist_bitmask)

    # get distributions and indx in extended state space
    dist_idx =  (1:length(ext_state))[ext_state_dist_bitmask]
    # dists = ext_state[dist_idx]
    dists = [ext_state[idx] for idx ∈ dist_idx ]# ext_state[dist_idx]

    # Define the integrand for expectation
    if batch == 0
      # create numerical state space values
        # ext_state_val = vcat(zero.(eltype.(ext_state))...)
        # ext_state_val[ext_state_val_bitmask] .= minimum.(ext_state[ext_state_val_bitmask])   # minimum used to extract value if Dirac or a number type

        # create numerical state space values


        ext_state_val = Array([minimum(es) for es ∈ ext_state])
        @show typeof(ext_state_val)

        integrand = function (x, p)
            # ext_state_val[dist_idx] .= x        # set values for indices corresponding to random variables

             ## Hack to avoid mutating array replacing ext_state_val[ext_state_dist_bitmask] .= x
            x_it = Iterators.Stateful(deepcopy(x));
            esv = [ext_state_dist_bitmask[idx] == true ? popfirst!(x_it) : ext_state_val[idx] for idx ∈ 1:length(ext_state_val)]

            w = prod(pdf(a, b) for (a, b) in zip(dists, x))
            sol = solve(remake(prob,u0=u0s_func(@view(esv[1:n_states])),
                      p=@view(esv[(n_states + 1):end])),
                      args...; kwargs...)

            k = g(sol)

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
    intprob = QuadratureProblem(integrand, minimum.(dists), maximum.(dists), ADparams, batch=batch, nout=nout)

    sol = solve(intprob,quadalg,reltol=ireltol,
                abstol=iabstol,maxiters=maxiters)
    @show sol.u[1]
    # return integrand(rand.(u0s),ps)

    sol[1]# ,sols,ks
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
