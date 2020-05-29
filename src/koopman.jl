@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

function koopman(g,prob,u0,p,args...; u0s_func = identity, kwargs...)
    g(solve(remake(prob,u0=u0s_func(u0),p=p),args...;kwargs...))
end

function koopman_expectation(g,u0s,ps,prob,args...;maxiters=0,
                      batch = 0,
                      quadalg = HCubatureJL(),
                      ireltol = 1e-2, iabstol=1e-2,
                      u0s_func = identity, kwargs...)
  n = length(u0s)
  if batch == 0
    _f = function (x,p)
      u0_params = x[1:n]
      u0 = u0s_func(u0_params)
      p  = x[n+1:end]
      k = koopman(g,prob,u0,p,args...;kwargs...)
      w = prod(pdf(a,b) for (a,b) in zip(u0s,u0_params))*
          prod(pdf(a,b) for (a,b) in zip(ps,p))
      k*w
    end
  else
    _f = function (dx,x,p)
      trajectories = size(x,2)
      prob_func = (prob,i,repeat) -> remake(prob,u0=u0s_func(x[1:n,i]),
                                                 p=x[n+1:end,i])
      output_func = function (sol,i)
        k = g(sol)
        u0= x[1:n,i]
        p = x[n+1:end,i]
        w = prod(pdf(a,b) for (a,b) in zip(u0s,u0))*
            prod(pdf(a,b) for (a,b) in zip(ps,p))
        k*w,false
      end

      ensembleprob = EnsembleProblem(prob,prob_func=prob_func,
                                     output_func = output_func)
      sol = solve(ensembleprob,args...;trajectories=trajectories,kwargs...)
      dx .= vec(sol.u)
    end
  end
  xs = [u0s;ps]
  intprob = QuadratureProblem(_f,minimum.(xs),maximum.(xs),batch=batch)
  sol = solve(intprob,quadalg,reltol=ireltol,
              abstol=iabstol,maxiters = maxiters)
end

function montecarlo_expectation(g,u0s,ps,prob,args...;
                                trajectories,u0s_func = identity,kwargs...)

  _rand(x::T) where T<:Sampleable = rand(x)
  _rand(x) = x

  prob_func = function (prob,i,repeat)
    remake(prob,u0=u0s_func(_rand.(u0s)),p=_rand.(ps))
  end
  output_func = (sol,i) -> (g(sol),false)
  monte_prob = EnsembleProblem(prob;
                                 output_func = output_func,
                                 prob_func = prob_func)
  mean(solve(monte_prob,args...;trajectories=trajectories,kwargs...).u)
end

function koopman_expectation2(g,u0s,ps,prob,args...;maxiters=0,
                      batch = 0,
                      quadalg = HCubatureJL(),
                      ireltol = 1e-2, iabstol=1e-2,
                      u0s_func = identity, kwargs...)

    n_states = length(u0s)

    # find indices corresponding to distributions, check if sampleable and has non-zero support.
    # ext_state = vcat(u0s, ps)
    ext_state = tuplejoin(u0s, ps)
    ext_state_dist_bitmask = isa.(ext_state,Sampleable) .& (minimum.(s for s ∈ ext_state) .!= maximum.( s for s ∈ ext_state))
    ext_state_val_bitmask = .!(ext_state_dist_bitmask)

    # get distributions and indx in extended state space
    dist_idx =  (1:length(ext_state))[ext_state_dist_bitmask]
    dists = ext_state[dist_idx]

    # Define the integrand for expectation
    if batch == 0
      # create numerical state space values
      ext_state_val = zeros(length(ext_state))
      ext_state_val[ext_state_val_bitmask] .= minimum.(ext_state[ext_state_val_bitmask])   # minimum used to extract value if Dirac or a number type

      integrand = function (x, p)
          ext_state_val[dist_idx] = x        # set values for indices corresponding to random variables
          w = prod(pdf(a, b) for (a, b) in zip(dists, x))
          k = koopman(g,prob, @view(ext_state_val[1:n_states]),
                              @view(ext_state_val[(n_states + 1):end]),
                              args...;u0s_func = u0s_func, kwargs...)
          return k*w
      end
    else
      integrand = function (dx, x, p)
          trajectories = size(x,2)

          ext_state_val = zeros(length(ext_state), trajectories)
          ext_state_val[ext_state_val_bitmask,:] .= minimum.(s for s ∈ ext_state[ext_state_val_bitmask])   # minimum used to extract value if Dirac or a number type
          ext_state_val[dist_idx,:] = x

          prob_func = (prob,i,repeat) -> remake(prob, u0 = u0s_func(@view(ext_state_val[1:n_states,i])),
                                                       p = @view(ext_state_val[n_states+1:end,i]))
          output_func = function (sol,i)
            w = prod(pdf(a,b) for (a,b) in zip(dists,x[:,i]))
            k = g(sol)
            return k*w,false
          end

          ensembleprob = EnsembleProblem(prob,prob_func=prob_func, output_func = output_func)
          sol = solve(ensembleprob,args...;trajectories=trajectories,kwargs...)
          dx .= vec(sol.u)
      end
    end

    #solve
    intprob = QuadratureProblem(integrand,minimum.(d for d ∈ dists),maximum.(d for d ∈ dists),batch=batch)
    sol = solve(intprob,quadalg,reltol=ireltol,
                abstol=iabstol,maxiters = maxiters)
end
