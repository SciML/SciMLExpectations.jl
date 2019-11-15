function make_S(prob,args...;kwargs...)
    function s_(ext_state)
        u0s = ext_state[1:length(prob.u0)]
        ps = ext_state[(length(u0) + 1):end]
        solve(remake(prob, u0 = u0s, p = ps), args...; kwargs...)
    end
end

function koopman(g,prob,u0,p,args...;kwargs...)
    g(solve(remake(prob,u0=u0,p=p),args...;kwargs...))
end

function expectation(g,u0s, ps,prob,args...; maxiters=0, batch = 0, quadalg = HCubatureJL(),ireltol = 1e-2, iabstol = 1e-2, kwargs...)
      a=3
    u0s_distIDX = isa.(u0s,Sampleable)
    ps_distIDX = isa.(ps,Sampleable)

    u0s_dist_pairs = [Pair(idx, u0s[idx]) for idx ∈ (1:length(u0s))[u0s_distIDX]]
    ps_dist_pairs = [Pair(idx, ps[idx]) for idx ∈ (1:length(ps))[ps_distIDX]]

    u0s_num = zeros(length(u0s))
    ps_num = zeros(length(ps))
# expectation(g,prob,u0s_num, ps_num, u0s_dist_pairs, ps_dist_pairs, args...; kwargs...)
    @. u0s_num[!u0s_distIDX] = u0s[!u0s_distIDX]
    @. ps_num[!ps_distIDX] = ps[!ps_distIDX]

    __expectation(g,u0s_num, ps_num, prob, u0s_dist_pairs, ps_dist_pairs, args...; kwargs...)
end

function __expectation(g,u0s,ps,prob,
                    u0s_dist_pairs,ps_dist_pairs, args...;
                    maxiters=0,
                    batch = 0,
                    quadalg = HCubatureJL(),
                    ireltol = 1e-2, iabstol = 1e-2, kwargs...)

    n_rand_u0s = length(u0s_dist_pairs)
    n_rand_ps  = length(ps_dist_pairs)

    @assert n_rand_ps <= length(ps)
    @assert n_rand_u0s <= length(u0s)

    stateIdx = getfield.(u0s_dist_pairs, :first)
    stateDist = getfield.(u0s_dist_pairs, :second)

    paramIdx = getfield.(ps_dist_pairs, :first)
    paramDist = getfield.(ps_dist_pairs, :second)

    indxs = vcat(stateIdx, paramIdx .+ length(u0s))
    dists = vcat(stateDist, paramDist)

    minBounds = minimum.(dists)
    maxBounds = maximum.(dists)

    ext_state = vcat(u0s, ps)  ## Type issues. Needs to not include distirubtion types

    S = make_S(prob, args...; kwargs...)

    function _f(x)
        ext_state[indxs] = x
        prob = prod(pdf(a, b) for (a, b) in zip(dists, x))
        return g(S(ext_state)) * prob
    end

    intprob = QuadratureProblem(_f,minBounds,maxBounds,batch=batch)
    sol = solve(intprob,quadalg,reltol=ireltol,
                abstol=iabstol,maxiters = maxiters)
end

function koopman_expectation(g,u0s,ps,prob,args...;maxiters=0,
                      batch = 0,
                      quadalg = HCubatureJL(),
                      ireltol = 1e-2, iabstol=1e-2,kwargs...)
  n = length(u0s)
  if batch == 0
    _f = function (x,p)
      u0 = x[1:n]
      p  = x[n+1:end]
      k = koopman(g,prob,u0,p,args...;kwargs...)
      w = prod(pdf(a,b) for (a,b) in zip(u0s,u0))*
          prod(pdf(a,b) for (a,b) in zip(ps,p))
      k*w
    end
  else
    _f = function (dx,x,p)
      trajectories = size(x,2)
      prob_func = (prob,i,repeat) -> remake(prob,u0=@view(x[1:n,i]),
                                                 p=@view(x[n+1:end,i]))
      output_func = function (sol,i)
        k = g(sol)
        u0= @view(x[1:n,i])
        p = @view(x[n+1:end,i])
        w = prod(pdf(a,b) for (a,b) in zip(u0s,u0))*
            prod(pdf(a,b) for (a,b) in zip(ps,p))
        k*w,false
      end

      ensembleprob = EnsembleProblem(prob,prob_func=prob_func,
                                     output_func = output_func)
      sol = solve(ensembleprob,args...;trajectories=trajectories,kwargs...)
      dx .= vec(sol.u)
      nothing
    end
  end
  xs = [u0s;ps]
  intprob = QuadratureProblem(_f,minimum.(xs),maximum.(xs),batch=batch)
  sol = solve(intprob,quadalg,reltol=ireltol,
              abstol=iabstol,maxiters = maxiters)
end

function montecarlo_expectation(g,u0s,ps,prob,args...;trajectories,kwargs...)
  prob_func = function (prob,i,repeat)
    remake(prob,u0=rand.(u0s),p=rand.(ps))
  end
  output_func = (sol,i) -> (g(sol),false)
  monte_prob = EnsembleProblem(prob;
                                 output_func = output_func,
                                 prob_func = prob_func)
  mean(solve(monte_prob,args...;trajectories=trajectories,kwargs...).u)
end
