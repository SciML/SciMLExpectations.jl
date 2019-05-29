function koopman(g,prob,u0,args...;kwargs...)
    g(solve(remake(prob,u0=u0),args...;kwargs...))
end

function koopman_cost(u0s,g,prob,args...;maxevals=0,
                      ireltol = 1e-3, iabstol=1e-3,kwargs...)
  function _f(u0)
    k = koopman(g,prob,u0,args...;kwargs...)
    p = prod(pdf(x,y) for (x,y) in zip(u0s,u0))
    k*p
  end

  hcubature(_f, minimum.(u0s), maximum.(u0s);
            reltol=ireltol, abstol=iabstol, maxevals = maxevals)
end

function montecarlo_cost(u0s,g,prob,args...;num_monte,kwargs...)
  prob_func = function (prob,i,repeat)
    remake(prob,u0=rand.(u0s))
  end
  output_func = (sol,i) -> (g(sol),false)
  monte_prob = MonteCarloProblem(prob;
                                 output_func = output_func,
                                 prob_func = prob_func)
  mean(solve(monte_prob,args...;num_monte=num_monte,kwargs...).u)
end
