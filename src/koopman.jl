function koopman(g,prob,u0,p,args...;kwargs...)
    g(solve(remake(prob,u0=u0,p=p),args...;kwargs...))
end

function koopman_cost(u0s,ps,g,prob,args...;maxevals=0,
                      ireltol = 1e-2, iabstol=1e-2,kwargs...)
  n = length(u0s)
  function _f(x)
    u0 = x[1:n]
    p  = x[n+1:end]
    k = koopman(g,prob,u0,p,args...;kwargs...)
    w = prod(pdf(a,b) for (a,b) in zip(u0s,u0))*
        prod(pdf(a,b) for (a,b) in zip(ps,p))
    k*w
  end
  xs = [u0s;ps]
  hcubature(_f, minimum.(xs), maximum.(xs);
            reltol=ireltol, abstol=iabstol, maxevals = maxevals)
end

function montecarlo_cost(u0s,ps,g,prob,args...;num_monte,kwargs...)
  prob_func = function (prob,i,repeat)
    remake(prob,u0=rand.(u0s),p=rand.(ps))
  end
  output_func = (sol,i) -> (g(sol),false)
  monte_prob = MonteCarloProblem(prob;
                                 output_func = output_func,
                                 prob_func = prob_func)
  mean(solve(monte_prob,args...;num_monte=num_monte,kwargs...).u)
end
