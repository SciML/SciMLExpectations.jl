struct ProbIntsCache{T}
  σ::T
  order::Int
end
function (p::ProbIntsCache)(integrator)
  integrator.u .= integrator.u .+ p.σ*sqrt(integrator.dt^(2*p.order))*randn(size(integrator.u))
end

"""
Conrad P., Girolami M., Särkkä S., Stuart A., Zygalakis. K, Probability
Measures for Numerical Solutions of Differential Equations, arXiv:1506.04592
"""
function ProbIntsUncertainty(σ,order,save=true)
  affect! = ProbIntsCache(σ,order)
  condtion = (t,u,integrator) -> true
  save_positions = (save,false)
  DiscreteCallback(condtion,affect!,save_positions=save_positions)
end

struct AdaptiveProbIntsCache
  order::Int
end
function (p::AdaptiveProbIntsCache)(integrator)
  integrator.u .= integrator.u .+ integrator.EEst*sqrt(integrator.dt^(2*p.order))*randn(size(integrator.u))
end

function AdaptiveProbIntsUncertainty(order,save=true)
  affect! = AdaptiveProbIntsCache(order)
  condtion = (t,u,integrator) -> true
  save_positions = (save,false)
  DiscreteCallback(condtion,affect!,save_positions=save_positions)
end
