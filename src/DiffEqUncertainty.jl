__precompile__()

module DiffEqUncertainty

using DiffEqBase

immutable ProbIntsCache{T}
  σ::T
  order::Int
end
function (p::ProbIntsCache)(integrator)
  integrator.u .= integrator.u .+ p.σ*sqrt(integrator.dt^(2*p.order))*randn(size(integrator.u))
end

function ProbIntsUncertainty(σ,order,save=true)
  affect! = ProbIntsCache(σ,order)
  condtion = (t,u,integrator) -> true
  save_positions = (save,false)
  DiscreteCallback(condtion,affect!,save_positions)
end

export ProbIntsUncertainty

end
