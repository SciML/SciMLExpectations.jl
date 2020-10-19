
# wraps Distribtuions.jl
# note: MultivariateDistributions do not support minimum/maximum. Setting to -Inf/Inf with
# the knowledge that it will cause quadrature integration to fail if Koopman is used. To use
# MultivariateDistributions the upper/lower bounds should be set with the kwargs.
_minimum(f::T) where T <: MultivariateDistribution = -Inf .* ones(eltype(f), size(f)...)
_minimum(f) = minimum(f)
_maximum(f::T) where T <: MultivariateDistribution = Inf .* ones(eltype(f), size(f)...)
_maximum(f) = maximum(f)
_rand(f::T) where T <: Distribution = rand(f)
_rand(x) = x
_pdf(f::T, x) where T <: Distribution = pdf(f,x)
_pdf(f, x) = one(eltype(x))

# creates a tuple of idices, or ranges, from array partition lengths
function accumulated_range(partition_lengths)
    c = [0, cumsum(partition_lengths)...]
    return Tuple(c[i]+1==c[i+1] ? c[i+1] : (c[i]+1):c[i+1] for i in 1:length(c)-1)
end
