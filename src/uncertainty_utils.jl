
# wraps Distribtuions.jl 
_rand(f::T) where T <: Distribution = rand(f)
_rand(x) = x
_pdf(f::T, x) where T <: Distribution = pdf(f,x)
_pdf(f, x) = one(eltype(x))

# creates a tuple of idices, or ranges, from array partition lengths
function accumulated_range(partition_lengths)
    c = [0, cumsum(partition_lengths)...]
    return Tuple(c[i]+1==c[i+1] ? c[i+1] : (c[i]+1):c[i+1] for i in 1:length(c)-1)
end
