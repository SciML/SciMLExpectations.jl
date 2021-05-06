## GenericDistribution wrapper

struct GenericDistribution{TF, TRF, TLB, TUB}
    pdf_func::TF
    rand_func::TRF
    lb::TLB
    ub::TUB
end

# included b/c Distribution.Product method of mixed distirbutions are type instable
function GenericDistribution(d, ds...)
    dists = (d, ds...)
    pdf_func(x) = exp(sum(logpdf(f,y) for (f,y) in zip(dists, x)))
    rand_func() =  [rand(d) for d in dists] 
    lb = map(minimum, dists)
    ub = map(maximum, dists)
    GenericDistribution(pdf_func, rand_func, lb, ub)
end

Distributions.pdf(d::GenericDistribution, x) = d.pdf_func(x)
Base.minimum(d::GenericDistribution) = d.lb
Base.maximum(d::GenericDistribution) = d.ub
Base.extrema(d::GenericDistribution) = minimum(d), maximum(d)
Base.rand(d::GenericDistribution) = d.rand_func()