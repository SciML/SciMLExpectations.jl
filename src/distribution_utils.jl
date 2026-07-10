"""
    GenericDistribution(pdf_func, rand_func, lb, ub)
    GenericDistribution(d::Distributions.Sampleable, ds...)

Represent a distribution by its density, sampler, and integration bounds.

`GenericDistribution` can be used for arbitrary joint densities and for products of
sampleable distributions. The `Distributions.Sampleable` constructor builds a joint
density from independent one-dimensional distributions without relying on
`Distributions.Product`.

## Arguments

  - `pdf_func`: Function called as `pdf_func(x)` to evaluate the density.
  - `rand_func`: Zero-argument function that returns one sample.
  - `lb`: Lower integration bound.
  - `ub`: Upper integration bound.
  - `d, ds...`: Independent sampleable distributions used to build a joint
    distribution.

## Fields

  - `pdf_func`: Stored density function.
  - `rand_func`: Stored sampler.
  - `lb`: Stored lower bound.
  - `ub`: Stored upper bound.

## Returns

A distribution-like object supporting `pdf`, `rand`, `minimum`, `maximum`, and
`extrema`.
"""
struct GenericDistribution{TF, TRF, TLB, TUB}
    pdf_func::TF
    rand_func::TRF
    lb::TLB
    ub::TUB
end

function GenericDistribution(d::Distributions.Sampleable, ds...)
    dists = (d, ds...)
    pdf_func(x) = exp(sum(logpdf(f, y) for (f, y) in zip(dists, x)))
    rand_func() = [rand(d) for d in dists]
    lb = SVector(map(minimum, dists)...)
    ub = SVector(map(maximum, dists)...)

    return GenericDistribution(pdf_func, rand_func, lb, ub)
end

Distributions.pdf(d::GenericDistribution, x) = d.pdf_func(x)
Base.minimum(d::GenericDistribution) = d.lb
Base.maximum(d::GenericDistribution) = d.ub
Base.extrema(d::GenericDistribution) = minimum(d), maximum(d)
Base.rand(d::GenericDistribution) = d.rand_func()
