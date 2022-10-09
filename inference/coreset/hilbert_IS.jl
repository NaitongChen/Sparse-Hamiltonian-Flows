using LinearAlgebra, Random, Statistics, StatsBase
include("random_projection.jl")

function hilbert_IS_coreset(xs, M, J, logp_ind, logp, d)
    N = size(xs,1)
    proj = random_projection(xs, J, logp_ind, logp, d) # J by N

    sigmas = zeros(N)
    for i in 1:N
        sigmas[i] = norm(proj[:, i])
    end
    sigma = sum(sigmas)

    ws = sigmas ./ sigma
    inds = sort(sample(1:N, Weights(ws), M, replace = false))

    is_weights = zeros(N)
    is_weights[inds] = (sigma ./ M) ./ sigmas[inds]

    return is_weights
end
