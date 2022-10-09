using LinearAlgebra, Random, Statistics

function uniform_coreset(xs, M)
    N = size(xs,1)
    inds = sort(sample(1:N, M, replace = false))
    ws = zeros(N)
    ws[inds] = (N/M) .* ones(M)
    return ws
end
