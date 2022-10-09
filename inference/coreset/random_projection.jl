using LinearAlgebra, Random, Statistics, NLsolve, Distributions, ProgressMeter
include("posterior_laplace_approx.jl")

function random_projection(xs, J, logp_ind, logp, d)
    N = size(xs,1)
    @info "laplace approximation of posterior"
    zs = sample_with_laplace_approx(logp, d, J) # J by d
    proj = zeros(J, N)

    @info "random projection"
    prog_bar = ProgressMeter.Progress(J, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for j in 1:J
        for n in 1:N
            proj[j,n] = logp_ind(xs[n,:], zs[j,:])
        end
        ProgressMeter.next!(prog_bar)
    end

    proj .-=  mean(proj, dims=1)
    proj .*= sqrt(1. / J)

    return proj
end
