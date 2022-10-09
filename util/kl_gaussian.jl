using LinearAlgebra, Statistics, Optim, ForwardDiff

function kl_gaussian(μq, Σq, μp, Σp)
    d = length(μq)
    if det(Σq) < 0
        return 0.5 * (logdet(Σp) - log(1e-20) - d + tr(Σp \ Σq) + transpose(μp - μq) * (Σp \ (μp - μq)))
    else    
        return 0.5 * (logdet(Σp) - logdet(Σq) - d + tr(Σp \ Σq) + transpose(μp - μq) * (Σp \ (μp - μq)))
    end
end

function kl_gaussian_nuts(D, μq, Σq)
    μp = vec(mean(D,dims=1))
    Σp = cov(D,D)
    return kl_gaussian(μq, Σq, μp, Σp)
end

function kl_gaussian_precision(μq, Σq, μp, Σp_inv)
    d = length(μq)
    return 0.5 * (-1.0 * logdet(Σp_inv) - logdet(Σq) - d + tr(Σp_inv * Σq) + transpose(μp - μq) * (Σp_inv * (μp - μq)))
end

function kl_gaussian_est(D::Matrix{Float64}, μp::Vector{Float64}, Σp_inv::Matrix{Float64})
    μq = vec(mean(D, dims =1))
    Σq = cov(D, D)

    return kl_gaussian_precision(μq, Σq, μp, Σp_inv)
end

function kl_gaussian_est_cov(D::Matrix{Float64}, μp::Vector{Float64}, Σp::Matrix{Float64})
    μq = vec(mean(D, dims =1))
    Σq = cov(D, D)

    return kl_gaussian(μq, Σq, μp, Σp)
end

function kl_gaussian_est(D::Matrix{Float64}, logp::Function)
    μq = vec(mean(D, dims =1))
    Σq = cov(D, D)
    # estimate using Laplace approxiimation
    μp, Σp_inv = laplace(logp, μq)

    return kl_gaussian_precision(μq, Σq, μp, Σp_inv)
end


function laplace(logp, x0)

    obj = z -> -logp(z)
    opt = optimize(obj, x0)
    μ = Optim.minimizer(opt)

    hess = ForwardDiff.hessian(obj, μ)
    return μ, hess
end
