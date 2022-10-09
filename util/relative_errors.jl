using LinearAlgebra, Random, Distributions

function rel_err_mean(D_full_HMC, D)
    μ = vec(mean(D_full_HMC, dims=1))
    μhat = vec(mean(D, dims=1))
    return norm(μ - μhat) / norm(μ)
end

function rel_err_cov(D_full_HMC, D)
    Σ = cov(D_full_HMC, D_full_HMC)
    Σhat = cov(D, D)
    return norm(Σ - Σhat) / norm(Σ)
end


function rel_err_mean_no_hmc(D, μ)
    μhat = vec(mean(D, dims=1))
    return norm(vec(μ) - μhat) / norm(μ)
end

function rel_err_mean_no_dat(μhat, μ)
    return norm(vec(μ) - μhat) / norm(μ)
end

function rel_err_cov_no_hmc(D, Σ)
    Σhat = cov(D, D)
    return norm(Σ - Σhat) / norm(Σ)
end

function rel_err_cov_no_dat(Σhat, Σ)
    return norm(Σ - Σhat) / norm(Σ)
end

function rel_err(D_full_HMC, D)
    e_mean = rel_err_mean(D_full_HMC, D)
    e_cov = rel_err_cov(D_full_HMC, D)
    return e_mean, e_cov
end


function rel_err_log_cov(D_full_HMC, D)
    Σ = cov(D_full_HMC, D_full_HMC)
    Σhat = cov(D, D)

    log_Diag = log.(diag(Σ))
    log_Diag_hat = log.(diag(Σhat))

    return mean(abs.((log_Diag_hat - log_Diag) ./ log_Diag))
end

function rel_err_log_cov_no_hmc(D, Σ)
    Σhat = cov(D, D)
    log_Diag = log.(diag(Σ))
    log_Diag_hat = log.(diag(Σhat))

    return mean(abs.((log_Diag_hat - log_Diag) ./ log_Diag))
end

function rel_err_log_cov_no_dat(Σhat, Σ)
    log_Diag = log.(diag(Σ))
    log_Diag_hat = log.(diag(Σhat))

    return mean(abs.((log_Diag_hat - log_Diag) ./ log_Diag))
end

function rel_err_no_hmc(post_mean, post_cov, D)
    e_mean = rel_err_mean_no_hmc(D, post_mean)
    e_cov = rel_err_cov_no_hmc(D, post_cov)
    e_log = rel_err_log_cov_no_hmc(D, post_cov)
    return e_mean, e_cov, e_log
end

function rel_err_no_dat(post_mean, post_cov, est_mean, est_cov)
    e_mean = rel_err_mean_no_dat(est_mean, post_mean)
    e_cov = rel_err_cov_no_dat(est_cov, post_cov)
    e_log = rel_err_log_cov_no_dat(est_cov, post_cov)
    return e_mean, e_cov, e_log
end

function energy_dist(post_mean, post_cov, D_z)
    n = size(D_z, 1)
    d = MvNormal(post_mean, post_cov)
    D_z_star = Matrix(rand(d, n)')
    
    n_half = Int(floor(n/2))
    
    D = D_z[1:n_half,:]
    D_prime = D_z[n_half+1:end,:]

    D_true = D_z_star[1:n_half,:]
    D_true_prime = D_z_star[n_half+1:end,:]

    return 2*mean(norm.(eachrow(D - D_true))) - mean(norm.(eachrow(D - D_prime))) - mean(norm.(eachrow(D_true - D_true_prime)))
end