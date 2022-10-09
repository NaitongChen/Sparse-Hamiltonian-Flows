using LinearAlgebra, Distributions, Random, Plots, StatsBase, Flux
using Zygote: ignore
using SparseHamiltonianFlows

# ```
# μ ∼ N(μ0, Σ0)
# x|μ ∼ N(μ, Σ), Σ is diagonal
# ```

n_run = 5

##############################
# data generation
##############################
d = 10
c_prior = 1.
c_lik = 100.
m = 0.0
μ = 10.0
Σ = c_lik * Matrix(I(d))
μ0 = 0.0 * ones(d) .+ m
Σ0 = c_prior * Matrix(I(d))
Random.seed!(2022)
d = 10
N = 10000
xs = Matrix(rand(MvNormal(μ .* ones(d), Σ), N)')

##############################
# sparse flow parameters
##############################
sub_xs = nothing
inds = nothing
# hyper params
M = 30
elbo_size = 1
number_of_refresh = 5
K = 10
lf_n = number_of_refresh * K
iter = 20000
cond = false
n_subsample_elbo = 100
save = true
S_init = 100
S_final = 10000
# optimizer
optimizer = Flux.ADAM(0.001)
# sampling and likelihood functions
logq0 = z -> -0.5*d*log(c_prior) -0.5/c_prior*(vec(z .- m)' *  vec(z .- m)) - (d / 2.) * log(2*pi)
log_prior = logq0
logp_ind_ind = (x, z) -> -0.5*d*log(c_lik) -0.5/c_lik * (vec(z .- x)' * vec(z .- x)) - (d/2.) * log(2*pi)
function logp_ind(z, xs)
    ttt = 0
    for i in 1:size(xs,1)
        ttt += logp_ind_ind(xs[i,:], z)
    end
    return ttt
end
function sample_q0(n)
    if n == 1
        return sqrt(c_prior)*randn(d) .+ m
    else
        return sqrt(c_prior)*randn(n, d) .+ m
    end
end
function ∇potential_by_hand(sub_xs, z, w)
    grads_p = (sub_xs .- z') ./ c_lik
    grad_prior = (μ0 .- z) ./ c_prior
    return grad_prior .+ grads_p' * w
end
# initialization
ϵ0 = 0.01
ϵ_unc = log.(ϵ0 .* ones(d))
# bundle arguments
a = Args(d, N, xs, sub_xs, inds, M, elbo_size, number_of_refresh,
                        K, lf_n, iter, cond, n_subsample_elbo, save,
                        S_init, S_final, optimizer,
                        log_prior, logq0, logp_ind, sample_q0, ∇potential_by_hand, [], [], [])

##############################
# his parameters
##############################
sample_size_for_metric_computation = 100
n_mcmc = number_of_refresh # number of tempering used for his/uha
save_freq = 200 # freq of saving ps
# init leapfrog stepsizes and tempering schedule
ϵ0his = ϵ0 * ones(d)
T0_his = ones(n_mcmc)
sample_q0() = sample_q0(1)
function logp_elbo(z, n_subsample_elbo)
    dx = size(xs,2)
    sub_xs = zeros(n_subsample_elbo, dx)
    ignore() do
        inds = sort(sample(1:N, n_subsample_elbo, replace = false))
        sub_xs = xs[inds, :]
    end
    πs = log_prior(z)
    for i in 1:n_subsample_elbo
        sub_xs_i = ones(dx)
        ignore() do
            sub_xs_i = sub_xs[i,:]
        end
        πs = πs + logp_ind_ind(sub_xs_i, z)*N/n_subsample_elbo
    end
    return πs
end
logq = (z) -> logq0(z)
∇logq = z -> (μ0 .- z)./c_prior
# subsample of size M
function ∇logp_mini(z, inds)
    d = size(xs,2)
    M = size(inds, 1)
    sub_xs = zeros(M, d)
    w = ones(M) * N/M
    ignore() do
        sub_xs = xs[inds, :]
    end
    grads_p = (sub_xs .- z') ./ c_lik
    grad_prior = (μ0 .- z) ./ c_prior
    return grad_prior .+ grads_p' * w
end

##############################
# uha
##############################
# init leapfrog stepsizes and tempering schedule
ϵ0uha = ϵ0 * ones(d)
T0_uha = ones(n_mcmc - 1)
# init damping coef (used in partial refreshement)
η0 = [0.5]
∇logp = z -> ∇potential_by_hand(xs, z, ones(N))
ljd = z -> logp_elbo(z, N)
∇ljd = zz -> ForwardDiff.gradient(ljd, zz)
# bridging distribution
logγ(z, β) = β * ljd(z) + (1.0 - β) * logq(z)
∇logγ(z, β) = β * ∇logp(z) .+ (1.0 - β) * ∇logq(z)

##############################
# HMC and NUTS
##############################
function coreset_posterior(z, ws)
    selected = ws .> 0
    sub_xs = xs[selected,:]
    sub_ws = ws[selected]
    N = size(sub_xs,1)
    πs = logq0(z)
    for i in 1:N
        πs = πs + sub_ws[i] * logp_ind_ind(sub_xs[i,:], z)
    end
    return πs
end

accept_ratio = 0.65 # for HMC and NUTS
nuts_acc_ratio = 0.65

##############################
# coresets
##############################
num_random_proj = 50 * d
Ms = [Int(0.5*d), d, 5*d, 10*d]

##############################
# old functions
##############################

# z is the location variable
function log_joint_density(xs, z, logq0, logp_ind)
    N = size(xs,1)
    πs = logq0(z)
    for i in 1:N
        πs = πs + logp_ind(xs[i,:], z)
    end
    return πs
end

# true mean and covariance
x̄ = vec(mean(xs, dims=1))
post_mean = vec(inv(inv(Σ0) + N * inv(Σ)) * (inv(Σ0) * μ0 + N * inv(Σ) * x̄))
post_var = inv(inv(Σ0) + N * inv(Σ))
post_cov = post_var
post_precision = Matrix((0.5 * (post_var + transpose(post_var)))\I(d))
println("posterior mean = ", post_mean)
println("posterior covariance = ", post_var)
