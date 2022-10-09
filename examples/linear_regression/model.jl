using LinearAlgebra, Distributions, Random, Plots, CSV, DataFrames, StatsBase, Flux, Tullio
using Zygote:Buffer, ignore
using SparseHamiltonianFlows

n_run = 5

##############################
# load data
##############################
d = 12 # intercept, 10 features, plus error
xs = CSV.read(joinpath("data", "linreg.csv"), DataFrame)
N = size(Matrix(xs),1)
xs = hcat(ones(N), Matrix(xs))
Random.seed!(2022)

##############################
# sparse flow parameters
##############################
sub_xs = nothing
inds = nothing
# hyper params
M = 30
elbo_size = 1
number_of_refresh = 8
K = 10
lf_n = number_of_refresh * K
iter = 50000
cond = false
n_subsample_elbo = 100
save = true
S_init = 100
S_final = 10000
# optimizer
optimizer = Flux.ADAM(0.002)
# sampling and likelihood functions
c_prior = 0.01 # variance (diagonal entries of cov matrix)
m = 15.
logq0 = z -> -0.5/c_prior * (vec(z .- m)' * vec(z .- m)) - 0.5*d* log(2*pi) -0.5*d*log(c_prior)
log_prior = z -> -0.5 * (z' * z) - 0.5*d* log(2*pi)
function linear_ind(x,z)
    features = zeros(d-1)
    response = -20.
    ignore() do
        features = x[1:d-1]
        response = x[d]
    end
    return -0.5 * z[d] - (1. / (2. * exp(z[d]))) * ((z[1:d-1]' * features) - response)^2. - 0.5 * log(2*pi)
end
logp_ind = linear_ind
function sample_q0(n)
    if n == 1
        return sqrt(c_prior) * randn(d) .+ m
    else
        return sqrt(c_prior) * randn(n, d) .+ m
    end
end
function logp_lik(z, xs)
    N = size(xs,1)
    rs = @view(xs[:, d])
    fs = @view(xs[:,1:d-1])
    diffs = rs .- fs * z[1:d-1]
    return -0.5 * exp(-z[d]) * sum(abs2, diffs) - 0.5 * N * log(2π) - 0.5 * N * z[d]
end

function ∇potential_by_hand(xs, z, w)
    N = size(xs,1)
    grads_p = Buffer(zeros(d))
    rs = @view(xs[:, d])
    β = @view(z[1:d-1])
    fs = @view(xs[:,1:d-1])
    diffs = rs .- fs * β
    @tullio s[j] :=  diffs[i]*w[i]*fs[i,j]
    grads_p[1:d-1] = -β .+ s ./ exp(z[d])
    @tullio t := diffs[i]^2*w[i]
    grads_p[d] = -z[d] + 0.5 * exp(-z[d]) * t - 0.5 * sum(w)
    grads_p_unbuf = copy(grads_p)
    return grads_p_unbuf
end

# initialization
ϵ_unc = vcat(log.(0.02 .* ones(d-1)), log(0.0002))
# bundle arguments
a = SparseHamiltonianFlows.Args(d, N, xs, sub_xs, inds, M, elbo_size, number_of_refresh, 
                        K, lf_n, iter, cond, n_subsample_elbo, save, 
                        S_init, S_final, optimizer, 
                        log_prior, logq0, logp_lik, sample_q0, ∇potential_by_hand, [], [], [])

##############################
# his parameters
##############################
sample_size_for_metric_computation = 100
n_mcmc = number_of_refresh # number of tempering used for his/uha
save_freq = 500 # freq of saving ps
# init leapfrog stepsizes and tempering schedule
ϵ0his = exp.(ϵ_unc)
T0_his = ones(n_mcmc)
sample_q0() = sample_q0(1)

function logp_elbo(z, n_subsample_elbo)
    dx = size(xs,2)
    sub_xs = zeros(n_subsample_elbo, dx)
    ignore() do
        inds = sort(sample(1:N, n_subsample_elbo, replace = false))
        sub_xs = xs[inds, :]
    end
    return log_prior(z) + logp_lik(z, sub_xs) * N/n_subsample_elbo
end
logq = (z) -> logq0(z)
∇logq = z -> (m .- z) ./ c_prior

function ∇logp_mini(z, inds)
    dx = size(xs,2)
    M = size(inds, 1)
    sub_xs = zeros(M, dx)
    w = ones(M) * N/M
    ignore() do
        sub_xs = xs[inds, :]
    end
    return ∇potential_by_hand(sub_xs, z, w)
end

##############################
# uha
##############################
# init leapfrog stepsizes and tempering schedule
ϵ0uha = exp.(ϵ_unc)
T0_uha = ones(n_mcmc - 1)
# init damping coef (used in partial refreshement)
η0 = [0.5]
∇logp = z -> ∇potential_by_hand(xs, z, ones(N))
ljd = z -> log_joint_density(xs, z, log_prior, logp_ind)
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
    πs = log_prior(z)
    for i in 1:N
        πs = πs + sub_ws[i] * logp_ind(sub_xs[i,:], z)
    end
    return πs
end

accept_ratio = 0.65 # for HMC and NUTS
nuts_acc_ratio = 0.65

##############################
# coresets
##############################
num_random_proj = 100 * d
Ms = [d, 5*d, 10*d]

##############################
# old functions
##############################

function log_joint_density(xs, z, logq0, logp_ind)
    return logq0(z) + logp_lik(z, xs)
end

# true mean and covariance
posts = CSV.read(joinpath("data", "linreg_100k.csv"), DataFrame)
posts = Matrix(posts)
posts = hcat(hcat(posts[:,end-1], posts[:,8:end-2]), posts[:,end])
post_mean = vec(mean(posts, dims=1))
post_var = cov(posts)
post_cov = post_var
post_precision = Matrix((0.5 * (post_var + transpose(post_var)))\I(d))
