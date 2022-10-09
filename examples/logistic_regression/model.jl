using LinearAlgebra, Distributions, Random, Plots, CSV, DataFrames, StatsBase, Flux, Tullio
using Zygote:Buffer, ignore
using Zygote.LogExpFunctions: logistic
using SparseHamiltonianFlows

n_run = 5

##############################
# load data
##############################
d = 11 # intercept plus 10 features
xs = CSV.read(joinpath("data", "logreg.csv"), DataFrame)
xs = Matrix(xs) # convert dataframe to matrix
N = size(xs,1)
Random.seed!(2022)

##############################
# sparse flow parameters
##############################
function stratified_sampling(M)
    ind_seq = [1:N ;]
    positives = ind_seq[xs[:,11] .== 1.]
    negatives = ind_seq[xs[:,11] .== 0.]

    count_positives = size(positives, 1)

    # take 50% positive, 50% negative (if possible)
    n_pos = min(Int(ceil(M / 2.)), count_positives)
    n_neg = M - n_pos

    inds_pos = sort(sample(positives, n_pos, replace = false))
    inds_neg = sort(sample(negatives, n_neg, replace = false))

    inds = sort(vcat(inds_pos, inds_neg))

    return inds
end
M = 30
inds = stratified_sampling(M)
sub_xs = xs[inds,:]
# hyper params
elbo_size = 1
number_of_refresh = 8
K = 10
lf_n = number_of_refresh * K
iter = 100000
cond = false
n_subsample_elbo = 100
save = true
S_init = 100
S_final = 10000
# optimizer
optimizer = Flux.ADAM(0.001)
# sampling and likelihood functions
function cauchy_sum(z)
    ret = 0.
    for i in 1:size(z,1)
        feat = 0.
        ignore() do
            feat = z[i]
        end
        ret = ret - log(pi) - log1p(feat^2)
    end
    return ret
end
log_prior = cauchy_sum
function grad_log_cauchy(z)
    return -2.0*z/(1.0+z^2.0)
end
function log_sigmoid(x)
    if x < -300
        return x
    else
        return -log1p(exp(-x))
    end
end
function neg_sigmoid(x)
    return -1.0/(1.0 + exp(-x))
end

# individual likelihood
function logistic_ind(x, z)
    features = zeros(d)
    label = -20.
    ignore() do
        features = vcat(1., x[1:d-1])
        label = x[d]
    end
    return label * log_sigmoid(z' * features) + (1. - label) * log_sigmoid(-z' * features)
end
logp_ind = logistic_ind
function logp_lik(z, xs)
    X = @view(xs[:,1:end-1])
    Y = @view(xs[:, end])
    Z = X*z[2:end] .+ z[1]
    @tullio llh := (Y[n] -1.0) *Z[n]
    return llh + sum(log_sigmoid.(Z))
end
c_prior = 0.0001 # variance (diagonal entries of cov matrix)
m = 15.
logq0 = z -> -0.5/c_prior * (vec(z .- m)' * vec(z .- m)) - 0.5*d* log(2*pi) -0.5*d*log(c_prior)
function sample_q0(n)
    if n == 1
        return sqrt(c_prior) * randn(d) .+ m
    else
        return sqrt(c_prior) * randn(n, d) .+ m
    end
end

function ∇potential_by_hand(xs, z, w)
    grad_prior = -2. .* z ./ (1. .+ z.^2)
    N = size(xs, 1)
    X = hcat(ones(N), @view(xs[:,1:end-1]))
    Y = @view(xs[:, end])
    Pfit = neg_sigmoid.(X*z) 
    @tullio G[j] := X[n,j]*(Pfit[n] + Y[n])*w[n] 
    return G .+ grad_prior
end

# initialization
ϵ_unc = copy(log.(0.0005 .* ones(d)))
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
save_freq = 1000 # freq of saving ps
# init leapfrog stepsizes and tempering schedule
ϵ0his = copy(exp.(ϵ_unc))
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
    # llh--unif subsample
    X = @view(sub_xs[:,1:end-1])
    Y = @view(sub_xs[:, end])
    β = @view(z[2:end])
    Z = X*β .+ z[1]
    @tullio llh := (Y[n] -1.0) *Z[n]
    return πs + (llh + sum(log_sigmoid.(Z)) )*N/n_subsample_elbo
end

logq = (z) -> logq0(z)
∇logq = z -> (m .- z) ./ c_prior

# subsample of size M
function ∇logp_mini(z, inds)
    dx = size(xs,2)
    M = size(inds, 1)
    sub_xs = zeros(M, dx)
    w = ones(M) * N/M
    ignore() do
        sub_xs = xs[inds, :]
    end
    return ∇potential_by_hand(sub_xs,z,w) 
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
num_random_proj = 50 * d
Ms = [d, 5*d, 10*d]

##############################
# old functions
##############################

# z is the location variable
function log_joint_density(xs, z, logq0, logp_ind)
    πs = logq0(z)
    X = @view(xs[:,1:end-1])
    Y = @view(xs[:, end])
    β = @view(z[2:end])
    Z = X*β .+ z[1]
    @tullio llh := (Y[n] -1.0) *Z[n] + log_sigmoid(Z[n])
    return πs + llh
end

# true mean and covariance
posts = CSV.read(joinpath("data", "logreg_100k.csv"), DataFrame)
posts = Matrix(posts)
posts = hcat(posts[:,end], posts[:,8:end-1])
post_mean = vec(mean(posts, dims=1))
post_var = cov(posts)
post_cov = post_var
post_precision = Matrix((0.5 * (post_var + transpose(post_var)))\I(d))


