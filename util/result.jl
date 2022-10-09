include("../inference/mcvae/util.jl")
include("ksd.jl")
include("kl_gaussian.jl")

#############################################
# Functions for result processing (for mcvae)
#############################################
function argparse_his(ps, sample_q0, logp_elbo, n_subsample, logq, ∇logp_mini, K, d, lf_n, mini_flow, mini_flow_size, data_size)
    ϵ = @. expm1(ps[1]) + 1.0
    T = logistic.(ps[2])
    return (sample_q0, logp_elbo, n_subsample,logq, ∇logp_mini, T, ϵ, K, d, lf_n, mini_flow, mini_flow_size, data_size)
end

function argparse_uha(ps, sample_q0, logp_elbo, n_subsample,logq, ∇logq, ∇logp_mini, K, d, lf_n, mini_flow, mini_flow_size, data_size )
    ϵ = @. expm1(ps[1]) + 1.0
    η = logistic.(ps[2])
    T = T_all(ps[3])
    return (sample_q0, logp_elbo, n_subsample, logq, ∇logq, ∇logp_mini, T, ϵ, η, K, d, lf_n, mini_flow, mini_flow_size, data_size)
end

#######################
#### generate repeated samples in a fixed setting (used for processing results)
#######################

function MCMCsampler(mcmc::Function, args; n_samples = 100)
    #=
    mcmc::Function: {ula_elbo, mala_elbo, his_elbo, uha_elbo}
    args...: arguments for mcmc function
    d: dim of sample
    n_sample: number of samples needed
    =#
    D = Vector{Vector{Float64}}(undef, n_samples)
    el = Vector{Float64}(undef, n_samples)
    prog_bar = ProgressMeter.Progress(n_samples, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    # use multithreading
    for i = 1:n_samples
        el[i],D[i] = mcmc(args...)
        ProgressMeter.next!(prog_bar)
    end

    # return final samples
    M = reduce(hcat,D)
    # if size(M) is a row matrix, reshape to make it a N×1 matrix
    Dat = size(M, 1) > 1 ? Matrix(M') : reshape(M, size(M, 2), 1)
    return mean(el), Dat
end

# computing both KSD and KL for given setting
function MCMC_progress(ps, mcmc::Function, argparse::Function, argparse_args,
                        ksd_est::Function, grd::Function, kl_est::Function, kl_args, 
                        rel_err::Function, rel_args,
                        n_samples)

    len = size(ps, 1)
    el = Vector{Float64}(undef, len)
    KSDs = Vector{Float64}(undef, len)
    KLs = Vector{Float64}(undef, len)
    err_ms = Vector{Float64}(undef, len)
    err_covs = Vector{Float64}(undef, len)
    err_logs = Vector{Float64}(undef, len)
    EDs = Vector{Float64}(undef, len)

    # progress bar
    prog_bar = ProgressMeter.Progress(len, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    for i = 1:len
        println("sample $i / $len")
        Args = argparse(ps[i], argparse_args...)
        # taking samples for fixed setting
        el[i], D = MCMCsampler(mcmc, Args; n_samples)
        # _, D_prime = MCMCsampler(mcmc, Args; n_samples)
        KSDs[i] = ksd_est(D, grd)
        KLs[i] = kl_est(D, kl_args...)
        err_ms[i], err_covs[i], err_logs[i] = rel_err(rel_args..., D)
        EDs[i] = energy_dist(rel_args..., D)
        # update progress bar
        ProgressMeter.next!(prog_bar)
    end
    return el, KLs, err_ms, err_covs, err_logs, EDs, KSDs
end
