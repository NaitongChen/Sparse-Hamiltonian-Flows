
using LinearAlgebra, Distributions, Random, Plots, ForwardDiff, Logging, JLD, TickTock, Flux
include("../../inference/coreset/uniform.jl")
include("../../inference/mcvae/hvi.jl")
include("../../util/result.jl")
include("../../util/relative_errors.jl")
include("../../util/kl_gaussian.jl")
include("../../util/ksd.jl")
include("model.jl")

function run_uha(id)
    Random.seed!(id);
    ####################
    # run vi
    ####################
    @info "ID: $id"
    @info "training uha with minibatch ($M) in flow"
    _, ls_uha, ps_uha, time_uha = uha_vi(sample_q0, logp_elbo, n_subsample_elbo, logq, ∇logq, ∇logp_mini,
                                            n_mcmc, K, iter, d, elbo_size, ϵ0uha, T0_uha, η0;
                                            mini_flow = true, mini_flow_size = M, data_size = N,
                                            optimizer = optimizer, logging_ps = true, verbose_freq = save_freq)


    ####################
    # computing metrics
    ####################
    @info "setting arguments for progressing"
    # arguments of MCMC_progress
    ws_unif = uniform_coreset(xs, M)
    ws_inds = [1:N ;][ws_unif .> 0.]
    ws_unif = ws_unif[ws_inds]
    sub_xs = xs[ws_inds, :] 
    ∇logp_unif = (z, inds) -> ∇potential_by_hand(sub_xs,z, ws_unif) 
    ∇logp_full = (z, inds) -> ∇potential_by_hand(xs,z, ones(N)) 
    # arguments of MCMC_progress
    progress_args_uha = (sample_q0, logp_elbo, n_subsample_elbo, logq, ∇logq)
    full_args = (∇logp_full, n_mcmc, d, K, false, N, N)
    mini_args = (∇logp_unif, n_mcmc, d, K, false, M, M) # using uniform coreset in flow
    prog_full_uha = (progress_args_uha..., full_args...)
    prog_mini_uha = (progress_args_uha..., mini_args...)


    @info "computing metric for uha without minibatch in flow"
    elbo_uha_full, kl_uha_full, err_mean_uha_full, err_cov_uha_full, err_logs_full, EDs_full, KSDs_full = MCMC_progress(ps_uha, uha_elbo, argparse_uha, prog_full_uha,
                                                        ksd_imq, ∇logp, kl_gaussian_est, (post_mean, post_precision),
                                                        rel_err_no_hmc, (post_mean, post_cov),
                                                        sample_size_for_metric_computation)
    
    @info "computing metric for uha with minibatch in flow"
    elbo_uha_mini, kl_uha_mini, err_mean_uha_mini, err_cov_uha_mini, err_logs_mini, EDs_mini, KSDs_mini = MCMC_progress(ps_uha, uha_elbo, argparse_uha, prog_mini_uha,
                                                        ksd_imq, ∇logp, kl_gaussian_est, (post_mean, post_precision),
                                                        rel_err_no_hmc, (post_mean, post_cov),
                                                        sample_size_for_metric_computation)
    ####################
    # saving metrics
    ####################
    time_uha = convert.(Float64, time_uha)

    file_path = joinpath("results/", string("metric_uha_$id",".jld"))
    JLD.save(file_path, "ID", id, "ls", ls_uha, "time_uha", time_uha,
                "elbo_uha_full", elbo_uha_full, "kl_uha_full", kl_uha_full, "err_mean_uha_full", err_mean_uha_full,
                "err_cov_uha_full", err_cov_uha_full, "err_logs_full", err_logs_full, "EDs_full", EDs_full, "KSDs_full", KSDs_full,
                "elbo_uha_mini", elbo_uha_mini, "kl_uha_mini", kl_uha_mini, "err_mean_uha_mini", err_mean_uha_mini,
                "err_cov_uha_mini", err_cov_uha_mini, "err_logs_mini", err_logs_mini, "EDs_mini", EDs_mini, "KSDs_mini", KSDs_mini)
end

##############33
## Actual run 
##############
id = parse(Int, ARGS[1])
run_uha(id)