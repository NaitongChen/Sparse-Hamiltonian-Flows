
using LinearAlgebra, Distributions, Random, Plots, ForwardDiff, Logging, JLD, TickTock, Flux
include("../../inference/coreset/uniform.jl")
include("../../inference/mcvae/hvi.jl")
include("../../util/result.jl")
include("../../util/relative_errors.jl")
include("../../util/kl_gaussian.jl")
include("../../util/ksd.jl")
include("model.jl")

function run_his(id)
    Random.seed!(id);
    ####################
    # run vi
    ####################
    @info "ID: $id"
    @info "training his with minibatch ($M) in flow"
    # third param n_subsample is number of samples used for estimating elbo
    # mini_flow_size is number of samples used for leapfrog
    _, ls_his, ps_his, time_his = his_vi(sample_q0, logp_elbo, n_subsample_elbo, logq, ∇logp_mini,
                                            n_mcmc, K, iter, d, elbo_size, ϵ0his, T0_his;
                                            mini_flow = true, mini_flow_size = M, data_size = N,
                                            optimizer = optimizer, logging_ps = true, verbose_freq = save_freq)

    ####################
    # computing ksd trace
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
    progress_args_his = (sample_q0, logp_elbo, n_subsample_elbo, logq)
    full_args = (∇logp_full, n_mcmc, d, K, false, N, N)
    mini_args = (∇logp_unif, n_mcmc, d, K, false, M, M) # using uniform coreset in flow
    prog_full_his = (progress_args_his..., full_args...)
    prog_mini_his = (progress_args_his..., mini_args...)

    @info "computing metric for his without minibatch in flow"
    # NOTE: KSD and its argument is not used 
    elbo_his_full, kl_his_full, err_mean_his_full, err_cov_his_full, err_logs_full, EDs_full, KSDs_full = MCMC_progress(ps_his, his_elbo, argparse_his, prog_full_his,
                                                        ksd_imq, ∇logp, kl_gaussian_est, (post_mean, post_precision),
                                                        rel_err_no_hmc, (post_mean, post_cov),
                                                        sample_size_for_metric_computation)

    @info "computing metric for his with minibatch in flow"
    elbo_his_mini, kl_his_mini, err_mean_his_mini, err_cov_his_mini, err_logs_mini, EDs_mini, KSDs_mini = MCMC_progress(ps_his, his_elbo, argparse_his, prog_mini_his,
                                                        ksd_imq, ∇logp, kl_gaussian_est, (post_mean, post_precision),
                                                        rel_err_no_hmc, (post_mean, post_cov),
                                                        sample_size_for_metric_computation)
    ####################
    # saving metrics
    ####################
    time_his = convert.(Float64, time_his)

    file_path = joinpath("results/", string("metric_his_$id",".jld"))
    JLD.save(file_path, "ID", id, "ls", ls_his, "time_his", time_his,
                "elbo_his_full", elbo_his_full, "kl_his_full", kl_his_full, "err_mean_his_full", err_mean_his_full,
                "err_cov_his_full", err_cov_his_full, "err_logs_full", err_logs_full, "EDs_full", EDs_full, "KSDs_full", KSDs_full,
                "elbo_his_mini", elbo_his_mini, "kl_his_mini", kl_his_mini, "err_mean_his_mini", err_mean_his_mini,
                "err_cov_his_mini", err_cov_his_mini, "err_logs_mini", err_logs_mini, "EDs_mini", EDs_mini, "KSDs_mini", KSDs_mini)
end

##############33
## Actual run 
##############
id = parse(Int, ARGS[1])
run_his(parse(Int, ARGS[1]))