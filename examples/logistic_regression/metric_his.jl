# cd("/home/zuheng/Research/compressing-hamiltonian-NF/code/experiments/examples/logistic_regression")
using LinearAlgebra, Distributions, Random, Plots, ForwardDiff, Logging, JLD, TickTock, Flux
include("../../inference/coreset/uniform.jl")
include("../../inference/mcvae/hvi.jl")
include("../../inference/mcvae/sampler.jl")
# include("../../util/result.jl")
include("../../util/relative_errors.jl")
include("../../util/kl_gaussian.jl")
include("../../util/ksd.jl")
include("model.jl")

function run_his(id)
    Random.seed!(id)
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
                                            optimizer = optimizer, stratified_sampler = stratified_sampling, logging_ps = true, verbose_freq = save_freq)

    ####################
    # computing ksd trace
    ####################
    @info "setting arguments for progressing"
    # arguments of MCMC_progress
    ws_unif = uniform_coreset(xs, M)
    ws_inds = [1:N ;][ws_unif .> 0.]
    ws_unif=  ws_unif[ws_inds]
    sub_xs = xs[ws_inds, :] 
    ∇logp_unif = z -> ∇potential_by_hand(sub_xs,z, ws_unif) 
    ∇logp_full = z -> ∇potential_by_hand(xs,z, ones(N)) 
    # arguments of MCMC_progress
    prog_full_his = (logq, sample_q0, ∇logp_full, logp_elbo, n_subsample_elbo, n_mcmc, d, K)
    prog_mini_his = (logq, sample_q0, ∇logp_unif, logp_elbo, n_subsample_elbo, n_mcmc, d, K)

    @info "computing metric for his without minibatch in flow"
    # NOTE: KSD and its argument is not used 
    grd = zz -> ∇potential_by_hand(xs, zz, ones(size(xs,1)))
    elbo_his_full, kl_his_full, err_mean_his_full, err_cov_his_full, err_logs_full, EDs_full, KSDs_full = HVI_progress(ps_his, his_sample_with_elbo, reparam_his, prog_full_his,
                                                                                                                        kl_gaussian_est, (post_mean, post_precision),
                                                                                                                        rel_err_no_hmc, (post_mean, post_cov),
                                                                                                                        sample_size_for_metric_computation, ksd_imq, grd)

    @info "computing metric for his with minibatch in flow"
    elbo_his_mini, kl_his_mini, err_mean_his_mini, err_cov_his_mini, err_logs_mini, EDs_mini, KSDs_mini = HVI_progress(ps_his, his_sample_with_elbo, reparam_his, prog_mini_his,
                                                                                                                        kl_gaussian_est, (post_mean, post_precision),
                                                                                                                        rel_err_no_hmc, (post_mean, post_cov),
                                                                                                                        sample_size_for_metric_computation, ksd_imq, grd)
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