using LinearAlgebra, Distributions, Random, Plots, ForwardDiff, Logging, JLD, TickTock, Flux
include("../../inference/sparse_flows/density_eval.jl")
include("../../inference/coreset/uniform.jl")
include("../../inference/mcvae/sampler.jl")
include("../../inference/mcvae/density_eval.jl")
include("../../util/timing.jl")
include("../../inference/sampling/adapt_NUTS.jl")
include("../../inference/sampling/hmc.jl")
include("model.jl")

function run_timing()
    Random.seed!(2022)
    z0, ρ0 = randn(d), randn(d)

    @info "timing density evaluation uha"
    uha_flow_args = (copy(z0), copy(ρ0), logq, ∇logγ, ones(number_of_refresh), 0.0001*ones(d), η0, number_of_refresh, d, K)
    time_eval_uha = noob_timing(uha_density_eval, uha_flow_args...; n_run = sample_size_for_metric_computation)
    
    # his
    @info "timing density evaluation his"
    his_flow_args = (copy(z0), copy(ρ0), ∇logp, ones(number_of_refresh), 0.0001*ones(d), number_of_refresh, d, K)
    time_eval_his = noob_timing(his_density_eval, his_flow_args...; n_run = sample_size_for_metric_computation)

    @info "timing density evaluation sparse flows"
    time_eval_sp = noob_timing(sp_flow_marg_density, z0, ρ0, K, xs[1:M,:], lf_n; n_run = sample_size_for_metric_computation)

    @info "timing sample generation sparse flows"
    a.iter = 10
    ϵ_unc_hist, w_unc_hist, _, _, _, _, r_states = SparseHamiltonianFlows.sparse_flows(a, ϵ_unc)
    time_sample_sp = noob_timing(SparseHamiltonianFlows.sampler, a, 1, ϵ_unc_hist[a.iter,:], w_unc_hist[a.iter,:], r_states, Matrix(z0'), Matrix(ρ0'); n_run = sample_size_for_metric_computation)

    @info "timing sample generation uha"
    time_sample_uha = noob_timing(uha_sample_single, sample_q0, ∇logγ, T_all(T0_uha), ϵ0uha, η0, number_of_refresh, d, K; n_run = sample_size_for_metric_computation)

    @info "timing sample generation his"
    time_sample_his = noob_timing(his_sample_single, sample_q0, ∇logp, logistic.(T0_his), ϵ0his, number_of_refresh, d, K; n_run = sample_size_for_metric_computation)

    @info "timing sample generation NUTS"
    time_sample_nuts = noob_timing(NUTS, z0, nuts_acc_ratio, ljd, ∇logp, 1, 1; n_run = sample_size_for_metric_computation)

    @info "timing sample generation hmc"
    time_sample_hmc = noob_timing(adapt_hmc, z0, nuts_acc_ratio, lf_n, ljd, ∇logp, 1, 1; n_run = sample_size_for_metric_computation)

    @info "timing sample generation his mini"
    ws_unif = w_unc_hist[a.iter,:]
    ∇logp_unif = z-> ∇potential_by_hand(a.sub_xs,z, ws_unif)
    time_sample_his_mini = noob_timing(his_sample_single, sample_q0, ∇logp_unif, logistic.(T0_his), ϵ0his, number_of_refresh, d, K; n_run = sample_size_for_metric_computation)

    @info "timing sample generation uha mini"
    ∇logγ_unif(z, β) = β * ∇logp_unif(z) .+ (1.0 - β) * ∇logq(z)
    time_sample_uha_mini = noob_timing(uha_sample_single, sample_q0, ∇logγ_unif, T_all(T0_uha), ϵ0uha, η0, number_of_refresh, d, K; n_run = sample_size_for_metric_computation)

    @info "timing density evaluation uha"
    uha_flow_args_mini = (copy(z0), copy(ρ0), logq, ∇logγ_unif, ones(number_of_refresh), 0.0001*ones(d), η0, number_of_refresh, d, K)
    time_eval_uha_mini = noob_timing(uha_density_eval, uha_flow_args_mini...; n_run = sample_size_for_metric_computation)
    
    # his
    @info "timing density evaluation his"
    his_flow_args_mini = (copy(z0), copy(ρ0), ∇logp_unif, ones(number_of_refresh), 0.0001*ones(d), number_of_refresh, d, K)
    time_eval_his_mini = noob_timing(his_density_eval, his_flow_args_mini...; n_run = sample_size_for_metric_computation)

    @info "saving time results"
    file_path = joinpath("results/", string("vae_timing",".jld"))
    JLD.save(file_path, "time_eval_his", time_eval_his, "time_eval_sp", time_eval_sp, "time_eval_uha", time_eval_uha,
                    "time_eval_his_core", time_eval_his_mini, "time_eval_uha_core", time_eval_uha_mini,
                    "time_sample_sp", time_sample_sp, "time_sample_his", time_sample_his, "time_sample_uha", time_sample_uha,
                    "time_sample_his_mini", time_sample_his_mini, "time_sample_uha_mini", time_sample_uha_mini,
                    "time_sample_nuts", time_sample_nuts, "time_sample_hmc", time_sample_hmc)
end

############################
#  Actually running
############################
run_timing()
