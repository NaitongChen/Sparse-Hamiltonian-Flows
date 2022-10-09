using LinearAlgebra, Distributions, Random, Plots, ForwardDiff, Logging, JLD, TickTock, Flux
include("model.jl")
include("../../inference/coreset/uniform.jl")
include("../../inference/coreset/OMP.jl")
include("../../inference/sampling/adapt_NUTS.jl")
include("../../inference/sampling/hmc.jl")
include("../../util/result.jl")
include("../../util/kl_gaussian.jl")
include("../../util/relative_errors.jl")
include("../../util/ksd.jl")

function run_coreset_methods(id)
    Random.seed!(id);
    z0 = sample_q0(1)

    KL_unif = zeros(length(Ms))
    mrel_unif = zeros(length(Ms))
    srel_unif = zeros(length(Ms))
    slogrel_unif = zeros(length(Ms))
    ED_unif = zeros(length(Ms))
    KSD_unif = zeros(length(Ms))

    KL_omp = zeros(length(Ms))
    mrel_omp = zeros(length(Ms))
    srel_omp = zeros(length(Ms))
    slogrel_omp = zeros(length(Ms))
    ED_omp = zeros(length(Ms))
    KSD_omp = zeros(length(Ms))

    KL_sp = zeros(length(Ms))
    mrel_sp = zeros(length(Ms))
    srel_sp = zeros(length(Ms))
    slogrel_sp = zeros(length(Ms))
    ED_sp = zeros(length(Ms))
    KSD_sp = zeros(length(Ms))

    @info "construct OMP coreset builder"
    build = OMP_coreset(xs, num_random_proj, logp_ind_ind, ljd, d)
    grd = zz -> ∇potential_by_hand(xs, zz, ones(size(xs,1)))

    for i in 1:length(Ms)
        @info "constructing OMP coreset"
        ws_omp = build(Ms[i])
        inds_omp = [1:N ;][build(Ms[i]) .> 0]
        omp_post = z -> coreset_posterior(z, ws_omp)
        ∇omp_post = z -> ForwardDiff.gradient(omp_post, z)
        @info "sampling from OMP coreset posterior using HMC"
        Random.seed!(id);
        z0 = sample_q0(1)
        samples_omp = adapt_hmc(z0, accept_ratio, lf_n, omp_post, ∇omp_post, sample_size_for_metric_computation, sample_size_for_metric_computation)
        @info "computing error metrics for OMP coreset HMC"
        KL_omp[i] = kl_gaussian(vec(mean(samples_omp, dims=1)), cov(samples_omp), post_mean, post_var)
        mrel_omp[i], srel_omp[i], slogrel_omp[i] = rel_err_no_hmc(post_mean, post_var, samples_omp)
        ED_omp[i] = energy_dist(post_mean, post_var, samples_omp)
        KSD_omp[i] = ksd_imq(samples_omp, grd)
    end

    for i in 1:length(Ms)
        @info "constructing uniform coreset"
        ws_unif = uniform_coreset(xs, Ms[i])
        unif_post = z -> coreset_posterior(z, ws_unif)
        ∇unif_post = z -> ForwardDiff.gradient(unif_post, z)
        @info "sampling from uniform coreset posterior using HMC"
        samples_unif = adapt_hmc(z0, accept_ratio, lf_n, unif_post, ∇unif_post, sample_size_for_metric_computation, sample_size_for_metric_computation)
        @info "computing error metrics for uniform coreset HMC"
        KL_unif[i] = kl_gaussian(vec(mean(samples_unif, dims=1)), cov(samples_unif), post_mean, post_var)
        mrel_unif[i], srel_unif[i], slogrel_unif[i] = rel_err_no_hmc(post_mean, post_var, samples_unif)
        ED_unif[i] = energy_dist(post_mean, post_var, samples_unif)
        KSD_unif[i] = ksd_imq(samples_unif, grd)
    end

    for i in 1:length(Ms)
        @info "training sparse flows"
        # include("model.jl")
        a.inds = nothing
        a.sub_xs = nothing
        a.M = Ms[i]
        Random.seed!(id);
        ϵ_unc_hist, w_unc_hist, _, _, _, _, r_states = SparseFlowsT.sparse_flows(a, ϵ_unc);
        @info "sampling from trained flows"
        zs = a.sample_q0(sample_size_for_metric_computation)
        ps = randn(sample_size_for_metric_computation, a.d)
        D_z, _, _ = SparseFlowsT.sampler(a, sample_size_for_metric_computation, ϵ_unc_hist[end,:], w_unc_hist[end,:], r_states, zs, ps)
        @info "computing error metrics of sparse flows"
        KL_sp[i] = kl_gaussian(vec(mean(D_z, dims=1)), cov(D_z), post_mean, post_var)
        mrel_sp[i], srel_sp[i], slogrel_sp[i] = rel_err_no_hmc(post_mean, post_var, D_z)
        ED_sp[i] = energy_dist(post_mean, post_var, D_z)
        KSD_sp[i] = ksd_imq(D_z, grd)
    end

    a.inds = nothing
    a.sub_xs = nothing
    a.M = M

    @info "saving results"
    file_path = joinpath("results/", string("coreset_metric_$id", ".jld"))
    JLD.save(file_path, "id", id,
            "KL_unif", KL_unif, "KL_omp", KL_omp, "KL_sp", KL_sp,
            "mrel_unif", mrel_unif, "mrel_omp", mrel_omp, "mrel_sp", mrel_sp,
            "srel_unif", srel_unif, "srel_omp", srel_omp, "srel_sp", srel_sp,
            "slogrel_unif", slogrel_unif, "slogrel_omp", slogrel_omp, "slogrel_sp", slogrel_sp,
            "ED_unif", ED_unif, "ED_omp", ED_omp, "ED_sp", ED_sp,
            "KSD_unif", KSD_unif, "KSD_omp", KSD_omp, "KSD_sp", KSD_sp)
end

run_coreset_methods(parse(Int, ARGS[1]))
