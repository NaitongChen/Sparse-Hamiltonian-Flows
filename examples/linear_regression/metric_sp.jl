using LinearAlgebra, Distributions, Random, Plots, ForwardDiff, Logging, JLD, TickTock, Flux
include("model.jl")
include("../../inference/sparse_flows/compute_metric.jl")
include("../../inference/sparse_flows/density_eval.jl")
include("../../util/relative_errors.jl")
include("../../util/kl_gaussian.jl")
include("../../util/ksd.jl")
include("../../inference/coreset/posterior_laplace_approx.jl")

##################
## sparse flow
##################

function run_sp(id)
    Random.seed!(id);

    @info "ID: $id"
    @info "training sparse flows"
    ϵ_unc_hist, w_unc_hist, μps_hist, logσp_hist, ls_hist, time_hist, r_states = SparseFlowsT.sparse_flows(a, ϵ_unc);

    @info "computing metrics for conditional sp flow"
    grd = zz -> ∇potential_by_hand(xs, zz, ones(size(xs,1)))
    ELBOs, KLs, mrels, srels, slogrels, EDs, KSDs = compute_metric_all_iters(save_freq, post_mean, post_var, sample_size_for_metric_computation, a, w_unc_hist, ϵ_unc_hist, μps_hist, logσp_hist, grd)

    time_hist = convert.(Float64, time_hist)

    Random.seed!(id);
    μ_laplace, Σ_laplace = laplace_approx(ljd, d, c_prior, m)
    KL = kl_gaussian(μ_laplace, Σ_laplace, post_mean, post_var)
    mrel, srel, slogrel = rel_err_no_dat(post_mean, post_cov, μ_laplace, Σ_laplace)
    distNormal = MvNormal(μ_laplace, Σ_laplace)
    D = Matrix(rand(distNormal, sample_size_for_metric_computation)')
    ED = energy_dist(post_mean, post_cov, D)
    KSD_ind = ksd_imq(D, grd)

    file_path = joinpath("results/", string("new_metric_sp_$id",".jld"))
    JLD.save(file_path, "ID", id, "ls", ls_hist, "time_hist", time_hist,
                "ELBOs", ELBOs, "KLs", KLs, "mrels", mrels, "srels", srels, "slogrels", slogrels, "EDs", EDs, "KSDs", KSDs,
                "KL", KL, "mrel", mrel, "srel", srel, "slogrel", slogrel, "ED", ED, "KSD_ind", KSD_ind)
end

##############
## Actual run
##############
id = parse(Int, ARGS[1])
run_sp(id)
