using LinearAlgebra, Distributions, Random, Statistics, Flux, ProgressMeter
include("../../util/ksd.jl")

function est_final_elbo(zs, ps, D_z, D_p, log_det, a)
    elbo = 0.
    for i in 1:size(zs,1)
        elbo += SparseFlowsT.single_elbo(a, zs[i,:], D_z[i,:], ps[i,:], D_p[i,:], log_det)
    end
    return elbo / size(zs,1)
end

function compute_metric_at_iteration(post_mean, post_var, sample_size, a, w_unc, ϵ_unc, μps, logσp, grd)
    μps_final_mat = reshape(μps, (a.number_of_refresh, a.d))
    logσp_final_mat = reshape(logσp, (a.number_of_refresh, a.d))
    r_states = []
    for i in 1:a.number_of_refresh
        push!(r_states, IdDict())
    end
    for i in 1:a.number_of_refresh
        μp = μps_final_mat[i,:]
        Σp = diagm((exp.(logσp_final_mat[i,:])).^2)
        get!(r_states[i], "key")do
            (nothing, μp, nothing, nothing, Σp, nothing)
        end
    end

    zs = a.sample_q0(sample_size)
    ps = randn(sample_size, a.d)
    D_z, D_p, log_det = SparseFlowsT.sampler(a, sample_size, ϵ_unc, w_unc, r_states, copy(zs), copy(ps))

    # @info "estimating all metrics"
    KL = kl_gaussian(vec(mean(D_z, dims=1)), cov(D_z), post_mean, post_var)
    mrel, srel, slogrel = rel_err_no_hmc(post_mean, post_var, D_z)
    ELBO = est_final_elbo(zs, ps, D_z, D_p, log_det, a)
    ed = energy_dist(post_mean, post_var, D_z)
    ksd = ksd_imq(D_z, grd)

    return ELBO, KL, mrel, srel, slogrel, ed, ksd
end

function compute_metric_all_iters(save_freq, post_mean, post_var, sample_size_for_metric_computation, a, w_unc_hist, ϵ_unc_hist, μps_hist, logσp_hist, grd)
    iters = [1:save_freq:size(w_unc_hist,1) ;]
    num_iters = size(iters,1)

    KLs = zeros(num_iters)
    mrels = zeros(num_iters)
    srels = zeros(num_iters)
    slogrels = zeros(num_iters)
    elbos = zeros(num_iters)
    EDs = zeros(num_iters)
    KSDs = zeros(num_iters)

    prog_bar = ProgressMeter.Progress(iter, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i in 1:num_iters
        elbos[i], KLs[i], mrels[i], srels[i], slogrels[i], EDs[i], KSDs[i] = compute_metric_at_iteration(post_mean, post_var, sample_size_for_metric_computation, a, w_unc_hist[iters[i],:], ϵ_unc_hist[iters[i],:], μps_hist[iters[i],:], logσp_hist[iters[i],:], grd)
        ProgressMeter.next!(prog_bar)
    end

    return elbos, KLs, mrels, srels, slogrels, EDs, KSDs
end

function compute_KSD_all_iters(save_freq, sample_size_for_metric_computation, a, w_unc_hist, ϵ_unc_hist, μps_hist, logσp_hist, c, β, grd)
    iters = [1:save_freq:size(w_unc_hist,1) ;]
    num_iters = size(iters,1)

    KSDs = zeros(num_iters)

    prog_bar = ProgressMeter.Progress(iter, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i in 1:num_iters
        KSDs[i] = compute_KSD_at_iteration(sample_size_for_metric_computation, a, w_unc_hist[iters[i],:], ϵ_unc_hist[iters[i],:], μps_hist[iters[i],:], logσp_hist[iters[i],:], c, β, grd)
        ProgressMeter.next!(prog_bar)
    end

    return KSDs
end

function compute_KSD_at_iteration(sample_size, a, w_unc, ϵ_unc, μps, logσp, c, β, grd)
    μps_final_mat = reshape(μps, (a.number_of_refresh, a.d))
    logσp_final_mat = reshape(logσp, (a.number_of_refresh, a.d))
    r_states = []
    for i in 1:a.number_of_refresh
        push!(r_states, IdDict())
    end
    for i in 1:a.number_of_refresh
        μp = μps_final_mat[i,:]
        Σp = diagm((exp.(logσp_final_mat[i,:])).^2)
        get!(r_states[i], "key")do
            (nothing, μp, nothing, nothing, Σp, nothing)
        end
    end

    zs = a.sample_q0(sample_size)
    ps = randn(sample_size, a.d)
    D_z, D_p, log_det = SparseFlowsT.sampler(a, sample_size, ϵ_unc, w_unc, r_states, copy(zs), copy(ps))

    return imq_ksd(D_z, c, β, grd)
end