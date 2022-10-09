using LinearAlgebra, Distributions, Random, Plots, ForwardDiff, Logging, JLD, TickTock, Flux
include("model.jl")

function run_flow_step()
    id = 2022
    Random.seed!(id);

    @info "train sparse flows"
    ϵ_unc_hist, w_unc_hist, μps_hist, logσp_hist, ls_hist, time_hist, r_states = SparseFlowsT.sparse_flows(a, ϵ_unc);

    @info "estimating intermediate elbo"
    elbos = SparseFlowsT.est_elbo_trained_flow(a, ϵ_unc_hist[end,:], w_unc_hist[end,:], r_states, 100*sample_size_for_metric_computation)

    @info "saving results"
    file_path = joinpath("results/", string("step_$id",".jld"))
    JLD.save(file_path, "id", id, "elbos", elbos)
end

run_flow_step()
