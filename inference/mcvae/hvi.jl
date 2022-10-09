using Zygote: gradient
using ForwardDiff
using Flux
include("elbo.jl")


################3
# VI with Hamiltonian dynamics (all learnable tempering schedule)
#################

### HISvae (https://arxiv.org/pdf/1805.11328.pdf)

### TODO: add mini_flow option
function his_vi(sample_q0::Function, logp::Function, n_subsample::Int, logq::Function, ∇logp_mini::Function,
                K::Int, n_lfrg::Int, niters::Int, d::Int, elbo_size::Int, 
                ϵ0::Vector{Float64}, logit_T0_his::Vector{Float64};
                mini_flow::Bool= false, mini_flow_size::Int = 0, data_size::Int = 0, 
                optimizer = Flux.ADAM(1e-3), stratified_sampler = nothing,
                kwargs...)

    # init stepsize
    logϵ = log.(ϵ0)
    T_logit = logit_T0_his  #ones(K)
    ps = Flux.params(logϵ, T_logit)

    # define loss function
    loss = ()-> begin
        # transform params
        Temp_sched = logistic.(T_logit)
        ϵ = @. expm1(logϵ) + 1.0

        # unbiased elbo estimates
        elbo = 0.0
        for i = 1: elbo_size
            elbo_i = -his_elbo(sample_q0, logp, n_subsample, logq, ∇logp_mini, Temp_sched, ϵ, K, d, n_lfrg, 
                                mini_flow, mini_flow_size, data_size; sampler = stratified_sampler)[1]
            elbo = elbo + (1. / elbo_size) * elbo_i
        end
        return elbo
    end

    # his tranining step
    elbo_log, ps_log, time_log = vi_train!(niters, loss, ps, optimizer; kwargs...)

    return [[copy(p) for p in ps]], elbo_log, ps_log, time_log
end




## Uncorrected Hamultonian annealing VAE (https://arxiv.org/pdf/2107.04150.pdf)
function uha_vi(sample_q0::Function, logp::Function, n_subsample::Int,logq::Function,  ∇logq::Function, ∇logp_mini::Function,
                K::Int, n_lfrg::Int, niters::Int, d::Int, elbo_size::Int, 
                ϵ0::Vector{Float64}, logit_T0_uha::Vector{Float64}, logit_η0::Vector{Float64}; 
                mini_flow::Bool= false, mini_flow_size::Int = 0, data_size::Int = 0, 
                optimizer = Flux.ADAM(1e-3), stratified_sampler = nothing,
                kwargs...)

    # init stepsize
    logϵ =  log.(ϵ0)
    logit_η = logit_η0
    T_logit = logit_T0_uha  #zeros(K-1) 
    ps = Flux.params(logϵ, logit_η, T_logit)

    # define loss
    loss =() -> begin
        # transform params back
        Temp_sched = T_all(T_logit) 
        ϵ = @. expm1(logϵ) + 1.0
        η = Zygote.LogExpFunctions.logistic.(logit_η) # damping coef 
        
        # unbiased elbo estimates
        elbo = 0.0
        for i = 1: elbo_size
            elbo_i = -uha_elbo(sample_q0, logp, n_subsample, logq, ∇logq, ∇logp_mini, Temp_sched, ϵ, η, K, d, n_lfrg,
                                mini_flow, mini_flow_size, data_size; sampler = stratified_sampler)[1]
            elbo = elbo + (1. / elbo_size) * elbo_i
        end
        return elbo
    end

    # uha training step
    elbo_log, ps_log, time_log = vi_train!(niters, loss, ps, optimizer; kwargs...)

    return [[copy(p) for p in ps]], elbo_log, ps_log, time_log
end

