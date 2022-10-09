using Zygote
using Zygote: @ignore, dropgrad
include("util.jl")
include("../../util/train.jl")



"""
ELBO: Hamiltonian importance sampling + VI (normalizing flow)

No bridging distribution and use momentum tempering

optimizing (no VI param included):
1. tempering sched 
2. leapfrog stepsize (d-dim)
"""
function his_elbo(sample_q0::Function, logp::Function, n_subsample::Int ,logq::Function, ∇logp_mini::Function, 
                T_alpha::Vector{Float64}, ϵ::Vector{Float64}, 
                K::Int, d::Int, n_lfrg::Int, 
                mini_flow::Bool, mini_flow_size::Int, data_size::Int; sampler = nothing)
    #= 
    Args: 
    logp(z) : log posterior density 
    logq(z, ϕ): log VI distribution
    ∇logp_mini(z, inds) :score function of subsampled posterior used for subsampled flow
    f_reparam(u, ϕ)::Function : u∼N(0, I), z =f_reparam(u, ϕ) ∼ q_ϕ 
    T_alpha::Vector : Tempering sched T_alpha = sqrt.[β0/β1, .., βK-1/βK], βK = 1. 
    ϵ::Vector : leapfrog stepsize
    ϕ:Vector :VI param
    K: number of his transitions 
    d: dim of posterior

    output: 
    elbo 
    (if return_last_sample) z: K-th sample 
    =# 

    z0, ρ0 = sample_q0(), randn(d) #sample from standard normal
    z = copy(z0)
    β0_sqrt = prod(T_alpha)
    ρ = ρ0 ./(β0_sqrt + 1e-8)

    # determine whether subsample in the flow
    if mini_flow
        inds = zeros(Int64, mini_flow_size)
        ignore() do 
            if isnothing(sampler)
                inds = sort(sample(1:data_size, mini_flow_size, replace = false))
            else
                inds = sampler(mini_flow_size)
            end
        end
        ∇logp = z -> ∇logp_mini(z, inds) 
    else
        ∇logp = z -> ∇logp_mini(z, ones(Int64, data_size))
    end

    for id = 1:K
        # no refresh but tempering momentum
        ρ, z = his_one_transition(∇logp, T_alpha, id, ρ, z, ϵ, n_lfrg)
    end

    # compute elbo
    elbo = ( logp(z, n_subsample) - 0.5*(ρ' * ρ) 
            - logq(z0) + 0.5 * (ρ0' * ρ0) )

    return (elbo, z) 
end


function his_one_transition(∇logp, T_alpha, id, ρ_current, z_current, ϵ, n_lfrg)
    #= 
    ∇logγ(z, β, ϕ): β ∇logp + (1-β) ∇logq_ϕ 
    T_alpha::Vector : Tempering sched T_alpha = sqrt.[β0/β1, .., βK-1/βK], βK = 1. 
    id::Int : transition id 
    z_current: curent position 
    ρ_current: current momentum var 
    logϵ::Float : leapfrog log stepsize 
    =# 
    
    for i in 1:n_lfrg 
        ρ_current +=  0.5 .* ϵ .* ∇logp(z_current) 
        z_current += ϵ .* ρ_current
        ρ_current += 0.5 .* ϵ .* ∇logp(z_current) 
    end 

    return T_alpha[id]*ρ_current , z_current 
end


"""
ELBO: Uncorrected Hamiltonian anealing + VI (SMC with sepcific reverse kernel)

Bridging distribution + partial momentum refreshment 

optimizing:
1. temp_sched 
2. leapfrog stepsize
"""
function uha_elbo(sample_q0::Function, logp::Function, n_subsample::Int, logq::Function, ∇logq::Function, ∇logp_mini::Function, 
                Temp_sched::Vector{Float64}, ϵ::Vector{Float64}, η::Vector{Float64}, 
                K::Int, d::Int, n_lfrg::Int, 
                mini_flow::Bool, mini_flow_size::Int, data_size::Int; sampler = nothing)
    #=
    Args: 
    logp(z) : log posterior density 
    logq(z, ϕ): log VI distribution
    logϵ::Float : log of leapfrog stepsize 
    logit_η::Float : transformed damping parameter η (momentum is only partially updated)
    D::Vector : D^2 = Σ (diagonal cov of kinetic energy) 
    ϕ:Vector :VI param
    K: number of his transitions 
    d: dim of posterior

    output: 
    elbo 
    (if return_last_sample) z: K-th sample 
    =#
    z0, ρ0 = sample_q0(), randn(d)
    z, ρ = copy(z0), copy(ρ0) 
    logW = -logq(z)

    # determine whether subsample in the flow
    if mini_flow
        inds = zeros(Int64, mini_flow_size)
        ignore() do 
            if isnothing(sampler)
                inds = sort(sample(1:data_size, mini_flow_size, replace = false))
            else
                inds = sampler(mini_flow_size)
            end
        end
        ∇logp = z -> ∇logp_mini(z, inds) 
    else
        ∇logp = z -> ∇logp_mini(z, ones(Int64, data_size))
    end
	∇logγ(z, β) = β * ∇logp(z) .+ (1.0 - β) * ∇logq(z)

    # run uha
    for id = 1:K 
        z, ρ, logW = uha_one_transition(∇logγ, Temp_sched, id, z, ρ, logW, ϵ, η, n_lfrg)
    end

    elbo = logW + logp(z, n_subsample)

    return (elbo, z) 
end
     

function uha_one_transition(∇logγ::Function, T, id, z_current, ρ_current, logW, 
                            ϵ, η, n_lfrg)
    #= 
    ∇logγ(z, β, ϕ): β ∇logp + (1-β) ∇logq_ϕ 
    T::Vector : Tempering sched 
    id::Int : transition id 
    z_current: curent position 
    ρ_current: current momentum var 
    logW::Float : ELBO (log ratio been updated)
    ϵ::Vector: leapfrog stepsize 
    logit_η::Float : transformed damping parameter η ∈ (0, 1)
    =# 
    
    # ϵ = @. expm1(logϵ) + 1.0
    # η = Zygote.LogExpFunctions.logistic.(logit_η) # damping coef 
    
    ρ0 = copy(ρ_current)

    # leapfrog transition  
    for i in 1:n_lfrg 
        ρ_current += 0.5 .* ϵ .* ∇logγ(z_current, T[id]) 
        z_current += ϵ .* ρ_current
        ρ_current += 0.5 .* ϵ .* ∇logγ(z_current, T[id]) 
    end
    
    # partially resample momentum
    u = @ignore randn(size(ρ_current, 1))
    ρ_refresh = @.(sqrt(1.0 - η^2.0)*u + η*ρ_current)
   
    #update elbo 
    logW += 0.5*(ρ0' * ρ0) - 0.5*(ρ_refresh' * ρ_refresh)  

    return  z_current, ρ_refresh , logW
end 


























##############################3
## currently not used
################################

"""
Unadjusted Langevin algorithm + VI 
(old version --- not used for current expt)
"""
function ula_elbo(logp::Function, logq, logγ, ∇logγ, f_reparam, ϕ, Temp_sched, ϵ, K, d;
                    learn_stepsize = false, target_acc_rate = 0.9) 
    #= 
    Args: 
    logp(z) : log posterior density 
    logq(z, ϕ): log VI distribution
    logγ(z, β, ϕ): β logp + (1-β) logq_ϕ 
    ∇logγ(z, β, ϕ): β ∇logp + (1-β) ∇logq_ϕ 
    f_reparam(u, ϕ)::Function : u∼N(0, I), z =f_reparam(u, ϕ) ∼ q_ϕ 
    ϕ:Vector :VI param
    Temp_sched::Vector : Tempering sched
    ϵ:Float : langevin stepsize
    update_stepsize::String : langevin stepseize adaptation scheme (now only update by targeting at specific acceptance ratio)
    target_acc_rate::Float : target acceptance ratio for adapting langevin stepsize
    K: number of his transitions 
    d: dim of posterior
    output: 
    logW: elbo 
    (if return_last_sample) z: K-th sample 
    =# 

    u = randn(d) #smaple from standard normal
    z = f_reparam(u, ϕ) # z = f_reparam(u, ϕ) ∼ q_ϕ 
    logW = -logq(z, ϕ)
 
    for id = 1:K 
        z_old, z, logW = ula_one_transition(∇logγ, Temp_sched, id, z, logW, ϵ, ϕ)
       
        # adapt stepsize---Float is not mutable
        if !learn_stepsize
            log_ratio = @ignore log_MH_lang_ratio(logγ, ∇logγ, z_old, z, Temp_sched, id, ϕ, ϵ)
            ϵ = adapt_stepsize!(ϵ, false, log_ratio, log(target_acc_rate) )
        end
    end 
    logW += logp(z)

    return (logW, z)
end


function ula_one_transition(∇logγ::Function, T::Vector{Float64}, id, z_current, logW, η, ϕ)
    #= 
    ∇logγ(z, β, ϕ): β ∇logp + (1-β) ∇logq_ϕ 
    T::Vector : Tempering sched
    id::Int : transition id 
    z_current: curent position 
    logW::Float : elbo
    η::Float :langevin stepseize 
    ϕ:Vector :VI param
    =# 

    z_old = copy(z_current)
    u = randn(size(z_current, 1))

    # one langevin step 
    z_current = z_current .+ η .* ∇logγ(z_current, T[id], ϕ) .+ sqrt.(2.0η) .* u
    # update log weight 
    V1= z_current .- z_old .- η.*∇logγ(z_old, T[id], ϕ)
    V2 = z_old .- z_current .-η.*∇logγ(z_current, T[id], ϕ)

    logW += 0.25 * ( V1' * (V1./η) - V2' * (V2./η) )
    return z_old, z_current, logW
end



"""
MALA + VI 
(old version --- not used for current expt)

The gradient estimates with control-vareiate is not implemented---need to code via buffer+ignore to save K-values
"""
function mala_elbo(logp::Function, logq::Function, logγ::Function, ∇logγ::Function, f_reparam::Function, 
                    ϕ::Vector{Float64}, Temp_sched::Vector{Float64}, ϵ, K::Int, d::Int;
                     learn_stepsize = false , target_acc_rate = 0.8)
    #= 
    Args: 
    logp(z) : log posterior density 
    logq(z, ϕ): log VI distribution
    logγ(z, β, ϕ): β logp + (1-β) logq_ϕ 
    ∇logγ(z, β, ϕ): β ∇logp + (1-β) ∇logq_ϕ 
    f_reparam(u, ϕ)::Function : u∼N(0, I), z =f_reparam(u, ϕ) ∼ q_ϕ 
    ϕ:Vector :VI param
    Temp_sched::Vector : Tempering sched
    ϵ:Float : langevin stepsize
    update_stepsize::String : langevin stepseize adaptation scheme (now only update by targeting at specific acceptance ratio)
    target_acc_rate::Float : target acceptance ratio for adapting langevin stepsize
    K: number of his transitions 
    d: dim of posterior
    output: 
    logW: elbo 
    log_alpha: log acceptance ratio (used for gradient estimator)
    (if return_last_sample) z: K-th sample 
    =# 
    
    
    u = randn(d) #smaple from standard normal
    z = f_reparam(u, ϕ) # z = f_reparam(u, ϕ) ∼ q_ϕ 
    logW, log_alpha = 0.0 , 0.0

    for id = 1:K
        z, logW, log_alpha, log_ratio = mala_one_transition(logp, logq, logγ, ∇logγ, Temp_sched, id, 
                                                            z, logW, log_alpha, ϵ, ϕ) 
                                                            
        # adapt stepsize
        if !learn_stepsize
           ϵ = adapt_stepsize!(ϵ, false, log_ratio, log(target_acc_rate) )
        end
    end
    
    return (logW, log_alpha, z)
end


function mala_one_transition(logp::Function, logq::Function, logγ::Function, ∇logγ::Function, 
                                T::Vector{Float64}, id::Int, z_current::Vector{Float64}, 
                                logW::Float64, log_alpha::Float64, 
                                η, ϕ::Vector{Float64})
     
    #= 
    logp(z) : log posterior density 
    logq(z, ϕ): log VI density 
    logγ(z, β, ϕ): β logp + (1-β) logq_ϕ 
    ∇logγ(z, β, ϕ): β ∇logp + (1-β) ∇logq_ϕ 
    T::Vector : Tempering sched
    id::Int : transition id 
    z_current: curent position 
    logW::Float : elbo
    log_alpha::Float : part of the output from mala_elbo, used for grad estimate
    η::Float :langevin stepseize 
    ϕ:Vector :VI param
    =# 
    
    z_old = copy(z_current) 
    # sample Gaussia noise
    u = randn(size(z_current, 1))
    # one langevin step 
    z_transition = z_current .+ η .* ∇logγ(z_current, T[id], ϕ) .+ sqrt.(2.0η) .* u
    # MALA update (whether accept z_transition) and compute log_alpha_update
    logt = log_MH_lang_ratio(logγ, ∇logγ, z_old, z_transition, T, id, ϕ, η)
    z_current, log_alpha_update = accept_reject_step(logt, z_old, z_transition)       

    # update log_acceptance function 
    log_alpha += log_alpha_update
    # update log weight  
    if id === 1
        logW += T[id] * (logp(z_old) - logq(z_old, ϕ))
    else
        logW += (T[id] - T[id - 1]) * (logp(z_old) - logq(z_old, ϕ))
    end
    return z_current, logW, log_alpha, logt
end 