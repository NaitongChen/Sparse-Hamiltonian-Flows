using Random, Statistics, Printf, LinearAlgebra, InvertedIndices
using Zygote.LogExpFunctions: logistic
import Zygote.LogExpFunctions: logistic, logit
using Zygote: @ignore, Buffer
using Base.Threads
# include("train.jl")

############
#### array operation without mutation issue
############

function flip(xs::Vector{Float64})
    buf = Buffer(xs, length(xs))
    for i = 1:length(xs)
        buf[i] = xs[end+1-i]
    end
    return copy(buf)
end

###################
###transforming annealing_schedule (used for reparameterization of tempering sched for optimization)
#####################

# linear:
function T_linear(K::Int)
    # K = number of trnasitions steps
    return  @ignore (Array(range(0.0, 1.0, length = K +1))[2:end])
end


# quadratic tempering
function T_quad(sqrt_b::Float64, K::Int)
    T = @. 1.0 /((1.0 - 1.0/sqrt_b)* ([1: K ;]/K)^2.0 +  1.0/ sqrt_b)
    return T
end

# sigmoidal:
function T_sigmoid(log_δ::Float64, K::Int)
    β1 = logistic(-exp(log_δ))
    βK = logistic(exp(log_δ))
    T_lin = T_linear(K)
    betas_unnormed = @. logistic( exp(log_δ) *(2.0*T_lin - 1.0 ))
    T_sig = @. (betas_unnormed - β1) / (βK - β1)
    return T_sig
end

# all_learnable:
function T_all(T_logit::Vector{Float64})
    return flip(cumprod(flip(vcat( logistic.(T_logit), 1.0))))
end

T_ratio(Temp_sched::Vector{Float64}) = @. sqrt(Temp_sched[1:end-1] / Temp_sched[2:end])


################
## stepsize adaptation and MH_ratio computation
################

# adapt stepsize of transitions based on target_acc_rate
function adapt_stepsize!(step_sizes, variance_sensitive_step, current_accept_rate, target_acc_rate;
                            decrease_rate = 0.98, increase_rate = 1.02, min_stepsize = 1e-3, max_stepsize = 2.0, kwargs...)


    if !variance_sensitive_step
        # using constant stepsize across all dimension
        # scale  step size based on AR
        step_sizes *= (current_accept_rate < target_acc_rate) ? decrease_rate : increase_rate

        # check in the range of min and max, otherwise clamp it
        step_sizes = clamp(step_sizes, min_stepsize, max_stepsize)

    else
        # using vector stepsize  (adam like update)
        # update stepsize in innerloop (stepsize in each transition is a vector)
        step_sizes .= 0.9*step_sizes + 0.1*η0 ./ (gradient_std .+ 1.0)  # gradient_std ∈ kwargs
        # adapt η0 based on ar
        if current_accept_rate[current_tran_id] < target_acc_rate
            η0 *= 0.99
        else
            η0 *= 1.02
        end
    end
end


# log Metroplis-Hasting ratio for MALA (eq 21)
function log_MH_lang_ratio(logγ::Function, ∇logγ::Function, z::Vector{Float64}, z_new::Vector{Float64},
                             T::Vector{Float64}, k::Int, ϕ::Vector{Float64}, η)
    return min(0.0 , logγ(z_new, T[k], ϕ)- logγ(z, T[k] ,ϕ) + sum((z .- z_new .- η .* ∇logγ(z_new, T[k],ϕ)).^2.0 ) - sum((z_new .- z .- η .* ∇logγ(z, T[k], ϕ)).^2.0))
end


# log MH ratio for Hamiltonian annealing
function log_MH_hmc_ratio(logγ::Function, z::Vector{Float64}, z_new::Vector{Float64}, p::Vector{Float64}, p_new::Vector{Float64},
                        T::Vector{Float64}, k::Int, ϕ::Vector{Float64}, Σ::Vector{Float64})
    #=
    logγ(z, β, ϕ): target distribution of current HMC transiiton
    z, z_new: old posiiton and updated position
    p, p_new : old momentum and updated momentum
    T: tempering sched
    ϕ: VI parameter
    k: transition id proposed
    Σ: cov for kenetic distribution (diagonal)
    =#
    return min(0.0 , logγ(z_new, T[k], ϕ)- logγ(z, T[k] ,ϕ) - 0.5*sum( p_new.^2.0 ./ Σ)  +  0.5 * sum(p .^2.0 ./ Σ) )
end

# MALA accept or rejection step
function accept_reject_step(logt::Float64, z_current::Vector{Float64}, z_new::Vector{Float64})
    # logt: current logMH_ratio
    log1t = log1p(-expm1(logt) + 1.0 + 1e-10) # log(1- ratio )
    log_probs = log(rand(Float64, ))
    # decision variable true/false
    a = @ignore log_probs ≤ logt
    # if accept logalpha = logt ; else logalpha = log(1-t)
    current_log_alpha = a ? logt : log1t

    # one langevin step (untrack gradient )
    z_update = a ? z_new : z_current

    return z_update, current_log_alpha
end
