include("hvi.jl")
include("sampler.jl")

# functions for evaluating his variational distrribution density 
function his_density_eval(z::Vector{Float64}, ρ::Vector{Float64}, ∇logp::Function, T_alpha::Vector{Float64}, ϵ::Vector{Float64},
                            K::Int, d::Int, n_lfrg::Int) 
    β0_sqrt = prod(T_alpha) + 1e-8
    ρ ./= β0_sqrt 
    
    for id = 1:K
        # no refresh but tempering momentum
        his_one_transition!(∇logp, T_alpha, id, ρ, z, ϵ, n_lfrg)
    end

    logβ0 = 2.0*log(β0_sqrt)
    return logq(z) -  0.5 *exp(logβ0) * (ρ' * ρ)  - d /2.0 * (logβ0 + log(2.0*pi))
end


# functions for evaluating uha variational distrribution density 
function uha_density_eval(z::Vector{Float64}, ρ::Vector{Float64}, log_q0::Function, ∇logγ::Function, Temp_sched::Vector{Float64}, ϵ::Vector{Float64}, η::Vector{Float64}, 
                            K::Int, d::Int, n_lfrg::Int)
    den_value = -0.5*(ρ' * ρ) - 0.5*log(2.0*pi)
    for id = 1:K 
        uha_one_transition_for_sample!(∇logγ, Temp_sched, id, z, ρ, ϵ, η, n_lfrg)
        den_value += -0.5*(ρ' * ρ)/(1.0 - η[1]^2.0) - 0.5*log(2.0*pi) - 0.5*log(1.0 - η[1]^2.0)
    end
    den_value += log_q0(z)
    return den_value
end

