################
### mcmc transition kernels
################

# abstract MCMC method
abstract type MCMCkernel end

#=
Meta function for getting samples
Naitong: you can modify it to customize Burnout samples
=#
function getsample(mcmc::MCMCkernel, z, logp, ∇logp, n)
    #=
    mcmc: abstract mcmc method (currently only HMC)
    z: init value
    logp, ∇logp: potential of target
    n: number of samples to get
    =#
    trace = Vector{typeof(z)}(undef, n)
    for b in 1:mcmc.nBurnin
        MCMCupdate!(mcmc, z, logp, ∇logp; save = false)
    end

    @inbounds for i  = 1:n
        trace[i] = MCMCupdate!(mcmc, z, logp, ∇logp; save = true)
        # println(z)
    end
    return reduce(hcat, trace)
end



#=
1. Hamiltonian monte carlo
- fully refresh momentum
- adapt acceptence ratio towards 0.7 during the burn in phase
- nBunrnin: use the first nBurnin of leapfrog to tune the stepsize
=#

mutable struct  HMC <: MCMCkernel
    η::Float64
    nleapfrog::Int
    nBurnin::Int
    ntransitions::Int
    target_acc_rate::Float64
end

# setting default value
HMC() = HMC(0.1, 5, 100, 5, 0.7)

# single transition
function onetransition!(o::HMC, z, logp, ∇logp)
    z_old = copy(z)
    u = randn(eltype(z), size(z, 1))
    u_old = copy(u)
    current_U = -logp(z_old)

    η = o.η
    # make a half step for momentum at begining
    u .+= 0.5 .* η .* ∇logp(z)
    # Alternate full steps for position and momentum
    for i = 1:o.nleapfrog
        z .+= η .* u
        u .+= η .* ∇logp(z)
    end
    # make a half step for momentum at the end
    u .+= 0.5 .* η .* ∇logp(z)
    # negate momentum to make symmetric proposal
    u .*= -1.0

    # evalute potential and kinetic energies
    proposed_U = -logp(z)
    current_K = u_old' * u_old  / 2.0
    proposed_K = u' * u /2.0

    log_acc_rate = current_U - proposed_U + current_K - proposed_K
    # println(log_acc_rate)
    if log(rand(Float64)) >  log_acc_rate
        z, z_old  = z_old, z
    end

    return log_acc_rate
end

# update for HMC
function MCMCupdate!(o::HMC, z, logp, ∇logp; save = false)
    N, r = o.ntransitions, log(o.target_acc_rate)

    for i  = 1:N
        log_acc_rate = onetransition!(o, z, logp, ∇logp)
        # adapt stepsize during burn in
        if i ≤ o.nBurnin
           o.η =  log_acc_rate > r ? 0.998 * o.η : 1.0002*o.η
           o.η = clamp(o.η, 1e-3, 1.0)
        end
    end
    # println(o.η)
    if save
        return copy(z)
    end
end


#=
example
# η, nleapfrog, nBurnin, ntransition, target_acc_rate
hmc_alg = HMC(1f-2, 5, 2, 5, 0.6f0)
T0 = getsample(hmc_alg, z0, logp, ∇logp, 1000)
=#