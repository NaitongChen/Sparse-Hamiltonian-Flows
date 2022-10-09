include("leapfrog.jl")
include("init_stepsize.jl")


##################3
# slight modificaiton to alg 5 of NUTS paper 
################3

function adapt_hmc(θ0, δ, n_lfrg, L, ∇L, M, Madapt; verbose = true)
    """
    - θ: init pos 
    - δ: target acc ratio   
    - L, ∇L: target 
    - M : nsamples
    - Madapt : num of samples used for adaptation
    """

    if verbose
        # progress bar
        prog_bar = ProgressMeter.Progress(M+Madapt, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    end
    θs = Vector{typeof(θ0)}(undef, M + Madapt + 1)

    θs[1], ϵ = θ0, find_reasonable_ϵ(θ0, L, ∇L)
    μ, γ, t_0, κ = log(10) + log(ϵ), 0.05, 10, 0.75
    logϵbar, Hbar = 0.0, 0.0

    if verbose
        @info "[HMC] start sampling for $M samples with inital ϵ = $ϵ"
    end

    for m in 2:M+1 + Madapt
        r = randn(size(θ0, 1))
        θs[m], θ1, r1 = θs[m-1], θs[m-1], r

        for i = 1:n_lfrg 
            θ1, r1 = leapfrog(θ1, r1, ϵ, ∇L)
        end 

        loga = min(0.0, L(θ1) - L(θs[m - 1]) -0.5*dot(r1, r1) + 0.5 * dot(r, r))
        if log(rand()) < loga 
            θs[m] = θ1
        end

        if m <= Madapt + 1
            # NOTE: Hbar goes to negative when δ - α / n_α < 0
            Hbar = (1.0 - 1.0 / (m + t_0)) * Hbar + 1.0 / (m + t_0) * (δ - exp(loga))
            logϵ = μ - sqrt(m) / γ * Hbar
            logϵbar= m^(-κ) * logϵ + (1 - m^(-κ)) * logϵbar
            ϵ = exp(logϵ)
        else
            ϵ = exp(logϵbar)
        end

        if verbose
            # update progress bar
            ProgressMeter.next!(prog_bar)
        end

    end
    if verbose
        @info "[HMC] sampling complete with final adapt ϵ = $ϵ"
    end
    
    M = reduce(hcat, θs[Madapt+2:end])
    # if size(M) is a row matrix, reshape to make it a N×1 matrix
    return size(M, 1) > 1 ? Matrix(M') : reshape(M, size(M, 2), 1)
end




# ###############
# ## example
# ###############
# using Plots

# logp = z -> -0.5* (z' * z)
# ∇logp = z -> -z

# T = adapt_hmc(100*ones(2), 0.7, 10,  logp, ∇logp, 1000, 500) 

# x = -20:.1:20
# y = -10:.1:20
# p1 = contour(x, y, (x,y)-> -0.5*(x^2 + y^2), seriescolor = cgrad(:blues), levels=0:-3:-35, 
#             legend = :topleft, colorbar = :none, title = "nuts-samples") 
# scatter(T[:,1], T[:,2], mark = 3, alpha = 0.6,label = "nuts samples")