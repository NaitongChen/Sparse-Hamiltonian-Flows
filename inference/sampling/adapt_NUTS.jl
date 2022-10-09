include("leapfrog.jl")
include("init_stepsize.jl")
using LinearAlgebra, ProgressMeter, Random

################
# NUTS with dual averaging for stepsize tuning (Alg 6) :: TODO not working
################

function build_tree(θ, r, logu, v, j, ϵ, θ0, r0, L, ∇L)
    """
      - θ   : model parameter
      - r   : momentum variable
      - logu: log of slice variable
      - v   : direction ∈ {-1, 1}
      - j   : depth of tree
      - ϵ   : leapfrog step size
      - θ0  : initial model parameter
      - r0  : initial mometum variable
    """
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        θ1, r1 = leapfrog(θ, r, v * ϵ, ∇L)
        # NOTE: this trick prevents the log-joint or its graident from being infinte
        while L(θ1) == -Inf || ∇L(θ1) == -Inf
            ϵ = ϵ * 0.5
            θ1, r1 = leapfrog(θ, r, v * ϵ, ∇L)
        end
        n1 = logu <= L(θ1) - 0.5 * dot(r1, r1)
        s1 = logu < Δmax + L(θ1) - 0.5 * dot(r1, r1)
        return θ1, r1, θ1, r1, θ1, n1, s1, min(1, exp(L(θ1) - L(θ0) - 0.5 * dot(r1, r1) + 0.5 * dot(r0, r0))), 1
    else
        # Recursion - build the left and right subtrees.
        θm, rm, θp, rp, θ1, n1, s1, α1, n1_α = build_tree(θ, r, logu, v, j - 1, ϵ, θ0, r0, L,∇L)
        if s1 == 1
            if v == -1
                θm, rm, _, _, θ2, n2, s2, α2, n2_α = build_tree(θm, rm, logu, v, j - 1, ϵ, θ0, r0, L, ∇L)
            else
                _, _, θp, rp, θ2, n2, s2, α2, n2_α = build_tree(θp, rp, logu, v, j - 1, ϵ, θ0, r0, L, ∇L)
            end
            if log(rand()) < log(n2)  - log(n1 + n2)
                θ1 = θ2
            end
            α1 = α1 + α2
            n1_α = n1_α + n2_α
            s1 = s2 & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
            n1 = n1 + n2
        end
        return θm, rm, θp, rp, θ1, n1, s1, α1, n1_α
    end
end



function NUTS(θ0, δ, L, ∇L, M, Madapt; verbose = true)
    """
      - θ0      : initial model parameter
      - δ       : desirable average accept rate (0.65 as magic number)
      - L       : log posterior density
      - ∇L      : gradient of log posterior density
      - M       : sample number
      - Madapt  : number of samples for step size adaptation
      - verbose : whether to show log
    """
    if verbose
        # progress bar
        prog_bar = ProgressMeter.Progress(M + Madapt, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    end
    θs = Vector{typeof(θ0)}(undef, M +Madapt + 1)

    θs[1], ϵ = θ0, find_reasonable_ϵ(θ0, L, ∇L)
    μ, γ, t_0, κ = log(10) + log(ϵ), 0.05, 10, 0.75
    logϵbar, Hbar = 0.0, 0.0

    if verbose
        @info "[NUTS] start sampling for $M samples with inital ϵ = $ϵ"
    end

    for m = 1:M + Madapt
        r0 = randn(length(θ0))
        logu = log(rand()) + L(θs[m]) - 0.5 * dot(r0, r0) # Note: θ^{m-1} in the paper corresponds to `θs[m]` in the code

        θm, θp, rm, rp, j, θs[m+1], n, s = θs[m], θs[m], r0, r0, 0, θs[m], 1, 1
        α, n_α = NaN, NaN
        while s == 1
            v = rand([-1, 1])
            if v == -1
                θm, rm, _, _, θ1, n1, s1, α, n_α = build_tree(θm, rm, logu, v, j, ϵ, θs[m], r0, L, ∇L)
            else
                _, _, θp, rp, θ1, n1, s1, α, n_α = build_tree(θp, rp, logu, v, j, ϵ, θs[m], r0, L, ∇L)
            end
            if s1 == 1
                if log(rand()) < min(0, log(n1) - log(n))
                    θs[m+1] = θ1
                end
            end
            n = n + n1
            s = s1 & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
            j = j + 1
        end
        if m <= Madapt
            # NOTE: Hbar goes to negative when δ - α / n_α < 0
            Hbar = (1.0 - 1.0 / (m + t_0)) * Hbar + 1.0 / (m + t_0) * (δ - α / n_α)
            logϵ = μ - sqrt(m) / γ * Hbar
            logϵbar= m^(-κ) * logϵ + (1 - m^(-κ)) * logϵbar
            ϵ = exp(logϵ)
            # println(ϵ)
        else
            ϵ = exp(logϵbar)
        # println(ϵ)
        end

        if verbose
            # update progress bar
            ProgressMeter.next!(prog_bar)
        end
    end

    if verbose
        @info "[NUTS] sampling complete with final apated ϵ = $ϵ"
    end

    M = reduce(hcat, θs[Madapt + 2:end])
    # if size(M) is a row matrix, reshape to make it a N×1 matrix
    return size(M, 1) > 1 ? Matrix(M') : reshape(M, size(M, 2), 1)
end






# ###############
# ## example
# ###############
# using Plots

# logp = z -> -0.5* (z' * z)
# ∇logp = z -> -z

# T = NUTS(100*ones(2), 0.65, logp, ∇logp, 15000, 10000)

# x = -20:.1:20
# y = -10:.1:20
# p1 = contour(x, y, (x,y)-> -0.5*(x^2 + y^2), seriescolor = cgrad(:blues), levels=0:-3:-35,
#             legend = :topleft, colorbar = :none, title = "nuts-samples")
# scatter(T[:,1], T[:,2], mark = 3, alpha = 0.6,label = "nuts samples")
