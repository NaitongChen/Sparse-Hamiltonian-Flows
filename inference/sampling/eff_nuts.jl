using LinearAlgebra
include("leapfrog.jl")


################
### efficient Non-U-turn HMC (alg 3: )
################

function build_tree(θ, r, logu, v, j, ϵ, L, ∇L)
    """
      - θ   : model parameter
      - r   : momentum variable
      - logu: log of slice variable
      - v   : direction ∈ {-1, 1}
      - j   : depth
      - ϵ   : leapfrog step size
    """
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        θ1,r1 = leapfrog(θ, r, v * ϵ, ∇L)
        n1 = logu <= L(θ1) - 0.5 * dot(r1, r1)
        s1 = logu < Δmax + L(θ1) - 0.5 * dot(r1, r1)
        return θ1, r1, θ1, r1, θ1, n1, s1
    else
        # Recursion - build the left and right subtrees.
        θm, rm, θp, rp, θ1, n1, s1 = build_tree(θ, r, logu, v, j - 1, ϵ, L, ∇L)
        if s1 == 1
            if v == -1
                θm, rm, _, _, θ2, n2, s2 = build_tree(θm, rm, logu, v, j - 1, ϵ, L ,∇L)
            else
                _, _, θp, rp, θ2, n2, s2 = build_tree(θp, rp, logu, v, j - 1, ϵ, L, ∇L)
            end
            if log(rand()) < log(n2) - log(n1 + n2)
                θ1 = θ2
            end
            s1 = s2 & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
            n1 = n1 + n2
        end
        return θm, rm, θp, rp, θ1, n1, s1
    end
end


function eff_NUTS(θ0, ϵ, L, ∇L, M)
    """
      - θ0      : initial model parameter
      - ϵ       : leapfrog step size
      - L       : log density of target distribution
      - ∇L      : grad of L 
      - M       : sample number
    """

    θs = Vector{typeof(θ0)}(undef, M + 1)
    θs[1] = θ0

    @info "[eff_NUTS] start sampling for $M samples with ϵ=$ϵ"

    for m = 1:M

        r0 = randn(size(θ0, 1))
        logu = log(rand()) + L(θs[m]) - 0.5 * dot(r0, r0) # Note: θ^{m-1} in the paper corresponds to `θs[m]` in the code

        θm, θp, rm, rp, j, θs[m+1], n, s = θs[m], θs[m], r0, r0, 0.0, θs[m], 1, 1
        while s == 1
            v_j = rand([-1, 1]) 
            if v_j == -1
                θm, rm, _, _, θ1, n1, s1 = build_tree(θm, rm, logu, v_j, j, ϵ, L, ∇L)
            else
                _, _, θp, rp, θ1, n1, s1 = build_tree(θp, rp, logu, v_j, j, ϵ, L, ∇L)
            end

            if s1 == 1
                if log(rand()) < min(0, log(n1))
                    θs[m+1] = θ1
                end
            end
            n = n + n1
            s = s1 & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
            j = j + 1
        end
    end

    @info "[eff_NUTS] sampling complete"
    
    M = reduce(hcat, θs)
    # if size(M) is a row matrix, reshape to make it a N×1 matrix
    return size(M, 1) > 1 ? Matrix(M') : reshape(M, size(M, 2), 1)
end


################
## example
#################
# using Plots

# logp = z -> -0.5* (z' * z)
# ∇logp = z -> -z

# T = eff_NUTS(100*ones(2), 0.1, logp, ∇logp, 1000)

# x = -20:.1:20
# y = -10:.1:20
# p1 = contour(x, y, (x,y)-> -0.5*(x^2 + y^2), seriescolor = cgrad(:blues), levels=0:-3:-35, 
#             legend = :topleft, colorbar = :none, title = "nuts-samples") 
# scatter(T[:,1], T[:,2], mark = 3, alpha = 0.6,label = "nuts samples")