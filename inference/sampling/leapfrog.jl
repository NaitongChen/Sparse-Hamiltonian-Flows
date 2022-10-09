
function leapfrog(θ, r, ϵ, ∇L)
    """
        can't use mutation
        - θ : model parameter
        - r : momentum variable
        - ϵ : leapfrog step size
    """
    r1 = r .+ ((ϵ/2.0).*∇L(θ))
    θ1 = θ .+ (ϵ*r1)
    r1 = r1 .+ ((ϵ/2.0).*∇L(θ1))
    return θ1, r1
end

# constant that will be used in NUTS
Δmax = 1000 # suggested by the paper in Eq(3)