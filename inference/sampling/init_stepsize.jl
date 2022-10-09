include("leapfrog.jl")


# initialize stepsize
function find_reasonable_ϵ(θ, L, ∇L)
    @info "choosing initial ϵ"

    ϵ, r = 1.0, randn(size(θ, 1))
    θ1, r1 = leapfrog(θ, r, ϵ, ∇L)

    # This trick prevents the log-joint or its graident from being infinte
    # Ref: code start from Line 111 in https://github.com/mfouesneau/NUTS/blob/master/nuts.py
    while isinf(L(θ1)) || any(isinf.(∇L(θ1)))
        ϵ = ϵ * 0.5
        θ1, r1 = leapfrog(θ, r, ϵ, ∇L)
    end

    a = 2.0 * ( L(θ1) - 0.5 * dot(r1, r1) - L(θ) + 0.5 * dot(r, r) > log(0.5)) - 1.0

    while (a * ( L(θ1) - 0.5 * dot(r1, r1) - L(θ) + 0.5 * dot(r, r) ) > -a * log(2.0))
        ϵ = exp( a*log(2) + log(ϵ))
        θ1, r1 = leapfrog(θ, r, ϵ, ∇L)
    end
    return ϵ
end
