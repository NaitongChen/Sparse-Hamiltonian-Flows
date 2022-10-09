using LinearAlgebra, Random, Statistics, Optim, ForwardDiff, Distributions

function sample_with_laplace_approx(logp, d, J)
    init_z = randn(d)
    obj = z -> -logp(z)
    opt = optimize(obj, init_z)
    μ = Optim.minimizer(opt)

    hess = ForwardDiff.hessian(obj, μ)
    Σ = hess \ I(d)
    Σ .= 0.5 .* (Σ + transpose(Σ)) # ensures symmetry

    if size(Σ,1) == 1
        return rand(MvNormal(μ, Σ .* I(d)), J)'
    else
        return rand(MvNormal(μ, Σ), J)'
    end
end

function laplace_approx(logp, d, c_prior, m)
    init_z = sqrt(c_prior) * randn(d) .+ m
    obj = z -> -logp(z)
    opt = optimize(obj, init_z)
    μ = Optim.minimizer(opt)

    hess = ForwardDiff.hessian(obj, μ)
    Σ = hess \ I(d)
    Σ .= 0.5 .* (Σ + transpose(Σ)) # ensures symmetry

    for i in 1:size(Σ,1)
        if Σ[i,i] < 0
            Σ[i,i] = 1e-20
        end
    end

    if minimum(eigvals(Σ)) < 0
        Σ = Σ + (-minimum(eigvals(Σ)) + 1e-3) .* Matrix(I(d))
    end

    return μ, Σ
end