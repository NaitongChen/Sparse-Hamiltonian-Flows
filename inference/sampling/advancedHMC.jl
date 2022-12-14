using AdvancedHMC

function advancedHMC(θ0, δ, L, ∇L, M, Madapt; verbose = true)

    # choose Mass matrix
    d = size(θ0, 1)
    metric = DiagEuclideanMetric(d)
    # define  hamiltonian system 
    # ajoint is a user specified gradient system, returning a tuple (log_post, gradient) 
    ajoint = θ -> (L(θ), ∇L(θ))
    hamiltonian = Hamiltonian(metric, L, ajoint)
    # Define a leapfrog solver, with initial step size chosen heuristically
    init_ϵ = find_good_stepsize(hamiltonian, θ0)
    integrator = Leapfrog(init_ϵ)

    # combined adapatation scheme 
    proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(δ, integrator)) # combined adaptaiton scheme using stan window adaptaiton

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(hamiltonian, proposal, θ0, M+Madapt, adaptor, Madapt; progress=verbose)

    @info "[AdavancedHMC] sampling complete"
    
    # return samples
    M = reduce(hcat, samples[Madapt+1:end])
    # if size(M) is a row matrix, reshape to make it a N×1 matrix
    return size(M, 1) > 1 ? Matrix(M') : reshape(M, size(M, 2), 1)
end

##############
# example 
################
# using Distributions, ForwardDiff
# D = 10
# L(θ) = logpdf(MvNormal(zeros(D), I), θ)
# ∇L(θ) = ForwardDiff.gradient(L, θ)
# MM = advancedHMC(100*ones(D), 0.7, L, ∇L, 10000, 1000)
