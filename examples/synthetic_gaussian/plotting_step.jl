using LinearAlgebra, Distributions, Random, Plots, ForwardDiff, Logging, JLD, StatsBase, Measures
include("model.jl")

id = 2022
dir = joinpath("results/", string("step_$id",".jld"))
elbos = load(dir, "elbos")
plot([1:a.lf_n ;], elbos, xlabel="leapfrog number", ylabel="ELBO", legend=false, guidefontsize=20, tickfontsize=15, margin=5mm, color=palette(:Paired_8)[2], formatter=:plain)
for i in 1:number_of_refresh
    plot!([K*i-1, K*i], elbos[K*i-1:K*i], linewidth=2, color=palette(:Paired_8)[6])
end
quiver!([19], [-415000], quiver=([-8], [0]), color="black")
annotate!([33], [-415000], text("quasi-refreshment", 15))
quiver!([18], [-400000], quiver=([-3], [12000]), color="black")
annotate!([33], [-403000], text("Hamiltonian dynamics", 15))

filepath = string("plots/step/step_$id.png")
savefig(filepath)
