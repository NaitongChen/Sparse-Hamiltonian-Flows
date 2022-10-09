using LinearAlgebra, Distributions, Random, Plots, JLD, StatsBase, StatsPlots, Interpolations, Flux, Measures
include("model.jl")
include("../../util/plotting_util.jl")

########################
# legends
########################
names = ["UNI" "Hilbert-OMP" "SHF"]
colours = [palette(:Paired_8)[2], palette(:Paired_10)[10], palette(:Paired_8)[4]]

########################
# load data
########################
dat_size = length(Ms)

KL_unif = zeros(n_run, dat_size)
KL_omp = zeros(n_run, dat_size)
KL_sp = zeros(n_run, dat_size)
mrel_unif = zeros(n_run, dat_size)
mrel_omp = zeros(n_run, dat_size)
mrel_sp = zeros(n_run, dat_size)
srel_unif = zeros(n_run, dat_size)
srel_omp = zeros(n_run, dat_size)
srel_sp = zeros(n_run, dat_size)
slogrel_unif = zeros(n_run, dat_size)
slogrel_omp = zeros(n_run, dat_size)
slogrel_sp = zeros(n_run, dat_size)
ED_unif = zeros(n_run, dat_size)
ED_omp = zeros(n_run, dat_size)
ED_sp = zeros(n_run, dat_size)
KSD_unif = zeros(n_run, dat_size)
KSD_omp = zeros(n_run, dat_size)
KSD_sp = zeros(n_run, dat_size)

for i in 1:n_run
    dir = joinpath("results/", string("coreset_metric_$i", ".jld"))
    KL_unif[i,:] = load(dir, "KL_unif")
    KL_omp[i,:] = load(dir, "KL_omp")
    KL_sp[i,:] = load(dir, "KL_sp")
    mrel_unif[i,:] = load(dir, "mrel_unif")
    mrel_omp[i,:] = load(dir, "mrel_omp")
    mrel_sp[i,:] = load(dir, "mrel_sp")
    srel_unif[i,:] = load(dir, "srel_unif")
    srel_omp[i,:] = load(dir, "srel_omp")
    srel_sp[i,:] = load(dir, "srel_sp")
    slogrel_unif[i,:] = load(dir, "slogrel_unif")
    slogrel_omp[i,:] = load(dir, "slogrel_omp")
    slogrel_sp[i,:] = load(dir, "slogrel_sp")
    ED_unif[i,:] = load(dir, "ED_unif")
    ED_omp[i,:] = load(dir, "ED_omp")
    ED_sp[i,:] = load(dir, "ED_sp")
    KSD_unif[i,:] = load(dir, "KSD_unif")
    KSD_omp[i,:] = load(dir, "KSD_omp")
    KSD_sp[i,:] = load(dir, "KSD_sp")
end

########################
# KL
########################
plot(Ms, vec(median(KL_unif, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(KL_unif))
plot!(Ms, vec(median(KL_omp, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(KL_omp))
plot!(Ms, vec(median(KL_sp, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(KL_sp), yscale = :log10, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legendfontsize=15, legend=(0.6,0.45))
ylabel!("KL")
xlabel!("coreset size")
filepath = string("plots/coreset/kl.png")
savefig(filepath)

########################
# relative mean error
########################
plot(Ms, vec(median(mrel_unif, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(mrel_unif))
plot!(Ms, vec(median(mrel_omp, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(mrel_omp))
plot!(Ms, vec(median(mrel_sp, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(mrel_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false, yscale=:log10, legendfontsize=15)
ylabel!("rel mean err")
xlabel!("coreset size")
filepath = string("plots/coreset/err_mean.png")
savefig(filepath)

############################
# relative covariance error
############################
plot(Ms, vec(median(srel_unif, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(srel_unif))
plot!(Ms, vec(median(srel_omp, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(srel_omp))
plot!(Ms, vec(median(srel_sp, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(srel_sp), legend = false, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain)
ylabel!("rel cov err")
xlabel!("coreset size")
filepath = string("plots/coreset/err_cov.png")
savefig(filepath)

############################################
# relative logged diagonal covariance error
############################################
plot(Ms, vec(median(slogrel_unif, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(slogrel_unif))
plot!(Ms, vec(median(slogrel_omp, dims=1)), label = names[2], color = colours[2],  ribbon = get_percentiles(slogrel_omp))
plot!(Ms, vec(median(slogrel_sp, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(slogrel_sp), legend = (0.5, 0.7), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legendfontsize=15)
ylabel!("rel log diag cov err")
xlabel!("coreset size")
filepath = string("plots/coreset/err_logs.png")
savefig(filepath)

############################
# energy distance
############################
plot(Ms, vec(median(ED_unif, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(ED_unif))
plot!(Ms, vec(median(ED_omp, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(ED_omp))
plot!(Ms, vec(median(ED_sp, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(ED_sp), legend = false, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain)
ylabel!("energy distance")
xlabel!("coreset size")
filepath = string("plots/coreset/ed.png")
savefig(filepath)

############################
# KSD
############################
plot(Ms, vec(median(KSD_unif, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(KSD_unif))
plot!(Ms, vec(median(KSD_omp, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(KSD_omp))
plot!(Ms, vec(median(KSD_sp, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(KSD_sp), legend = false, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, yscale=:log10)
ylabel!("KSD")
xlabel!("coreset size")
filepath = string("plots/coreset/ksd.png")
savefig(filepath)