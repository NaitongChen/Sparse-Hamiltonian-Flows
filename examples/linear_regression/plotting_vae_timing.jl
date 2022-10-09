using LinearAlgebra, Distributions, Random, Plots, JLD, StatsBase, StatsPlots, Interpolations, Flux, Measures
include("model.jl")

########################
# legends
########################
names = ["HMC" "NUTS" "HIS-Full" "HIS-Coreset" "UHA-Full" "UHA-Coreset" "SHF"]
colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1],
            palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

########################
# load data
########################
dir = joinpath("results/", string("vae_timing", ".jld"))
time_eval_his = load(dir, "time_eval_his")
time_eval_his_core = load(dir, "time_eval_his_core")
time_eval_sp = load(dir, "time_eval_sp")
time_eval_uha = load(dir, "time_eval_uha")
time_eval_uha_core = load(dir, "time_eval_uha_core")
time_sample_sp = load(dir, "time_sample_sp")
time_sample_his = load(dir, "time_sample_his")
time_sample_uha = load(dir, "time_sample_uha")
time_sample_his_mini = load(dir, "time_sample_his_mini")
time_sample_uha_mini = load(dir, "time_sample_uha_mini")
time_sample_nuts = load(dir, "time_sample_nuts")
time_sample_hmc = load(dir, "time_sample_hmc")

########################
# density evaluation
########################
boxplot([names[3]], time_eval_his, label = names[3], color = colours[3])
boxplot!([names[4]], time_eval_his_core, label = names[4], color = colours[3])
boxplot!([names[5]], time_eval_uha, label = names[5], color = colours[5])
boxplot!([names[6]], time_eval_uha_core, label = names[6], color = colours[5])
boxplot!([names[7]], time_eval_sp, label = names[7], color = colours[7], yscale = :log10, legend = false, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, xrotation=20)
ylabel!("density eval time(s)")
filepath = string("plots/vae/timing/density_eval.png")
savefig(filepath)

########################
# time per sample
########################
boxplot([names[1]], time_sample_hmc, label = names[1], color = colours[1])
boxplot!([names[2]], time_sample_nuts, label = names[2], color = colours[2])
boxplot!([names[3]],time_sample_his, label = names[3], color = colours[3])
boxplot!([names[4]],time_sample_his_mini, label = names[4], color = colours[4])
boxplot!([names[5]],time_sample_uha, label = names[5], color = colours[5])
boxplot!([names[6]],time_sample_uha_mini, label = names[6], color = colours[6])
boxplot!([names[7]],time_sample_sp, label = names[7], color = colours[7], yscale = :log10, legend = false, guidefontsize=20, ytickfontsize=15, xtickfontsize=15, margin=5mm, formatter=:plain, xrotation=20)
ylabel!("sample gen time(s)")
filepath = string("plots/vae/timing/time_per_sample.png")
savefig(filepath)
