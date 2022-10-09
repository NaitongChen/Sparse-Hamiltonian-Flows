using LinearAlgebra, Distributions, Random, Plots, JLD, StatsBase, StatsPlots, Interpolations, Flux, Measures
include("model.jl")
include("../../util/plotting_util.jl")

########################
# legends
########################
names = ["HIS-Full" "HIS-Coreset" "UHA-Full" "UHA-Coreset" "SHF" "Laplace"]
colours = [palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4], palette(:Paired_8)[8]]

########################
# load data
########################
dat_size = Int(iter/save_freq) + 1

time_his = zeros(n_run, dat_size)
elbo_his_full = zeros(n_run, dat_size)
kl_his_full = zeros(n_run, dat_size)
err_mean_his_full = zeros(n_run, dat_size)
err_cov_his_full = zeros(n_run, dat_size)
err_logs_his_full = zeros(n_run, dat_size)
ed_his_full = zeros(n_run, dat_size)
ksd_his_full = zeros(n_run, dat_size)
elbo_his_mini = zeros(n_run, dat_size)
kl_his_mini = zeros(n_run, dat_size)
err_mean_his_mini = zeros(n_run, dat_size)
err_cov_his_mini = zeros(n_run, dat_size)
err_logs_his_mini = zeros(n_run, dat_size)
ed_his_mini = zeros(n_run, dat_size)
ksd_his_mini = zeros(n_run, dat_size)

time_uha = zeros(n_run, dat_size)
elbo_uha_full = zeros(n_run, dat_size)
kl_uha_full = zeros(n_run, dat_size)
err_mean_uha_full = zeros(n_run, dat_size)
err_cov_uha_full = zeros(n_run, dat_size)
err_logs_uha_full = zeros(n_run, dat_size)
ed_uha_full = zeros(n_run, dat_size)
ksd_uha_full = zeros(n_run, dat_size)
elbo_uha_mini = zeros(n_run, dat_size)
kl_uha_mini = zeros(n_run, dat_size)
err_mean_uha_mini = zeros(n_run, dat_size)
err_cov_uha_mini = zeros(n_run, dat_size)
err_logs_uha_mini = zeros(n_run, dat_size)
ed_uha_mini = zeros(n_run, dat_size)
ksd_uha_mini = zeros(n_run, dat_size)

time_sp = zeros(n_run, dat_size)
elbo_sp = zeros(n_run, dat_size)
kl_sp = zeros(n_run, dat_size)
err_mean_sp = zeros(n_run, dat_size)
err_cov_sp = zeros(n_run, dat_size)
err_logs_sp = zeros(n_run, dat_size)
ed_sp = zeros(n_run, dat_size)
ksd_sp = zeros(n_run, dat_size)

kl_laplace = zeros(n_run, dat_size)
err_mean_laplace = zeros(n_run, dat_size)
err_cov_laplace = zeros(n_run, dat_size)
err_logs_laplace = zeros(n_run, dat_size)
ed_laplace = zeros(n_run, dat_size)
ksd_laplace = zeros(n_run, dat_size)

for i in 1:n_run
    dir_his = joinpath("results/", string("metric_his_$i", ".jld"))
    dir_uha = joinpath("results/", string("metric_uha_$i", ".jld"))
    dir_sp = joinpath("results/", string("new_metric_sp_$i", ".jld"))

    time_his_full = load(dir_his, "time_his")'

    time_his[i,:] = time_his_full[[1:save_freq:iter+1 ;]]
    elbo_his_full[i,:] = load(dir_his, "elbo_his_full")
    kl_his_full[i,:] = load(dir_his, "kl_his_full")
    err_mean_his_full[i,:] = load(dir_his, "err_mean_his_full")
    err_cov_his_full[i,:] = load(dir_his, "err_cov_his_full")
    err_logs_his_full[i,:] = load(dir_his, "err_logs_full")
    ed_his_full[i,:] = load(dir_his, "EDs_full")
    ksd_his_full[i,:] = load(dir_his, "KSDs_full")
    elbo_his_mini[i,:] = load(dir_his, "elbo_his_mini")
    kl_his_mini[i,:] = load(dir_his, "kl_his_mini")
    err_mean_his_mini[i,:] = load(dir_his, "err_mean_his_mini")
    err_cov_his_mini[i,:] = load(dir_his, "err_cov_his_mini")
    err_logs_his_mini[i,:] = load(dir_his, "err_logs_mini")
    ed_his_mini[i,:] = load(dir_his, "EDs_mini")
    ksd_his_mini[i,:] = load(dir_his, "KSDs_mini")

    time_uha_full = load(dir_uha, "time_uha")

    time_uha[i,:] = time_uha_full[[1:save_freq:iter+1 ;]]
    elbo_uha_full[i,:] = load(dir_uha, "elbo_uha_full")
    kl_uha_full[i,:] = load(dir_uha, "kl_uha_full")
    err_mean_uha_full[i,:] = load(dir_uha, "err_mean_uha_full")
    err_cov_uha_full[i,:] = load(dir_uha, "err_cov_uha_full")
    err_logs_uha_full[i,:] = load(dir_uha, "err_logs_full")
    ed_uha_full[i,:] = load(dir_uha, "EDs_full")
    ksd_uha_full[i,:] = load(dir_uha, "KSDs_full")
    elbo_uha_mini[i,:] = load(dir_uha, "elbo_uha_mini")
    kl_uha_mini[i,:] = load(dir_uha, "kl_uha_mini")
    err_mean_uha_mini[i,:] = load(dir_uha, "err_mean_uha_mini")
    err_cov_uha_mini[i,:] = load(dir_uha, "err_cov_uha_mini")
    err_logs_uha_mini[i,:] = load(dir_uha, "err_logs_mini")
    ed_uha_mini[i,:] = load(dir_uha, "EDs_mini")
    ksd_uha_mini[i,:] = load(dir_uha, "KSDs_mini")

    time_sp_full = load(dir_sp, "time_hist")

    time_sp[i,:] = time_sp_full[[1:save_freq:iter+1 ;]]
    elbo_sp[i,:] = load(dir_sp, "ELBOs")
    kl_sp[i,:] = load(dir_sp, "KLs")
    err_mean_sp[i,:] = load(dir_sp, "mrels")
    err_cov_sp[i,:] = load(dir_sp, "srels")
    err_logs_sp[i,:] = load(dir_sp, "slogrels")
    ed_sp[i,:] = load(dir_sp, "EDs")
    ksd_sp[i,:] = load(dir_sp, "KSDs")

    kl_laplace[i,:] = load(dir_sp, "KL") * ones(dat_size)
    err_mean_laplace[i,:] = load(dir_sp, "mrel") * ones(dat_size)
    err_cov_laplace[i,:] = load(dir_sp, "srel") * ones(dat_size)
    err_logs_laplace[i,:] = load(dir_sp, "slogrel") * ones(dat_size)
    ed_laplace[i,:] = load(dir_sp, "ED") * ones(dat_size)
    ksd_laplace[i,:] = load(dir_sp, "KSD_ind") * ones(dat_size)
end

########################
# ELBO iter
########################
iters = [1:save_freq:iter+1 ;]
plot(iters, vec(median(elbo_his_full, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(elbo_his_full))
plot!(iters, vec(median(elbo_his_mini, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(elbo_his_mini))
plot!(iters, vec(median(elbo_uha_full, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(elbo_uha_full))
plot!(iters, vec(median(elbo_uha_mini, dims=1)), label = names[4], color = colours[4], ribbon = get_percentiles(elbo_uha_mini))
plot!(iters, vec(median(elbo_sp, dims=1)), label = names[5], color = colours[5], ribbon = get_percentiles(elbo_sp), legend = :bottomright, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legendfontsize=15)
xlabel!("iteration")
ylabel!("ELBO")
filepath = string("plots/vae/metric_vs_iter/iter_vs_elbo.png")
savefig(filepath)

########################
# KL iter
########################
plot(iters, vec(median(kl_his_full, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(kl_his_full))
plot!(iters, vec(median(kl_his_mini, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(kl_his_mini))
plot!(iters, vec(median(kl_uha_full, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(kl_uha_full))
plot!(iters, vec(median(kl_uha_mini, dims=1)), label = names[4], color = colours[4], ribbon = get_percentiles(kl_uha_mini))
plot!(iters, vec(median(kl_sp, dims=1)), label = names[5], color = colours[5], ribbon = get_percentiles(kl_sp), yscale = :log10, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("iteration")
ylabel!("KL")
filepath = string("plots/vae/metric_vs_iter/iter_vs_kl.png")
savefig(filepath)

###############################
# relative error on mean iter
###############################
plot(iters, vec(median(err_mean_his_full, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(err_mean_his_full))
plot!(iters, vec(median(err_mean_his_mini, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(err_mean_his_mini))
plot!(iters, vec(median(err_mean_uha_full, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(err_mean_uha_full))
plot!(iters, vec(median(err_mean_uha_mini, dims=1)), label = names[4], color = colours[4], ribbon = get_percentiles(err_mean_uha_mini))
plot!(iters, vec(median(err_mean_sp, dims=1)), label = names[5], color = colours[5], ribbon = get_percentiles(err_mean_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("iteration")
ylabel!("rel mean err")
filepath = string("plots/vae/metric_vs_iter/iter_vs_err_mean.png")
savefig(filepath)

####################################
# relative error on covariance iter
####################################
plot(iters, vec(median(err_cov_his_full, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(err_cov_his_full))
plot!(iters, vec(median(err_cov_his_mini, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(err_cov_his_mini))
plot!(iters, vec(median(err_cov_uha_full, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(err_cov_uha_full))
plot!(iters, vec(median(err_cov_uha_mini, dims=1)), label = names[4], color = colours[4], ribbon = get_percentiles(err_cov_uha_mini))
plot!(iters, vec(median(err_cov_sp, dims=1)), label = names[5], color = colours[5], ribbon = get_percentiles(err_cov_sp), yscale=:log10, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("iteration")
ylabel!("rel cov err")
filepath = string("plots/vae/metric_vs_iter/iter_vs_err_cov.png")
savefig(filepath)

#############################################
# relative error on diagonal covariance iter
#############################################
plot(iters, vec(median(err_logs_his_full, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(err_logs_his_full))
plot!(iters, vec(median(err_logs_his_mini, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(err_logs_his_mini))
plot!(iters, vec(median(err_logs_uha_full, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(err_logs_uha_full))
plot!(iters, vec(median(err_logs_uha_mini, dims=1)), label = names[4], color = colours[4], ribbon = get_percentiles(err_logs_uha_mini))
plot!(iters, vec(median(err_logs_sp, dims=1)), label = names[5], color = colours[5], ribbon = get_percentiles(err_logs_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("iteration")
ylabel!("rel log diag cov err")
filepath = string("plots/vae/metric_vs_iter/iter_vs_err_logs.png")
savefig(filepath)

########################
# ED iter
########################
plot(iters, vec(median(ed_his_full, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(ed_his_full))
plot!(iters, vec(median(ed_his_mini, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(ed_his_mini))
plot!(iters, vec(median(ed_uha_full, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(ed_uha_full))
plot!(iters, vec(median(ed_uha_mini, dims=1)), label = names[4], color = colours[4], ribbon = get_percentiles(ed_uha_mini))
plot!(iters, vec(median(ed_sp, dims=1)), label = names[5], color = colours[5], ribbon = get_percentiles(ed_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("iteration")
ylabel!("energy distance")
filepath = string("plots/vae/metric_vs_iter/iter_vs_ed.png")
savefig(filepath)

########################
# KSD iter
########################
plot(iters, vec(median(ksd_his_full, dims=1)), label = names[1], color = colours[1], ribbon = get_percentiles(ksd_his_full))
plot!(iters, vec(median(ksd_his_mini, dims=1)), label = names[2], color = colours[2], ribbon = get_percentiles(ksd_his_mini))
plot!(iters, vec(median(ksd_uha_full, dims=1)), label = names[3], color = colours[3], ribbon = get_percentiles(ksd_uha_full))
plot!(iters, vec(median(ksd_uha_mini, dims=1)), label = names[4], color = colours[4], ribbon = get_percentiles(ksd_uha_mini))
plot!(iters, vec(median(ksd_sp, dims=1)), label = names[5], color = colours[5], ribbon = get_percentiles(ksd_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("iteration")
ylabel!("KSD")
filepath = string("plots/vae/metric_vs_iter/iter_vs_ksd.png")
savefig(filepath)

########################
# ELBO time
########################
plot(time_range(time_his), time_median(elbo_his_mini, time_his), label = names[2], color = colours[2], ribbon = time_percentiles(elbo_his_mini, time_his))
plot!(time_range(time_uha), time_median(elbo_uha_mini, time_uha), label = names[4], color = colours[4], ribbon = time_percentiles(elbo_uha_mini, time_uha))
plot!(time_range(time_sp), time_median(elbo_sp, time_sp), label = names[5], color = colours[5], ribbon = time_percentiles(elbo_sp, time_sp), legend = :bottomright, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legendfontsize=15)
xlabel!("time(s)")
ylabel!("ELBO")
filepath = string("plots/vae/metric_vs_time/time_vs_elbo.png")
savefig(filepath)

plot(time_range(time_his), time_median(elbo_his_mini, time_his), label = names[2], color = colours[2], ribbon = time_percentiles(elbo_his_mini, time_his))
plot!(time_range(time_uha), time_median(elbo_uha_mini, time_uha), label = names[4], color = colours[4], ribbon = time_percentiles(elbo_uha_mini, time_uha))
plot!(time_range(time_sp), time_median(elbo_sp, time_sp), label = names[5], color = colours[5], ribbon = time_percentiles(elbo_sp, time_sp), legend = :bottomright, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legendfontsize=15)
xlabel!("time(s)")
ylabel!("ELBO")
filepath = string("plots/vae/metric_vs_time/time_vs_elbo_no_full.png")
savefig(filepath)

########################
# KL time
########################
plot(time_range(time_his), time_median(kl_his_full, time_his), label = names[1], color = colours[1], ribbon = time_percentiles(kl_his_full, time_his))
plot!(time_range(time_his), time_median(kl_his_mini, time_his), label = names[2], color = colours[2], ribbon = time_percentiles(kl_his_mini, time_his))
plot!(time_range(time_uha), time_median(kl_uha_full, time_uha), label = names[3], color = colours[3], ribbon = time_percentiles(kl_uha_full, time_uha))
plot!(time_range(time_uha), time_median(kl_uha_mini, time_uha), label = names[4], color = colours[4], ribbon = time_percentiles(kl_uha_mini, time_uha))
plot!(time_range(time_sp), time_median(kl_sp, time_sp), label = names[5], color = colours[5], ribbon = time_percentiles(kl_sp, time_sp), yscale=:log10, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("time(s)")
ylabel!("KL")
filepath = string("plots/vae/metric_vs_time/time_vs_kl.png")
savefig(filepath)

###############################
# relative error on mean time
###############################
plot(time_range(time_his), time_median(err_mean_his_full, time_his), label = names[1], color = colours[1], ribbon = time_percentiles(err_mean_his_full, time_his))
plot!(time_range(time_his), time_median(err_mean_his_mini, time_his), label = names[2], color = colours[2], ribbon = time_percentiles(err_mean_his_mini, time_his))
plot!(time_range(time_uha), time_median(err_mean_uha_full, time_uha), label = names[3], color = colours[3], ribbon = time_percentiles(err_mean_uha_full, time_uha))
plot!(time_range(time_uha), time_median(err_mean_uha_mini, time_uha), label = names[4], color = colours[4], ribbon = time_percentiles(err_mean_uha_mini, time_uha))
plot!(time_range(time_sp), time_median(err_mean_sp, time_sp), label = names[5], color = colours[5], ribbon = time_percentiles(err_mean_sp, time_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("time(s)")
ylabel!("rel mean err")
filepath = string("plots/vae/metric_vs_time/time_vs_err_mean.png")
savefig(filepath)

####################################
# relative error on covariance time
####################################
plot(time_range(time_his), time_median(err_cov_his_full, time_his), label = names[1], color = colours[1], ribbon = time_percentiles(err_cov_his_full, time_his))
plot!(time_range(time_his), time_median(err_cov_his_mini, time_his), label = names[2], color = colours[2], ribbon = time_percentiles(err_cov_his_mini, time_his))
plot!(time_range(time_uha), time_median(err_cov_uha_full, time_uha), label = names[3], color = colours[3], ribbon = time_percentiles(err_cov_uha_full, time_uha))
plot!(time_range(time_uha), time_median(err_cov_uha_mini, time_uha), label = names[4], color = colours[4], ribbon = time_percentiles(err_cov_uha_mini, time_uha))
plot!(time_range(time_sp), time_median(err_cov_sp, time_sp), label = names[5], color = colours[5], ribbon = time_percentiles(err_cov_sp, time_sp), yscale=:log10, guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("time(s)")
ylabel!("rel cov err")
filepath = string("plots/vae/metric_vs_time/time_vs_err_cov.png")
savefig(filepath)

#############################################
# relative error on diagonal covariance time
#############################################
plot(time_range(time_his), time_median(err_logs_his_full, time_his), label = names[1], color = colours[1], ribbon = time_percentiles(err_logs_his_full, time_his))
plot!(time_range(time_his), time_median(err_logs_his_mini, time_his), label = names[2], color = colours[2], ribbon = time_percentiles(err_logs_his_mini, time_his))
plot!(time_range(time_uha), time_median(err_logs_uha_full, time_uha), label = names[3], color = colours[3], ribbon = time_percentiles(err_logs_uha_full, time_uha))
plot!(time_range(time_uha), time_median(err_logs_uha_mini, time_uha), label = names[4], color = colours[4], ribbon = time_percentiles(err_logs_uha_mini, time_uha))
plot!(time_range(time_sp), time_median(err_logs_sp, time_sp), label = names[5], color = colours[5], ribbon = time_percentiles(err_logs_sp, time_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("time(s)")
ylabel!("rel log diag cov err")
filepath = string("plots/vae/metric_vs_time/time_vs_err_logs.png")
savefig(filepath)

########################
# ED time
########################
plot(time_range(time_his), time_median(ed_his_full, time_his), label = names[1], color = colours[1], ribbon = time_percentiles(ed_his_full, time_his))
plot!(time_range(time_his), time_median(ed_his_mini, time_his), label = names[2], color = colours[2], ribbon = time_percentiles(ed_his_mini, time_his))
plot!(time_range(time_uha), time_median(ed_uha_full, time_uha), label = names[3], color = colours[3], ribbon = time_percentiles(ed_uha_full, time_uha))
plot!(time_range(time_uha), time_median(ed_uha_mini, time_uha), label = names[4], color = colours[4], ribbon = time_percentiles(ed_uha_mini, time_uha))
plot!(time_range(time_uha), time_median(ed_laplace, time_uha), label = names[6], color = colours[6], ribbon = time_percentiles(ed_laplace, time_uha))
plot!(time_range(time_sp), time_median(ed_sp, time_sp), label = names[5], color = colours[5], ribbon = time_percentiles(ed_sp, time_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("time(s)")
ylabel!("energy distance")
filepath = string("plots/vae/metric_vs_time/time_vs_ed.png")
savefig(filepath)

########################
# KSD time
########################
plot(time_range(time_his), time_median(ksd_his_full, time_his), label = names[1], color = colours[1], ribbon = time_percentiles(ksd_his_full, time_his))
plot!(time_range(time_his), time_median(ksd_his_mini, time_his), label = names[2], color = colours[2], ribbon = time_percentiles(ksd_his_mini, time_his))
plot!(time_range(time_uha), time_median(ksd_uha_full, time_uha), label = names[3], color = colours[3], ribbon = time_percentiles(ksd_uha_full, time_uha))
plot!(time_range(time_uha), time_median(ksd_uha_mini, time_uha), label = names[4], color = colours[4], ribbon = time_percentiles(ksd_uha_mini, time_uha))
plot!(time_range(time_uha), time_median(ksd_laplace, time_uha), label = names[6], color = colours[6], ribbon = time_percentiles(ksd_laplace, time_uha))
plot!(time_range(time_sp), time_median(ksd_sp, time_sp), label = names[5], color = colours[5], ribbon = time_percentiles(ksd_sp, time_sp), guidefontsize=20, tickfontsize=15, margin=5mm, formatter=:plain, legend=false)
xlabel!("time(s)")
ylabel!("KDS")
filepath = string("plots/vae/metric_vs_time/time_vs_ksd.png")
savefig(filepath)