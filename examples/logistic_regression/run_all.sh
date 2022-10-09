#!/bin/bash

echo -e "start HIS"
julia --threads 4 metric_his.jl 1 &
julia --threads 4 metric_his.jl 2 &
julia --threads 4 metric_his.jl 3 &
julia --threads 4 metric_his.jl 4 &
julia --threads 4 metric_his.jl 5 &
wait
echo -e "HIS done"

echo -e "start UHA"
julia --threads 4 metric_uha.jl 1 &
julia --threads 4 metric_uha.jl 2 &
julia --threads 4 metric_uha.jl 3 &
julia --threads 4 metric_uha.jl 4 &
julia --threads 4 metric_uha.jl 5 &
wait
echo -e "UHA done"

echo -e "start SHF"
julia --threads 4 metric_sp.jl 1 &
julia --threads 4 metric_sp.jl 2 &
julia --threads 4 metric_sp.jl 3 &
julia --threads 4 metric_sp.jl 4 &
julia --threads 4 metric_sp.jl 5 &
wait
echo -e "SHF done"

echo -e "start coreset"
julia --threads 4 metric_coreset.jl 1 &
julia --threads 4 metric_coreset.jl 2 &
julia --threads 4 metric_coreset.jl 3 &
julia --threads 4 metric_coreset.jl 4 &
julia --threads 4 metric_coreset.jl 5 &
wait
echo -e "coreset done"

echo -e "timing vae methods"
julia --threads 4 timing_vae.jl 
echo -e "done timing vae methods"