#!/bin/bash


echo -e "plotting"
julia plotting_step.jl &
julia plotting_vae_timing.jl &
wait 
echo -e "step and vae timing done"
julia plotting_coreset.jl &
wait 
echo -e "coreset plot done"
julia plotting_vae_error_metric.jl &
wait 
echo -e "vae metric done"
