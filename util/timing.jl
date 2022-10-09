using TickTock, Suppressor, ProgressMeter
include("train.jl")

# repeatedly evaluating a function and average the time
function noob_timing(f::Function, args...; n_run = 1000)
    time_log = zeros(n_run+1)
    count = 0
    prog_bar = ProgressMeter.Progress(n_run+1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @suppress while count < n_run+1
        tick();
        call(f, args...)
        t = tok()
        time_log[count + 1] = t
        count += 1
        ProgressMeter.next!(prog_bar)
    end

    return time_log[2:end]
end
