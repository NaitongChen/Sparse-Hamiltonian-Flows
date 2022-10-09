using StatsBase, LinearAlgebra, Interpolations

function get_percentiles(dat; p1=25, p2=75)
    if size(dat,2) == 1
        plow = zeros(n)
        phigh = zeros(n)
    else
        n = size(dat,2)
        median_dat = vec(median(dat, dims=1))

        plow = zeros(n)
        phigh = zeros(n)

        for i in 1:n
            dat_remove_inf = (dat[:,i])[iszero.(isinf.(dat[:,i]))]
            plow[i] = median_dat[i] - percentile(vec(dat_remove_inf), p1)
            phigh[i] = percentile(vec(dat_remove_inf), p2) - median_dat[i]
        end
    end

    return plow, phigh
end

function time_range(time)
    if size(time,2) == 1
        ts = time
    else
        begin_time = maximum(time[:,1])
        end_time = minimum(time[:,end])
        ts = [begin_time:(end_time - begin_time)/(iter / save_freq ):end_time ;]
    end

    return ts
end

function get_interpolated_data(dat, time)
    m, n = size(dat)
    ts = time_range(time)
    interpolated_data = zeros(m,n)

    for i in 1:m
        interp = LinearInterpolation(time[i,:], dat[i,:])
        interpolated_data[i,:] = interp(ts)
    end

    return interpolated_data
end

function time_median(dat, time)
    if size(dat,2) == 1
       return dat
    else
        dat = get_interpolated_data(dat, time)
        return vec(median(dat, dims=1))
    end
end

function time_percentiles(dat, time; p1=25, p2=75)
    if size(dat,2) == 1
        plow = zeros(n)
        phigh = zeros(n)
    else
        dat = get_interpolated_data(dat, time)
        n = size(dat,2)
        median_dat = vec(median(dat, dims=1))

        plow = zeros(n)
        phigh = zeros(n)

        for i in 1:n
            plow[i] = median_dat[i] - percentile(vec(dat[:,i]), p1)
            phigh[i] = percentile(vec(dat[:,i]), p2) - median_dat[i]
        end
    end

    return plow, phigh
end