using LinearAlgebra, Random, Statistics, NonNegLeastSquares, ProgressMeter
include("random_projection.jl")

function OMP_coreset(xs, J, logp_ind, logp, d)
    N = size(xs,1)
    @info "random projection"
    proj = random_projection(xs, J, logp_ind, logp, d) # J by N
    overall_liks = vec(sum(proj, dims=2))

    # return ws # N by 1
    build = mm -> OMP(proj, overall_liks, mm)
    return build
end

function error(A, b, w)
    return norm(A * w - b)
end

function OMP(A, b, M)
    m,n = size(A)
    An = zeros(m,n)
    for i in 1:m
        An[i,:] = A[i,:] ./ norm(A[i,:])
    end

    ws = zeros(n)
    inds = zeros(n)

    prog_bar = ProgressMeter.Progress(M, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i in 1:M
        prev_error = error(A,b,ws)
        prev_w = copy(ws)

        # select
        res = b .- (A * ws)
        dots = transpose(An) * res

        if sum(ws) == 0.
            inds[argmax(dots)] = 1
        else
            added = false
            counter = 1
            sorted_args = sortperm(dots, rev = true)
            while !added
                if counter > n
                    @goto escape_label
                end
                fpos = sorted_args[counter]
                pos = dots[fpos]
                nz_idcs = ws .> 0
                fneg = argmax(-dots[nz_idcs])
                neg = -dots[nz_idcs][fneg]

                if pos >= neg
                    if inds[fpos] == 0
                        inds[fpos] = 1
                        added = true
                    end
                else
                    if inds[Array(1:n)[nz_idcs][fneg]] == 0
                        inds[Array(1:n)[nz_idcs][fneg]] = 1
                        added = true
                    end
                end
                counter += 1
            end
        end

        if error(A,b,ws) > prev_error
            ws = prev_w
            println("revert step")
        end

        nz_idcs = inds .> 0
        new_weights = nonneg_lsq(A[:,nz_idcs], b; alg=:nnls)
        ws[nz_idcs] = new_weights
        ProgressMeter.next!(prog_bar)
    end
    @label escape_label
    return ws
end
