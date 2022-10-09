using LinearAlgebra

### just for timing: many hacky things inside

function fflow_cond(z, p, K, sub_xs, lf_n)
    d = size(z, 1)
    ∇U = (zz, ww) -> ∇potential_by_hand(sub_xs, zz, ww)
    for n in 1:lf_n
        oneleapfrog!(z, p, 0.000001*ones(d), ∇U, ones(size(sub_xs, 1)))
        if n % K == 0
            Σ_inv_sqrt = Matrix(I(d))
            μp = zeros(d)
            Σ_mean = Matrix(I(d))
            μz = zeros(d)
            p .= Σ_inv_sqrt * p .- (μp .+ Σ_mean * (z .- μz))
        end
    end
    return z, p
end

function oneleapfrog!(z, p, ϵ_unc, ∇U, w_unc)
    ϵ = exp.(ϵ_unc)
    w = exp.(w_unc)
    p .+= (ϵ ./ 2) .* ∇U(z, w)
    z .+= ϵ .* p
    p .+= (ϵ ./ 2) .* ∇U(z, w)
end

function fflow_marg(z, p, K, sub_xs, lf_n)
    d = size(z, 1)
    ∇U = (zz, ww) -> ∇potential_by_hand(sub_xs, zz, ww)
    for n in 1:lf_n
        oneleapfrog!(z, p, 0.000001*ones(d), ∇U, ones(size(sub_xs, 1)))
        if n % K == 0
            Σ_inv_sqrt = Matrix(I(d))
            μp = zeros(d)
            p .= Σ_inv_sqrt * (p .- μp)
        end
    end
    return z, p
end


function sp_flow_cond_density(z0, p0, K, sub_xs, lf_n)
    d = size(z0,1)
    z, ρ = fflow_cond(z0, p0, K, sub_xs, lf_n)
    log_det = 1.0
    return logq(z) -  0.5*(ρ' * ρ)  - d /2.0 *  log(2.0*pi) + log_det
end

function sp_flow_marg_density(z0, p0, K, sub_xs, lf_n)
    d = size(z0,1)
    z, ρ = fflow_marg(z0, p0, K, sub_xs, lf_n)
    log_det = 1.0
    return logq(z) -  0.5*(ρ' * ρ)  - d /2.0 *  log(2.0*pi) + log_det
end
