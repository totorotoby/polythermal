ousing Smoothing
using Arpack
include("assemble.jl")
include("sol_tests.jl")



function timestep(H, Pc, Γ, params, t_ops, g_ops, Δt)

    p = params.p
    Tsurf = params.Tsurf
    ϕbase = params.ϕbase
    z = params.z
    inflow = params.inflow
    implicit = params.implicit
    Γ_nodes = EToN(Γ, p)
    Nt = Γ_nodes[end]
    
    #--- new solver ---#

    # do enthalpy either implicitly
    if implicit == true
        #picard!(H, Pc, params, z, inflow, t_ops, g_ops, Δt, Tsurf, ϕbase, Nt, 1e-8, 100)
        Γ = picard!(H, Pc, Γ,
                    params, t_ops, g_ops,
                    z, inflow,
                    Δt, Tsurf, ϕbase,
                    Nt, 1e-8, 100)
    else
        # or explicitly
        H[:] = RK4(H, Δt, Nt, params, g_ops, enthalpy_rhs)
    end
        
    #--- re-partition ---#
    T = get_temp(H, 0.0)
    Γ = partition_temp_cold(T, p, z)
    
    ϕ = get_porosity(H, 0.0)
    update_ϕ_ops!(Γ, ϕ, Nt, params, t_ops)

    #plot(Pc[1:Nt], z[1:Nt], label="Pc")
    #display(plot!(H[:], z, label="H"))
    #sleep(.05)
    
    return (Γ, H, Pc)

end

function solve_Pc!(Nt, Pc, params, t_ops)

    κ = params.κ
    δ = params.δ
    η = params.η
    g = params.g
    Pcbase = params.Pcbase

    Kϕ = @view t_ops.Kϕ[1:Nt, 1:Nt]
    Mϕ = @view t_ops.Mϕ[1:Nt, 1:Nt]
    Fϕ = @view t_ops.Fϕ[1:Nt]

    
    A = -κ * δ .* Kϕ - 1/η .* Mϕ
    R = κ * g .* Fϕ
    enforce_dirchlet!(A, R, Pcbase, 1)

    Pc[1:Nt] .= A\R
    
end

#explict timestepping for enthalpy method
function enthalpy_rhs(h, inflow, ϕbase, Nt, g_ops)

    Q = g_ops.Q
    S = g_ops.S
    Mlump = g_ops.Mlump
    F = g_ops.F

    A = Mlump * (- S - Q)
    RHS = A * h + Mlump * F
    RHS[end] = 0.0
    if inflow == true
        RHS[Nt] = 0.0
    else
        RHS[1] = ϕbase
    end
        
    return RHS
end


function RK4(u, Δt, Nt, params, g_ops, rhs)

    k1 = Δt * rhs(u[:], params.inflow, params.ϕbase, Nt, g_ops)
    k2 = Δt * rhs(u[:] + k1/2, params.inflow, params.ϕbase, Nt, g_ops)
    k3 = Δt * rhs(u[:] + k2/2, params.inflow, params.ϕbase, Nt, g_ops)
    k4 = Δt * rhs(u[:] + k3, params.inflow, params.ϕbase, Nt, g_ops)

    u_raw = u[:] + (k1 + 2k2 + 2k3 + k4) / 6
    u_smooth = Smoothing.binomial(u_raw, 1)

    return u_raw
    
end

# returns the element number in which the temperate boundary exists
function partition_temp_cold(T, p, z)
    for i=1:length(T)-1
        if T[i] >= 0 && T[i+1] < 0
            e, nodes = XToN(z[i], p, z)
            return e
        end
    end
end

function picard!(H, Pc, Γ, params, t_ops, g_ops,
                 z, inflow, Δt, Tsurf, ϕbase,
                 Nt, tol, maxiter)


    M = g_ops.M
    S = g_ops.S
    F = g_ops.F

    eps = 1e-12
    iter = 0
    err = Inf

    # previous timestep state

    H_old  = copy(H)
    Pc_old = copy(Pc)
    H_iter = copy(H_old)
    Pc_iter = copy(Pc_old)

    # previous timestep Q
    solve_Pc!(Nt, Pc, params, t_ops)
    T0 = get_temp(H_old, 0.0)
    Γ0 = partition_temp_cold(T0, params.p, z)
    ϕ0 = get_porosity(H_old, 0.0)
    update_ϕ_ops!(Γ0, ϕ0, Nt, params, t_ops)
    update_Q!(Γ0, Nt, Pc_old, params, t_ops, g_ops)
    Q_old = copy(g_ops.Q)

    H_prev  = similar(H_iter)
    Pc_prev = similar(Pc_iter)
    Γ_prev = copy(Γ)

    #picard loop
    while err > tol && iter < maxiter
        # get previous iteration k
        H_prev .= H_iter
        Pc_prev .= Pc_iter
        Γ_prev = Γ

        # compute new H and Pc k+1
        T = get_temp(H_iter, 0.0)
        Γ = partition_temp_cold(T, params.p, z)
        ϕ = get_porosity(H_iter, 0.0)
        update_ϕ_ops!(Γ, ϕ, Nt, params, t_ops)
        solve_Pc!(Nt, Pc_iter, params, t_ops)
        update_Q!(Γ, Nt, Pc_iter, params, t_ops, g_ops)
        Q_new = g_ops.Q
        A = M + (Δt/2) * (S + Q_new)
        R = (M - (Δt/2) * (S + Q_old)) * H_old + Δt * F
        enforce_dirchlet!(A, R, Tsurf, size(A,1))
        if inflow
            enforce_dirchlet!(A, R, 0.0, Nt)
        else
            enforce_dirchlet!(A, R, ϕbase, 1)
        end
        H_iter .= A \ R

        # convergence check
        err_H  = norm(H_iter - H_prev) / (norm(H_prev) + eps)
        err_Pc = norm(Pc_iter - Pc_prev) / (norm(Pc_prev) + eps)
        errΓ = norm(Γ .- Γ_prev) / (norm(Γ_prev) + eps)

        err = max(err_H, err_Pc, eps)
        iter += 1
    end

    H .= H_iter
    Pc .= Pc_iter

    return Γ
end
