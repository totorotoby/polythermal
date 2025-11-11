using Smoothing
using Arpack
include("assemble.jl")
include("sol_tests.jl")



function timestep(H, Pc, Γ, params, t_ops, g_ops, Δt)

    p = params.p
    Tsurf = params.Tsurf
    z = params.z
    
    Γ_nodes = EToN(Γ, p)
    Nt = Γ_nodes[end]

    #--- new solver ---#

    solve_Pc!(Nt, Pc, params, t_ops)
    update_Q!(Γ, Nt, Pc, params, t_ops, g_ops)
    picard!(H, g_ops, Δt, Tsurf, Nt, .0001, 100)
    
    #display(Pc1 - Pc)
    #error()
    #=
    ops = (Nt = Nt,
           Q = Q,
           S = g_ops.S,
           Mlump = g_ops.Mlump,
           F = g_ops.F,
           Tsurf = Tsurf)

    
    #H[:] = RK4(H, Δt, ops, enthalpy_rhs)
    =#
    #--- re-partition ---#
    T = get_temp(H, 0.0)
    Γ = partition_temp_cold(T, p, z)
    
    ϕ = get_porosity(H, 0.0)
    update_ϕ_ops!(Γ, ϕ, Nt, params, t_ops)

    
    plot(Pc[1:Nt], z[1:Nt], label="Pc")
    display(plot!(H[:], z, label="H"))

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
function enthalpy_rhs(h, params)

    Nt = params.Nt
    Q = params.Q
    S = params.S
    Mlump = params.Mlump
    F = params.F
    Tsurf = params.Tsurf
    A = Mlump * (- S - Q)
    #enforce_dirchlet!(A, F, 0, size(A)[1])
    RHS = A * h + Mlump * F
    RHS[end] = 0.0
    RHS[Nt] = 0.0

    return RHS
end


function RK4(u, Δt, params, rhs)
    Nt = params.Nt
    
    k1 = Δt * rhs(u[:], params)
    k2 = Δt * rhs(u[:] + k1/2, params)
    k3 = Δt * rhs(u[:] + k2/2, params)
    k4 = Δt * rhs(u[:] + k3, params)

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


function picard!(H, g_ops, Δt, Tsurf, Nt, tol, maxiter)

    M = g_ops.M
    S = g_ops.S
    Q = g_ops.Q
    F = g_ops.F
    
    Hprev = copy(H)

    A = (M + Δt/2 .* (S + Q))
    R = (M - Δt/2 .* (S + Q)) * Hprev + Δt .* F
    enforce_dirchlet!(A, R, Tsurf, size(A)[1])
    enforce_dirchlet!(A, R, 0.0, Nt)

    H[:] .= A\R
    
    #iter = 0

    #while sum((H - Hprev).^2) > tol && iter < maxiter
    #end
    
end

