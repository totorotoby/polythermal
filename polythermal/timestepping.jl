using Smoothing
using Arpack
include("assemble.jl")
include("sol_tests.jl")



function timestep(H, H1, Pc, Pc1, Γ, Γ1, params, t_ops, g_ops, Δt)

    N = params.N
    Ne = params.Ne
    Nbasis = params.Nbasis
    p = params.p
    z = params.z
    u = params.u
    a = params.a
    Tsurf = params.Tsurf
    Pcbase = params.Pcbase
    Pe_inv = params.Pe_inv
    δ = params.δ
    α = params.α
    η = params.η
    g = params.g
    κ = params.κ
    
    Γ_nodes = EToN(Γ, p)
    Nt = Γ_nodes[end]
    
    #---- compaction pressure solve ----#
    
    ϕ = get_porosity(H, 0.0)
    
    Kcomp, Mcomp,
    Fcomp, ϕαinterp = get_compaction_ops(Γ, Nbasis,
                                         p, z,
                                         ϕ, α)
    A = -κ * δ .* Kcomp - 1/η .* Mcomp
    R = κ * g .* Fcomp
    enforce_dirchlet!(A, R, Pcbase, 1)
    Pc[1:Nt] .= A\R
    plot(Pc[1:Nt], z[1:Nt], label="Pc")
    #--- solve for ethalpy ---#
    Q, S, M,
    Mlump, F = get_enth_ops(Ne, N, Γ,
                            Nt, Nbasis,
                            p, z, u, a,
                            Pc)
    

    ops = (Δt = Δt,
           F = F,
           M = M,
           Q = Q,
           S = S,
           z = z,
           Tsurf = Tsurf,
           Nt = Nt)
    
    # picard iterations
    picard!(H, ops, .0001, 3)

    plot!(H, z, label='H')
    T = get_temp(H, 0.0)
    ϕ = get_porosity(H, 0.0)
    Γ = partition_temp_cold(T, p, z)
    # explicit (and stiff) solve
    #=
    ops = (Nt = Nt,
           Q = Q,
           S = S,
           Mlump = Mlump,
           F = F,
           Tsurf = Tsurf)

    H[:] = RK4(H, Δt, ops, enthalpy_rhs)
    =#
    #--- new solver ---#
    
    ϕ1 = get_porosity(H1, 0.0)
    update_ϕ_ops!(Γ, ϕ1, Nt, params, t_ops)

    #display(Kcomp - t_ops.Kϕ[1:Nt,1:Nt])
    #display(Mcomp - t_ops.Mϕ[1:Nt,1:Nt])
    
    solve_Pc!(Nt, Pc1, params, t_ops)
    #display(maximum(abs.(Pc1 - Pc)))
    update_Q!(Γ, Nt, Pc1, params, t_ops, g_ops)
    picard!(H1, g_ops, Δt, Tsurf, Nt, .0001, 100)
    
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
    T1 = get_temp(H1, 0.0)
    Γ1 = partition_temp_cold(T1, p, z)
    #display(abs.(T1 - T))
    #plot(ϕ[1:Nt], z[1:Nt], label="ϕ")
    plot!(Pc1[1:Nt], z[1:Nt], label="Pc1")
    display(plot!(H1[:], z, label="H1"))
    #display(plot!(T[:,1], z, label="T"))
    sleep(.05)

    return (Γ, Γ1, H, H1, Pc, Pc1)
    
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

function porosity_rhs(ϕ, params)

    Nt = params.Nt
    Mlump = params.Mlump
    Stemp = params.Stemp
    Mpc = params.Mpc
    Ftemp = params.Ftemp

    # TODO: really need Pe_inv * Mpc but coded myself into a hole
    RHS = Mlump * ((-Stemp - Mpc) * ϕ + Ftemp)
    RHS[end] = 0.0
    return RHS

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


function picard!(H, ops, tol, maxiter)

    Δt = ops.Δt
    S = ops.S
    M = ops.M
    Q = ops.Q
    F = ops.F
    z = ops.z
    Tsurf = ops.Tsurf
    Nt = ops.Nt

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
