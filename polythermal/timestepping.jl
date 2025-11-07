using Smoothing
using Arpack
include("assemble.jl")
include("sol_tests.jl")



function timestep(H, T, ϕ, Pc, Γ, params, ops, Δt)

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
    
    Γc = Ne - Γ
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
    # sovle BVP for compation pressure
    Pc[1:Nt] .= A\R

    plot(Pc[1:Nt], z[1:Nt])
    display(plot!(Pc[1:Nt], z[1:Nt]))
    error()

    #--- solve for ethalpy ---#
    Q, S, M,
    Mlump, F = get_enth_ops(Ne, N, Γ,
                                     Nt, Nbasis,
                                     p, z, u, a,
                                     Pc)

    #=
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
    =#
    
    # explicit (and stiff) solve
    ops = (Nt = Nt,
           Q = Q,
           S = S,
           Mlump = Mlump,
           F = F,
           Tsurf = Tsurf)

    H[:] = RK4(H, Δt, ops, enthalpy_rhs)

    Kϕ, Mϕ, MPe, Fϕ = get_temp_ops()



    
    update_cold_ops!()



    #--- re-partition ---#
    T_temp = get_temp(H, 0.0)
    Γ = partition_temp_cold(T_temp, p, z)


    #plot(ϕ[1:Nt], z[1:Nt], label="ϕ")
    #plot(Pc[1:Nt], z[1:Nt], label="Pc")
    #display(plot!(H[:], z, label="H"))
    #display(plot!(T[:,1], z, label="T"))
    #sleep(.05)

    
    return (Γ, H, T, ϕ, Pc)
    
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

    iter = 0

    #while sum((H - Hprev).^2) > tol && iter < maxiter
    #end
         
    
end
