include("sol_tests.jl")


function timestep(T, ϕ, Pc, Γ, params, Δt)

    N = params.N
    p = params.p
    z = params.z
    K = params.K
    S = params.S
    M = params.M
    F = params.F
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
    
    #---- Set up domains to solve on by partitioning ----#
    #=
    # TODO: this is probably pretty memory inefficent and should be done with views
    M, K, S, F = restrict_cold(M, K, S, F, Γ)
    
    #---- Temperature solve ----#

    ### Crank-Nicolson for time discretization
    A = M + Δt/2 .* (K + S)
    # forcing term can be replaced with 2 time slices if variable (evolving velocity field)
    R = (M - Δt/2 .* (K + S)) * T[Γ:end,2] + Δt/2 .* (F + F)
    # this is redundent computation after the first timestep...
    enforce_dirchlet!(A, R, Tsurf, 1)
    
    # solve for next temperature
    T[Γ:end,1] .= A\R
    T[:,2] .= T[:,1]
    =#
    #---- compaction pressure solve ----#
    ϕ[1:Γ, 1] .= 1
    Kϕα, Mϕ, Fϕα, ϕαinterp = get_compaction_ops(Int(floor((Γ - 2)/p)), N,
                                                p + 1, p, z,
                                                ϕ, α)

    @show κ, δ, g, Γ-1
    
    A = -κ * δ .* Kϕα #- 1/η .* Mϕ
    R = κ * g .* Fϕα
    enforce_dirchlet!(A, R, Pcbase, 0)

    display(A)
    display(R)
    # sovle BVP for compation pressure
    Pc[1:Γ-1] .= A\R

    
    dp = (Pc[Γ-1] - Pc[Γ-2])/(z[Γ-1] - z[Γ-2])
    nv = dp + g/δ

    test_sol = compation_test_1.(z[1:Γ-1])
    
    plot(test_sol, z[1:Γ-1])
    display(plot!(Pc[1:Γ-1], z[1:Γ-1]))
    #@show nv
    @assert nv ≈ 0 

    #---- porosity solve ----#

    #Pc = ones(length(F))
    Mlump, Mpc, Stemp, Ftemp = get_porosity_ops(Int((Γ - 2)/p), Γ-1, p + 1, p, z[1:Γ - 1], u, a, Pc[1:Γ-1])
    
    ops = (Γ = Γ,
           Mlump = Mlump,
           Stemp = Stemp,
           Mpc = Mpc,
           Ftemp)

    RK4!(ϕ, Δt, ops)

    #plot!(ϕ[1:Γ-1, 2], z[1:Γ-1], label="ϕ")
    #display(plot!(T[Γ:end,1], z[Γ:end], label="T" ))
    sleep(.3)
    # re-partition
    return partition_temp_cold(T[:, 1])
    
end

function porosity_rhs(ϕ, params)

    Mlump = params.Mlump
    Stemp = params.Stemp
    Mpc = params.Mpc
    Ftemp = params.Ftemp
    
    # TODO: really need Pe_inv * Mpc but coded myself into a hole
    RHS = Mlump * ((-Stemp - Mpc) * ϕ + Ftemp)
    RHS[end] = 0.0
    return RHS

end

function RK4!(ϕ, Δt, params)

    Γ = params.Γ
    
    k1 = Δt * porosity_rhs(ϕ[1:Γ-1, 2], params)
    k2 = Δt * porosity_rhs(ϕ[1:Γ-1, 2] + k1/2, params)
    k3 = Δt * porosity_rhs(ϕ[1:Γ-1, 2] + k2/2, params)
    k4 = Δt * porosity_rhs(ϕ[1:Γ-1, 2] + k3, params)

    ϕ[1:Γ-1, 1] .= ϕ[1:Γ-1, 2] + (k1 + 2k2 + 2k3 + k4) / 6
    ϕ[1:Γ-1, 2] .= ϕ[1:Γ-1, 1]
    
end

function partition_temp_cold(T)
    for i=1:length(T)
        if T[i] >= 0 && T[i+1] < 0
            return i
        end
    end
end

function restrict_cold(M, K, S, F, Γ)
    
    N = size(M)[1]

    Mn = M[1, :]
    Kn = K[1, :]
    Sn = S[1, :]
    Fn = F[1]
    
    M = M[Γ:end,Γ:end]
    K = K[Γ:end,Γ:end]
    S = S[Γ:end,Γ:end]
    F = F[Γ:end]

    M[1, :] = Mn[1 : N - Γ + 1]
    K[1, :] = Kn[1 : N - Γ + 1]
    S[1, :] = Sn[1 : N - Γ + 1]
    F[1] = Fn
    
    return M, K, S, F 
end
