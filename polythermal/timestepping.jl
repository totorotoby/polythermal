using Smoothing
using Arpack
include("assemble.jl")
include("sol_tests.jl")



function timestep(T, ϕ, Pc, Γ, params, Δt)

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
    Nc = N - Nt

    #---- Set up domains to solve on by partitioning ----#
    # TODO: this is probably pretty memory inefficent and should be done with views, and rescaling of matrices
    #---- get cold operators ----#
    (K, S, M, F) = get_temperature_ops(Ne - Γ, Nbasis,
                                       p, z, u, a, α)
    #---- Temperature solve ----#
    ### Crank-Nicolson for time discretization
    A = M + Δt/2 .* (K + S)
    # forcing term can be replaced with 2 time slices if variable (evolving velocity field)
    R = (M - Δt/2 .* (K + S)) * T[Nt:end, 2] + Δt/2 .* (F + F)
    # this is redundent computation after the first timestep...
    enforce_dirchlet!(A, R, Tsurf, 1)
    enforce_dirchlet!(A, R, 0, 0)
    
    # solve for next temperature
    T[Nt:end,1] .= A\R
    T[:, 2] .= T[:,1]
    
    #---- compaction pressure solve ----#
    Kcomp, Mcomp, Fcomp, ϕαinterp = get_compaction_ops(Γ,
                                                       p + 1, p, z,
                                                       ϕ, α)
    
    A = -κ * δ .* Kcomp - 1/η .* Mcomp
    R = κ * g .* Fcomp
    enforce_dirchlet!(A, R, Pcbase, 0)

    # sovle BVP for compation pressure
    Pc[1:Nt] .= A\R

    #---- porosity solve ----#
    Mlump, Mpc, Stemp, Ftemp = get_porosity_ops(Γ, p + 1, p, z[1:Nt], u, a, Pc[1:Nt])
    
    ops = (Nt = Nt,
           Mlump = Mlump,
           Stemp = Stemp,
           Mpc = Mpc,
           Ftemp)

    RK4!(ϕ, Δt, ops)
    
    plot(ϕ[1:Nt, 1], z[1:Nt], label="ϕ")
    plot!(Pc[1:Nt], z[1:Nt], label="Pc")
    display(plot!(T[:,1], z, label="T"))
    #3sleep(.05)
    # re-partition
    return (partition_temp_cold(T[:, 1], p, z), ϕ, T, Pc)
    
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

    Nt = params.Nt
    
    k1 = Δt * porosity_rhs(ϕ[1:Nt, 2], params)
    k2 = Δt * porosity_rhs(ϕ[1:Nt, 2] + k1/2, params)
    k3 = Δt * porosity_rhs(ϕ[1:Nt, 2] + k2/2, params)
    k4 = Δt * porosity_rhs(ϕ[1:Nt, 2] + k3, params)

    ϕ_raw = ϕ[1:Nt, 2] + (k1 + 2k2 + 2k3 + k4) / 6
    ϕ_smooth = Smoothing.binomial(ϕ_raw, 1)
    ϕ[1:Nt, 1] .= ϕ_smooth
    ϕ[1:Nt, 2] .= ϕ[1:Nt, 1]
    
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


