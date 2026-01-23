using Printf
using Plots
using DelimitedFiles
using FastGaussQuadrature

include("assemble.jl")
include("timestepping.jl")
#include("GLL.jl")

mutable struct tOps
    nnzt::Int64
    Kϕ::SparseMatrixCSC{Float64, Int64}
    Mϕ::SparseMatrixCSC{Float64, Int64}
    Fϕ::Vector{Float64}
    mt::Matrix{Float64}
    kt::Matrix{Float64}
    dm::Matrix{Float64}
end

mutable struct gOps
    Q::SparseMatrixCSC{Float64, Int64}
    S::SparseMatrixCSC{Float64, Int64}
    F::Vector{Float64} 
    Mlump::SparseMatrixCSC{Float64, Int64}
    M::SparseMatrixCSC{Float64, Int64}
    Kc::SparseMatrixCSC{Float64, Int64}
end

let

    #---- physical parameters ----#
    # inflow or outflow problem
    inflow = false
    u(z) = nothing
    # velocity
    if inflow == true
        u(z) = -1.0
    else
        u(z) = 1.0
    end
    # inverse peclet number
    Pe_inv(z) = 1.0
    # dissipation rate
    a(z) = 1.0
    # thermal conductivity
    if inflow == true
        κ = 1.0
    else
        κ = .25
    end
    # gravitational acceleration
    g = -1.0
    # carman-kozeny exponent
    α = 2.33
    # compaction parameter
    δ = 1.25e-2
     # ice viscosity
    η = 1.0
    
    #---- numerical parameters ----#
    #GLL interp nodes
    GLL = true
    # implicit or explict timestepping
    implicit = true
    # number of elements
    Ne = 64
    # basis order
    p = 2
    # number basis functions
    Nbasis = p + 1
    # number of nodes
    N = p*Ne + 1
    # domain boundarys [L, B]
    L = 1.0
    B = 0
    # length of element
    he = (L-B)/Ne
    # nodes
    ref_nodes, weights = gausslobatto(Nbasis)
    z = get_mesh(Ne, p, L, N, he, ref_nodes)
    # minimum element size
    hmin = z[2] - z[1]
    zfine = collect(B:hmin/3:L)
    #---- initial and boundary data ----#
    # surface temperature
    if inflow == true
        Tsurf = -.1
    else
        Tsurf = -.5
    end
    # compaction pressure at the base
    Pcbase = 1.0
    # porosity base
    ϕbase = .2

    # solution to steady BVP for temperature
    cold_steady_test(z) = Tsurf + a.(z)/u.(z) * (z - H) +
        (a.(z)/u.(z).^2) * (exp(u.(z) * (H-B)) - exp(u.(z) * (z - B)))
    
    # initial condition
    s(t) = 3t^2 - 2t^3
    initial_enth(z) = nothing
    if inflow == true
        initial_enth(z) = z > .5 ? Tsurf * s.((z - .5) / .5) : -.1 * (z - .5)
    else
        initial_enth(z) = z > .5 ? Tsurf * s.((z - .5) / .5) : -.4 * (z - .5)
    end
    initial_temp(z) = z > .5 ? Tsurf * s.((z - .5) / .5) : 0

    # initial enthalpy
    H = zeros(N)
    H[:] = initial_enth.(z)
    H1 = zeros(N)
    H1[:] = initial_enth.(z)
    
    # compaction pressure
    Pc = zeros(N)

    # advective cfl
    if implicit == true
        Δt = he/(2*abs(u(1)))
        Δt = min(he/abs(u(1)), (1/4) * he^2/κ)
    else
        Δt = min(he/abs(u(1)), (1/4) * he^2/κ)
    end
    
    #--- interface info ---#
    Γ = partition_temp_cold(H, p, z)
    Γ_prev = Γ
    Γc = Ne - Γ
    Γ_nodes = EToN(Γ, p)
    Nt = Γ_nodes[end]

    nnzt = NNZ(Γ, Nbasis)
    nnz = NNZ(Ne, Nbasis)
    It, Jt = get_sparsity(Γ, nnzt, Nbasis, p)
    
    # element tensor matrix used to assemble coupled matrices
    nodes = z[1:p+1]
    fine = 0:.00001:nodes[end]
    #=
    p1 = plot()
    for i in 1:p+1
        p1 = plot!(fine, [lb(f, i, nodes) for f in fine])
    end
    display(p1)
    =#
    mt = precompute_local_tensor(Nbasis, p, nodes, lb, lb, lb)
    kt = precompute_local_tensor(Nbasis, p, nodes, dlb, dlb, lb)
    dm = precompute_local_mat(Nbasis, p, nodes, dlb, lb)

    ϕ = get_porosity(H, 0.0)
    Kϕ, Mϕ, Fϕ = get_temperate_ops(Γ, N, nnzt, Nbasis, p,
                                   ϕ, α, mt, kt, dm, It, Jt)

    t_ops = tOps(nnzt, Kϕ, Mϕ, Fϕ, mt, kt, dm)

    # static global operators
    Mlump, M = get_lumped_mass(Ne, Nbasis, p, z, N)
    S = get_advection_matrix(Ne, Nbasis, p, z, u, N)
    Kc = get_diffusion_matrix(Γc, Nt, Nbasis, p, κ, z, N)
    F = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, z, lb, a, one, F)

    g_ops = gOps(spzeros(N,N), S, F, Mlump, M, Kc)

    params = (inflow = inflow,
              implicit = implicit,
              N = N,
              Ne = Ne,
              Nbasis = Nbasis,
              p = p,
              z = z,
              u = u,
              a = a,
              Tsurf = Tsurf,
              Pcbase = Pcbase,
              ϕbase = ϕbase,
              Pe_inv = Pe_inv,
              δ = δ,
              α = α,
              η = η,
              g = g,
              κ = κ)

    t_final = 10
    tsteps = Int(ceil(t_final / Δt))
    
    for i = 1:tsteps
        (Γ, H, Pc) = timestep(H, Pc, Γ, params, t_ops, g_ops, Δt)
        plot(H, z, label='H')
        display(plot!(Pc, z, label="Pc"))
        #sleep(1)
        #@show Γ
    end

    display(plot(H, z, label='H'))
    display(plot!(Pc, z, label="Pc"))
    
    Γ_nodes = EToN(Γ, p)
    Nt = Γ_nodes[end]

    nothing
     
end
