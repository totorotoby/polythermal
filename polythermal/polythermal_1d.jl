using Printf
using Plots
using DelimitedFiles
using FastGaussQuadrature

include("assemble.jl")
include("timestepping.jl")

mutable struct tOps
    nnzt::Int64
    Kϕ::SparseMatrixCSC{Float64, Int64}
    Mϕ::SparseMatrixCSC{Float64, Int64}
    Fϕ::Vector{Float64}
    mt::Matrix{Float64}
    kt::Matrix{Float64}
    dm::Matrix{Float64}
    km::Matrix{Float64}
    st::Matrix{Float64}
    sv::Vector{Float64}
end

mutable struct gOps
    Q::SparseMatrixCSC{Float64, Int64}
    S::SparseMatrixCSC{Float64, Int64}
    F::Vector{Float64}
    Fsupg::Vector{Float64}
    Mlump::SparseMatrixCSC{Float64, Int64}
    M::SparseMatrixCSC{Float64, Int64}
    Msupg::SparseMatrixCSC{Float64, Int64}
    Kc::SparseMatrixCSC{Float64, Int64}
end

let
    #---- testing solutions ----#
    # solution to steady BVP for temperature

    cold_steady_test(z) = Tsurf + a.(z)/u.(z) * (z - H) +
        (a.(z)/u.(z).^2) * (exp(u.(z) * (H-B)) - exp(u.(z) * (z - B)))

    s(t) = 3t^2 - 2t^3
    initial_enth(z) = z > .5 ? Tsurf * s.((z - .5) / .5) : -.1 * (z - .5)
    initial_temp(z) = z > .5 ? Tsurf * s.((z - .5) / .5) : 0
    initial_pore(z) = z < .5 ? -.1 * (z - .5) : 0


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
    κ = 1.0
    # gravitational acceleration
    g = -1.0
    # carman-kozeny exponent
    α = 2.33
    # compaction parameter
    δ = 1.25e-2
    # ice viscosity
    η = 1.0

    #---- numerical parameters ----#

    # implicit or explict timestepping
    implicit = true
    # SUPG stabilization
    SUPG = true
    # number of elements
    Ne = 64
    # basis order
    p = 4
    # number basis functions
    Nbasis = p + 1
    # number of nodes
    N = p*Ne + 1
    # domain boundarys [L, B]
    L = 1.0
    B = 0
    # length of element
    he = (L-B)/Ne
    # regularization function
    ϵ = 3*he
    χfunc(H) = .5 * (1 + tanh(H/ϵ))
    # nodes
    ref_nodes, weights = gausslobatto(Nbasis)
    z = get_mesh(Ne, p, L, N, he, ref_nodes)
    #SUPG strength param
    τ = (he /2 * abs(u(.5)))
    #---- initial and boundary data ----#
    # surface temperature
    Tsurf = -.1
    # compaction pressure at the base
    Pcbase = 1.0
    # porosity base
    ϕbase = .2

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
    mt = precompute_local_tensor(Nbasis, p, nodes, lb, lb, lb)
    # TODO: FIGURE OUT IF THIS CORRECT BELOW
    st = precompute_local_tensor(Nbasis, p, nodes, dlb, lb, lb)
    kt = precompute_local_tensor(Nbasis, p, nodes, dlb, dlb, lb)
    dm = precompute_local_mat(Nbasis, p, nodes, dlb, lb)
    km = precompute_local_mat(Nbasis, p, nodes, dlb, dlb)
    mv = precompute_local_vec(Nbasis, p, nodes, lb)
    sv = precompute_local_vec(Nbasis, p, nodes, dlb)


    ϕ = get_porosity(H, 0.0)
    Kϕ, Mϕ, Fϕ = get_temperate_ops(Γ, N, nnzt, Nbasis, p,
                                   ϕ, α, mt, kt, dm, It, Jt)
    t_ops = tOps(nnzt, Kϕ, Mϕ, Fϕ, mt, kt, dm, km, st, sv)

    # static global operators
    Mlump, M = get_lumped_mass(Ne, Nbasis, p, z, N)
    S = get_advection_matrix(Ne, Nbasis, p, z, u, N)
    Kc = get_diffusion_matrix(Γc, Nt, Nbasis, p, z, N)
    F = zeros(N)
    assemble_global_static_vec_from_local_vec!(Ne, Nbasis, p, a(.5), mv, F, false)
    g_ops = gOps(spzeros(N,N),
                 S, F, zeros(N),
                 Mlump, M,
                 spzeros(N,N), Kc)
    
    params = (inflow = inflow,              
              implicit = implicit,
              SUPG = SUPG,
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
              κ = κ,
              τ = τ)

    t_final = 2.0
    tsteps = Int(ceil(t_final / Δt))

    for i = 1:tsteps
        (Γ, H, Pc) = timestep(H, Pc, Γ, params, t_ops, g_ops, Δt)
        #plot(H, z, label='H')
        #display(plot!(Pc, z, label="Pc"))
    end

    #plot(H, z, label='H')
    #display(plot!(Pc, z, label="Pc"))

    Γ_nodes = EToN(Γ, p)
    Nt = Γ_nodes[end]

    nothing

end
