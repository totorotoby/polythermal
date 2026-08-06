using Printf
using Plots
using DelimitedFiles
using FastGaussQuadrature
using LinearAlgebra

include("assemble.jl")
include("timestepping.jl")

mutable struct elementOps
    mt::Matrix{Float64}
    kt::Matrix{Float64}
    sm::Matrix{Float64}
    km::Matrix{Float64}
    st::Matrix{Float64}
    sv::Vector{Float64}
    W::Matrix{Float64}
    B::Matrix{Float64}
    dB::Matrix{Float64}
    Hq::Vector{Float64}
    χq::Vector{Float64}
    ϕq::Vector{Float64}
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
    u(z) = inflow ? -1.0 : 1.0
    # inverse peclet number
    Pe_inv(z) = 1.0
    # dissipation rate
    a(z) = 1.0
    # thermal conductivity
    κ = inflow ? 1.0 : 0.25
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
    # number quadtrature points
    Nq = 16
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
    #regularization params
    ϵ = 2*he
    ϵp = 1e-3
    # nodes
    ref_nodes, weights = gausslobatto(Nbasis)
    z = get_mesh(Ne, p, L, N, he, ref_nodes)
    #SUPG strength param
    τ = (he /2 * abs(u(.5)))
    #---- initial and boundary data ----#
    # surface temperature
    Tsurf = inflow ? -.1 : -.5
    # compaction pressure at the base
    Pcbase = 1.0
    # porosity base
    ϕbase = .2

    # initial enthalpy
    H = zeros(N)
    H[:] = initial_enth.(z)
    Hq = zeros(Ne * Nq)
    χq = zeros(Ne * Nq)
    ϕq = zeros(Ne * Nq)
    # compaction pressure
    Pc = zeros(N)
    
    # advective cfl
    if implicit == true
        Δt = he/(2*abs(u(1)))
    else
        Δt = min(he/abs(u(1)), (1/4) * he^2/κ)
    end

    #---- operator assembly ----#
    
    nnz = NNZ(Ne, Nbasis)

    nodes = z[1:p+1]
    mt = precompute_local_tensor(Nbasis, p, nodes, lb, lb, lb)
    st = precompute_local_tensor(Nbasis, p, nodes, dlb, lb, lb)
    kt = precompute_local_tensor(Nbasis, p, nodes, dlb, dlb, lb)
    sm = precompute_local_mat(Nbasis, p, nodes, dlb, lb)
    km = precompute_local_mat(Nbasis, p, nodes, dlb, dlb)
    mv = precompute_local_vec(Nbasis, p, nodes, lb)
    sv = precompute_local_vec(Nbasis, p, nodes, dlb)

    W = Diagonal([0.1894506104550685,	
                  0.1894506104550685,	
                  0.1826034150449236,	
                  0.1826034150449236,	
                  0.1691565193950025,	
                  0.1691565193950025,	
                  0.1495959888165767,	
                  0.1495959888165767,	
                  0.1246289712555339,	
                  0.1246289712555339,	
                  0.0951585116824928,
                  0.0951585116824928,	
                  0.0622535239386479,	
                  0.0622535239386479,	
                  0.0271524594117541,	
                  0.0271524594117541])
    
    B = precompute_basis_quad(Nbasis, nodes, lb)
    dB = precompute_basis_quad(Nbasis, nodes, dlb)

    eops = elementOps(mt, kt, sm, km, st, sv, W, B, dB, Hq, χq, ϕq)

    params = (inflow = inflow,              
              implicit = implicit,
              SUPG = SUPG,
              N = N,
              Ne = Ne,
              Nq = Nq,
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
              τ = τ,
              ϵ = ϵ,
              ϵp = ϵp)

    t_final = 2.0
    tsteps = Int(ceil(t_final / Δt))

    for i = 1:tsteps
        (H, Pc) = timestep(H, Pc, params, eops, Δt)
    end

    #plot(H, z, label='H')
    #display(plot!(Pc, z, label="Pc"))

    nothing

end
