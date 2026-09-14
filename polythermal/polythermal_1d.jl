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
    Mχ::SparseMatrixCSC{Float64, Int64}
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
    initial_enth(z, Tsurf) = z > .5 ? Tsurf * s.((z - .5) / .5) : -.1 * (z - .5)
    initial_temp(z, Tsurf) = z > .5 ? Tsurf * s.((z - .5) / .5) : 0
    initial_pore(z) = z < .5 ? -.1 * (z - .5) : 0


    #--- Options ---#
    # implicit or explict timestepping
    implicit = true
    # SUPG stabilization
    SUPG = true
    # chi-regularization
    # reg == true is implicit-only
    reg = true
    # if DG then regularization is ignored
    DG = true
    # inflow or outflow problem
    inflow = false
    
    #---- physical parameters ----#
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
    # number of elements
    #Nes = 8:8:8 + (8 * 8)
    #for Ne in Nes
    Ne = 16
    # basis order
    p = 2
    # number basis functions
    Nbasis = p + 1
    # number of nodes
    N = DG ? Ne * Nbasis : p*Ne + 1
    # domain boundarys [L, B]
    L = 1.0
    B = 0
    # length of element
    he = (L-B)/Ne
    # regularization function
    ϵ = 8 * he
    # permeability floor for the regularized (whole-domain) compaction solve
    ϵp = 0
    # strength of compaction pressure regularization
    γ = 1
    χfunc(H) = .5 * (1 + tanh(H/ϵ))
    # nodes
    ref_nodes, weights = gausslobatto(Nbasis)
    z = get_mesh(Ne, p, L, N, he, ref_nodes, DG)
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
    H[:] = initial_enth.(z, Tsurf)
    H1 = zeros(N)
    H1[:] = initial_enth.(z, Tsurf)

    # compaction pressure
    Pc = zeros(N)

    # advective cfl
    if implicit == true
        Δt = he/(2*abs(u(1)))
    else
        Δt = min(he/abs(u(1)), (1/4) * he^2/κ)
    end

    # element tensor matrix used to assemble coupled matrices (shared)
    nodes = z[1:p+1]
    mt = precompute_local_tensor(Nbasis, p, nodes, lb, lb, lb)
    st = precompute_local_tensor(Nbasis, p, nodes, dlb, lb, lb)
    kt = precompute_local_tensor(Nbasis, p, nodes, dlb, dlb, lb)
    dm = precompute_local_mat(Nbasis, p, nodes, dlb, lb)
    km = precompute_local_mat(Nbasis, p, nodes, dlb, dlb)
    mv = precompute_local_vec(Nbasis, p, nodes, lb)
    sv = precompute_local_vec(Nbasis, p, nodes, dlb)

    nnz = NNZ(Ne, Nbasis)

    # static global operators (shared by both paths)
    Mlump, M = get_lumped_mass(Ne, Nbasis, p, z, N)
    S = get_advection_matrix(Ne, Nbasis, p, z, u, N)
    F = zeros(N)
    assemble_global_static_vec_from_local_vec!(Ne, Nbasis, p, a(.5), mv, F, false)

    #--- operator/interface setup
    if !reg

        Γ = partition_temp_cold(H, p, z)
        Γ_prev = Γ
        Γc = Ne - Γ
        Γ_nodes = EToN(Γ, p)
        Nt = Γ_nodes[end]

        nnzt = NNZ(Γ, Nbasis)
        It, Jt = get_sparsity(Γ, nnzt, Nbasis, p)

        ϕ = get_porosity(H, 0.0)
        Kϕ, Mϕ, Mχ, Fϕ = get_temperate_ops(N)
        t_ops = tOps(nnzt, Kϕ, Mϕ, Mχ, Fϕ, mt, kt, dm, km, st, sv)

        Kc = get_diffusion_matrix(Γc, Nt, Nbasis, p, z, N)
        g_ops = gOps(spzeros(N,N),
                     S, F, zeros(N),
                     Mlump, M,
                     spzeros(N,N), Kc)
    else

        Γ = 0
        I, J = get_sparsity(Ne, nnz, Nbasis, p)

        ϕ = max.(χfunc.(H) .* H, 0.0)
        Kϕ, Mϕ, Mχ, Fϕ = get_temperate_ops(N)
        t_ops = tOps(nnz, Kϕ, Mϕ, Mχ, Fϕ, mt, kt, dm, km, st, sv)
        Q0     = sparse(I, J, ones(nnz), N, N); fill!(Q0.nzval, 0.0)
        Msupg0 = sparse(I, J, ones(nnz), N, N); fill!(Msupg0.nzval, 0.0)
        g_ops = gOps(Q0,
                     S, F, zeros(N),
                     Mlump, M,
                     Msupg0, spzeros(N,N))
    end
    
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
              τ = τ,
              reg = reg,
              ϵ = ϵ,
              ϵp = ϵp,
              γ = γ,
              χ = χfunc)

    t_final = 2.7
    tsteps = Int(ceil(t_final / Δt))

    for i = 1:tsteps
        if !reg
            (Γ, H, Pc) = timestep(H, Pc, Γ, params, t_ops, g_ops, Δt)
        else
            (H, Pc) = timestep_reg(H, Pc, params, t_ops, g_ops, Δt)
        end
        #plot(H, z, label='H')
        #display(plot!(Pc, z, label="Pc"))
    end

    #=
    #plot(H, z, label='H')
    #display(plot!(Pc, z, label="Pc"))
    open("upwards_advect_8elm.txt", "a") do io
        writedlm(io, [z, H, Pc])
    end
    =#
    
    if !reg
        Γ_nodes = EToN(Γ, p)
        Nt = Γ_nodes[end]
    end
    #end
    
nothing
end
