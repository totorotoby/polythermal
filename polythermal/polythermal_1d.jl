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
    bpc::Vector{Float64}
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
    initial_enth(z, Tsurf) = z > .5 ? Tsurf * s.((z - .5) / .5) : -.4 * z + .2
    initial_temp(z, Tsurf) = z > .5 ? Tsurf * s.((z - .5) / .5) : 0
    initial_pore(z) = z < .5 ? -.1 * (z - .5) : 0


    #--- Options ---#
    # implicit or explict timestepping
    implicit = true
    # SUPG stabilization
    SUPG = true
    # chi-regularization
    # reg == true is implicit-only
    reg = false
    # if DG then regularization is ignored
    DG = false
    # inflow or outflow problem
    inflow = true
    
    #---- physical parameters ----#
    u(z) = nothing
    # velocity
    u(z) = inflow ? -1.0 : 1.0
    # inverse peclet number
    Pe_inv(z) = 1.0
    # dissipation rate
    a(z) = 1.0
    # permeability
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
    Ne = 64
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
    ϵ = 1 * he
    # permeability floor for the regularized (whole-domain) compaction solve
    ϵp = 0
    # strength of compaction pressure regularization
    γ = 1
    χfunc(H) = .5 * (1 + tanh(H/ϵ))
    # nodes
    ref_nodes, weights = gausslobatto(Nbasis)
    z = get_mesh(Ne, p, B, L, N, he, ref_nodes, DG)
    #SUPG strength param
    τ = (he /2 * abs(u(.5)))
    # SIPG penalty (DG diffusion); scale ~ C * k * p^2 / he
    σ = 10.0 * p^2 / he
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

    # static global operators
    Mlump, M = get_lumped_mass(Ne, Nbasis, p, z, N, DG)
    
    S = get_advection_matrix(Ne, Nbasis,
                             p, z, u, N,
                             DG, inflow,
                             ϕbase, Tsurf)
    F = zeros(N)
    assemble_global_static_vec_from_local_vec!(Ne, Nbasis,
                                               p, a(.5),
                                               mv, F, false)

    #--- operator/interface setup
    # continuous galerkin
    if !DG
        # continuous not regularized
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
            t_ops = tOps(nnzt, Kϕ, Mϕ, Mχ, Fϕ, mt, kt, dm, km, st, sv, zeros(N))

            Kc = get_diffusion_matrix(Γc, Nt, Nbasis, p, z, N, DG)
            g_ops = gOps(spzeros(N,N),
                         S, F, zeros(N),
                         Mlump, M,
                         spzeros(N,N), Kc)
            
        # continuous regularized 
        elseif reg

            nnz = NNZ(Ne, Nbasis)
            Γ = 0
            I, J = get_sparsity(Ne, nnz, Nbasis, p)
            ϕ = max.(χfunc.(H) .* H, 0.0)
            Kϕ, Mϕ, Mχ, Fϕ = get_temperate_ops(N)
            t_ops = tOps(nnz, Kϕ, Mϕ, Mχ, Fϕ, mt, kt, dm, km, st, sv, zeros(N))
            Q = sparse(I, J, ones(nnz), N, N)
            fill!(Q.nzval, 0.0)
            Msupg = sparse(I, J, ones(nnz), N, N)
            fill!(Msupg.nzval, 0.0)
            g_ops = gOps(Q,
                         S, F, zeros(N),
                         Mlump, M,
                         Msupg, spzeros(N,N))
        end
        
    # Discontinuous galerkin
    else

        nnz = NNZDG(Ne, Nbasis)
        Kϕ, Mϕ, Mχ, Fϕ = get_temperate_ops(N)
        t_ops = tOps(nnz, Kϕ, Mϕ, Mχ, Fϕ, mt, kt, dm, km, st, sv, zeros(N))

        K = get_diffusion_matrix_DG(Ne, Nbasis, p, z, one, N, σ, inflow, Tsurf, ϕbase)
        Q = spzeros(N, N)
        
        g_ops = gOps(Q,
                     S, F, zeros(N),
                     Mlump, M,
                     spzeros(N,N), K)

        b = get_advective_boundary(N, Ne, Nbasis, u, inflow, Tsurf, ϕbase) .+
            get_diffusion_boundary(N, Ne, Nbasis, p, z, one, σ, inflow, Tsurf, ϕbase)
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

    t_final = 2
    #Δt = 0.7 * he / (abs(u(1)) * (2p + 1))
    tsteps = Int(ceil(t_final / Δt))

    #operator_eigens(S, M)
    for i = 1:tsteps
        
        #test_advect_DG!(S, M, b, H, Δt)

        if DG
            timestep_DG!(H, Pc, b, params, t_ops, g_ops, Δt)
        elseif !reg
            (Γ, H, Pc) = timestep(H, Pc, Γ, params, t_ops, g_ops, Δt)
        else
            (H, Pc) = timestep_reg(H, Pc, params, t_ops, g_ops, Δt)
        end

        if DG
            plt = plot()
            nodes = 1:Nbasis
            for e = 1:Ne
                plot!(plt, H[nodes], z[nodes], color=:orange, legend=false, xlims=[-0.5, .5])
                nodes = nodes[end] + 1 : nodes[end] + Nbasis
            end
            display(plt)
        else
            plot(H, z, label="H")
            display(plot!(Pc, z, label="Pc"))
        end
    end

    #=
    #plot(H, z, label='H')
    #display(plot!(Pc, z, label="Pc"))
    open("upwards_advect_8elm.txt", "a") do io
        writedlm(io, [z, H, Pc])
    end
    =#
    #=
    if !reg
        Γ_nodes = EToN(Γ, p)
        Nt = Γ_nodes[end]
    end
    =#
    #end
    
nothing
end
