using Printf
using Plots
using DelimitedFiles

include("assemble.jl")
include("timestepping.jl")

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
    
    # velocity
    u(z) = -1.0
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
    # length between nodes
    h = (L-B)/(N-1)
    # length of element
    he = (L-B)/Ne
    # nodes
    z = collect(B:h:L)
    zfine = collect(B:h/2:L)
    
    #---- initial and boundary data ----#
    # surface temperature
    Tsurf = -.1
    # compaction pressure at the base
    Pcbase = 1.0
    
    # initial enthalpy
    H = zeros(N)
    H[:] = initial_enth.(z)

    # initial temperature data
    T = zeros(N, 2)
    T[:, 2] = initial_temp.(z)

    # initial porosity    
     ϕ = zeros(N)
     ϕ[:] = initial_pore.(z)

    # compaction pressure
    Pc = zeros(N)

    # advective cfl
    #Δt = h/abs(u(1))
    Δt = min(h/abs(u(1)), (1/3) * h^2/κ)

    #--- interface info ---#
    Γ = partition_temp_cold(T[:,2], p, z)
    Γc = Ne - Γ
    Γ_nodes = EToN(Γ, p)
    Nt = Γ_nodes[end]

    nnzt = NNZ(Γ, Nbasis)
    nnz = NNZ(Ne, Nbasis)
    It, Jt = get_sparsity(Γ, nnzt, Nbasis, p)
    I, J = get_sparsity(Ne, nnz, Nbasis, p)
    
    # element tensor matrix used to assemble coupled matrices
    nodes = collect(0:he/p:he)
    mt = precompute_local_tensor(Nbasis, p, nodes, lb, lb, lb)
    kt = precompute_local_tensor(Nbasis, p, nodes, dlb, dlb, lb)
    de = precompute_local_mat(Nbasis, p, nodes, dlb, lb)
    
    # static global operators
    Mlump = get_lumped_mass(Ne, Nbasis, p, z, N)
    S = get_advection_matrix(Ne, Nbasis, p, z, u, N)
    VK, Ic, Jc = get_diffusion_matrix(Γc, Nt, Nbasis, p, z, N)
    
    # melting source term
    F = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, z, lb, a, one, F)

    VKϕ = zeros(nnzt)
    VMϕ = zeros(nnzt)
    VMP = zeros(nnzt)
    VQ = zeros(nnz)
    Fϕ = zeros(Nt)
    
    params = (N = N,
              Ne = Ne,
              Nbasis = Nbasis,
              p = p,
              z = z,
              u = u,
              a = a,
              Tsurf = Tsurf,
              Pcbase = Pcbase,
              Pe_inv = Pe_inv,
              δ = δ,
              α = α,
              η = η,
              g = g,
              κ = κ)

    c_ops = (Ic = Ic,
             Jc = Jc,
             VK = VK)
    
    g_ops = (I = I,
             J = J,
             VQ = VQ,
             S = S,
             F = F,
             Mlump = Mlump)
    
    t_ops = (nnzt = nnzt,
             It = It,
             Jt = Jt,
             VKϕ = VKϕ,
             VMϕ = VMϕ,
             VMP = VMP,
             Fϕ = Fϕ,
             mt = mt,
             kt = kt,
             de = de)

    
    for i = 1:1
        (Γ, H, T, ϕ, Pc) = timestep(H, T, ϕ,
                                    Pc, Γ, params,
                                    c_ops, t_ops,
                                    g_ops, Δt)
    end

    Γ_nodes = EToN(Γ, p)
    Nt = Γ_nodes[end]

    #writedlm("H.end", H, ',')
    
    #plot(ϕ[1:Nt], z[1:Nt], label="ϕ")
    plot(Pc[1:Nt], z[1:Nt], label="Pc")
    display(plot!(H[:], z, label="H"))
    #display(plot!(T[:,1], z, label="T"))
    
    
    
    nothing
     
end
