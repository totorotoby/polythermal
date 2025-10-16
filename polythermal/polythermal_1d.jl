using Printf
using Plots

include("assemble.jl")
include("timestepping.jl")

let

    #---- testing solutions ----#
    # solution to steady BVP for temperature
    cold_steady_test(z) = Hsurf + a.(z)/u.(z) * (z - H) +
        (a.(z)/u.(z).^2) * (exp(u.(z) * (H-B)) - exp(u.(z) * (z - B)))
    
    s(t) = 3t^2 - 2t^3
    initial_enth(z) = z > .5 ? Hsurf * s.((z - .5) / .5) : -.1 * (z - .5)
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
    Ne = 128
    # basis order
    p = 2
    # number of nodes
    N = p*Ne + 1
    # domain boundarys [L, B]
    L = 1.0
    B = 0
    # length of element
    h = (L-B)/(N-1)
    # nodes
    z = collect(B:h:L)
    zfine = collect(B:h/2:L)
    # number basis functions
    Nbasis = p + 1
    
    #---- initial and boundary data ----#
                      
    # surface temperature
    Tsurf = -.1
    Hsurf = -.1
    # compaction pressure at the base
    Pcbase = 1.0
    
    # initial enthalpy
    H = zeros(N, 2)
    H[:, 2] = initial_enth.(z)

    display(plot(H[:,2], z))
    
    # initial temperature data
    T = zeros(N, 2)
    T[:, 2] = initial_temp.(z)

    # initial porosity    
     ϕ = zeros(N, 2)
     ϕ[:, 1] = initial_pore.(z)
     ϕ[:, 2] = initial_pore.(z)

    # compaction pressure
    Pc = zeros(N)

    #---- Solution sequence ----#

    # advective cfl
    Δt = h/abs(u(1))

    # get divide index
    Γ = partition_temp_cold(T[:,2], p, z)
    
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
    
    for i = 1:200
        (Γ, H, T, ϕ, Pc) = timestep(H, T, ϕ, Pc, Γ, params, Δt)
    end

    nothing
     
end
