using Printf
using Plots

include("assemble.jl")
include("timestepping.jl")

let

    #---- testing solutions ----#
    # solution to steady BVP for temperature
    cold_steady_test(z) = Tsurf + a.(z)/u.(z) * (z - H) +
        (a.(z)/u.(z).^2) * (exp(u.(z) * (H-B)) - exp(u.(z) * (z - B)))
    
    s(t) = 3t^2 - 2t^3
    initial_temp(z) = z > .5 ? Tsurf * s.((z - .5) / .5) : 0
    initial_pore(z) = z < .5 ? -.1 * (z - .5) : 0
    
    #---- physical parameters ----#
    
    # velocity
    u(z) = -1.0
    # inverse peclet number
    Pe_inv(z) = 1.0
    # dissipation rate
    a(z) = Pe_inv(z) * 1
    # thermal conductivity
    κ = 1.0
    # gravitational acceleration
    g = -1.0
    # carman-kozeny exponent
    α = 2.0
    # compation parameter
    δ = 1.0
    # ice viscosity
    η = 1.0
    
    #---- numerical parameters ----#
    
    # number of elements
    Ne = 15
    # basis order
    p = 2
    # number of nodes
    N = p*Ne + 1
    # domain boundarys [H, B]
    H = .5
    B = 0
    # length of element
    h = (H-B)/(N-1)
    # nodes
    z = collect(B:h:H)
    zfine = collect(B:h/2:H)
    # number basis functions
    Nbasis = p + 1
    
    #---- initial and boundary data ----#
                      
    # surface temperature
    Tsurf = -.05
    # compaction pressure at the base
    Pcbase = .5
    
    # initial temperature data
    T = zeros(N, 2)
    T[:, 2] = initial_temp.(z)
    # test parabola: -1.05/B^2 .* (z .- B).^2 .+ 1
    
    # initial porosity    
    ϕ = zeros(N, 2)
    ϕ[:, 1] = initial_pore.(z)
    ϕ[:, 2] = initial_pore.(z)
    # compaction pressure
    Pc = zeros(N)

    #---- get cold operators ----#
    (K, S, M, F) = get_temperature_ops(N, Ne, Nbasis,
                                       p, z, u, ϕ[:, 2], a, α)
    #---- Solution sequence ----#
    #=
    # cold steady state test
    F = F
    A = K
    enforce_dirchlet!(A, F, Tsurf, 1)
    display(K)
    display(F)
    T[:,1] .= A\F

    T_sol = cold_steady_test.(z)

    error = sqrt(sum((T[:,1] - T_sol).^2))
    @show error
    
    plot(T[:,1], z)
    display(plot!(cold_steady_test.(z), z))
    =#
    
    # advective cfl
    Δt = h/abs(u(1))

    # get divide index
    # Γ = partition_temp_cold(T[:,2])
    Γ = 0
    
    params = (N = N,
              Ne = Ne, 
              p = p,
              z = z,
              K = K,
              S = S,
              M = M,
              F = F,
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
    @show N
    for i = 0:0
        Γ = timestep(T, ϕ, Pc, Γ, params, Δt)
    end

    nothing
     
end
