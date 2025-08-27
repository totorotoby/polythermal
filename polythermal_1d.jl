using Printf
using ForwardDiff
using Plots
using SparseArrays
using LinearAlgebra
using Statistics
using DataStructures

    

#=
gaussian integration of funcs multiplied together with args for each function
weights and abscissa pulled from: https://pomax.github.io/bezierinfo/legendre-gauss.html
element - list of at least the start and end nodes of the element to integrate over
=#
function gauss_integrate(element, funcs...)

    weights = [0.6521451548625461
               0.6521451548625461
               0.3478548451374538
               0.3478548451374538]

    abscissa = [-0.3399810435848563
    	        0.3399810435848563
    	        -0.8611363115940526
    	        0.8611363115940526]

    val = 0.0
    scale = (element[end] - element[1]) * .5
    c = (element[end] + element[1]) * .5

    for l in 1:length(weights)
        val += weights[l] * 
            reduce(*, [f(scale * abscissa[l] + c) for f in funcs])
    end
    return scale *  val
end

#=
This function assembles a discrete diffusion and advection operators from the basis functions:
    Ne: number of elements
    Nbasis: number of basis functions per element = p + 1 (might not need to be carrying this around
    p: order of basis
    func1: function to integrate in element (a set of basis functions or its derivative)
    func2: same as func1
    k: parameter, can be known or
       a guess if doing the inverse problem
    I: non zero row indices
    J: non zero column indices
    V: non zero values
=#
function assemble_matrix!(Ne, Nbasis, p,
                          x, func1, func2, k,
                          I, J, V)
    for e in 1:Ne
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            for j in 1:Nbasis
                col = (p*e) + (j-p)

                nodes = EToN(e, p, x)
                v = gauss_integrate(nodes, x -> func1(x, i, nodes) , x ->  func2(x, j, nodes), k)
                idx = inCOO(I, J, row, col)
                
                if idx > 0 
                    V[idx] += v
                else
                    push!(I, row)
                    push!(J, col)
                    push!(V, v)
                end
            end
        end
    end
end



function assemble_forcing!(Ne, Nbasis, p, x, func1, func2, forcing, F)
    for e in 1:Ne
        nodes = EToN(e, p, x)
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            F[row] += gauss_integrate(nodes, x -> func1(x, i, nodes), forcing, func2)
        end
    end
end

function inCOO(I, J, i, j)
    for idx in 1:length(I)
        if I[idx] == i && J[idx] == j
            return idx
        end
    end
    return -1
end

function enforce_dirchlet!(A, F, v, side)

    if side == 0 
        A[1, 1] = 1.0
        A[1, 2:end] .= 0
        F[1] = v
    else
        A[end, end] = 1.0
        A[end, 1:(end-1)] .= 0.0
        F[end] = v
    end
end

#---- Barycentric lagragian interpolation ----#

# computes numerator
function lag(x, nodes)
    l = 1
    for i in 1:length(nodes)
        l *= (x - nodes[i])
    end
    return l
end

# derivative of numerator for weights
dlag(x, nodes) = ForwardDiff.derivative(x -> lag(x, nodes), x)

# evaluate basis function local index j at x with element nodes "nodes"
function lb(x, j, nodes)

    l = lag(x, nodes)
    w = 1/dlag(nodes[j], nodes)
    
    if x != nodes[j]
        return (l * w)/(x - nodes[j])
    else
        return 1.0
    end
end

# basis function derivative
dlb(x, j, nodes) = ForwardDiff.derivative(x -> lb(x, j, nodes), x)

# p order lagrangian basis expansion with current coords at x
function expansion(x, p, coords, n_global)

    # get local nodes, and local coordinates
    e, n_local = XToN(x, p, n_global)
    coords_local = coords[1 + (e-1) * p : 1 + e*p]
    eval = 0
    for i in 1:p+1
        eval += coords_local[i] * lb(x, i, n_local)
    end
    
    return eval
end

# given point in domain, which element (nodes in element) is it in
function XToN(x, p, nodes)
    
    elements = nodes[1:p:end]
    e = searchsortedfirst(elements, x)
    e = e == 1 ? 1 : e - 1
    
    return e, nodes[1 + (e-1) * p : 1 + e*p]
end

#=
 Element to Nodes:
 e - element number
 p - order of basis
 nodes - list of global nodes
 returns nodes in element =#
EToN(e, p, nodes) = nodes[(e-1)*p + 1 : (e-1)*p + p + 1]

function timestep!(T, ϕ, Pc, p, Δt)

    z = p.z
    K = p.K
    S = p.S
    M = p.M
    F = p.F
    Tsurf = p.Tsurf
    Γ = p.Γ_idx

    # TODO: this is probably pretty memory inefficent
    M, K, S, F = restrict_cold(M, K, S, F, Γ)
    
    ### Crank-Nicolson for time discretization
    A = M + Δt/2 .* (K + S)
    # forcing term can be replaced with 2 time slices if variable
    R = (M - Δt/2 .* (K + S)) * T[Γ:end,2] + Δt/2 .* (F + F)

    # this is redundent computation after the first timestep...
    enforce_dirchlet!(A, R, Tsurf, 1)
    
    # solve for next temperature
    T[Γ:end,1] .= A\R
    T[:,2] .= T[:,1]

    
end

function partition_temp_cold(T)
    dist = abs.(T)
    mindist = minimum(dist)
    return findlast(==(mindist), dist)
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

let

    #---- testing solutions ----#
    # solution to steady BVP for temperature
    #cold_steady_test(z) = Tsurf + a.(z)/u.(z) * (z - H) +
    #    (a.(z)/u.(z).^2) * (exp(u.(z) * (H-B)) - exp(u.(z) * (z - B)))
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
    k = 1.0
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
    Ne = 255
    # basis order
    p = 1
    # number of nodes
    N = p*Ne + 1
    # domain boundarys [H, B]
    H = 1
    B = 0
    # length of element
    h = (H-B)/(N-1)
    # nodes
    z = collect(B:h:H)
    zfine = collect(B:.01:H)
    # number basis functions
    Nbasis = p + 1
    
    #---- initial and boundary data ----#
    # surface temperature
    Tsurf = -.05
    # inital base temperature
    Tbase = 1
    # initial temperature data
    T = zeros(N, 2)
    T[:, 2] = initial_temp.(z)
    # test parabola: -1.05/B^2 .* (z .- B).^2 .+ 1
    
    # initial porosity    
    ϕ = zeros(N, 2)
    ϕ[:, 2] = initial_pore.(z)
    
    # compaction pressure
    Pc = zeros(N)
    
    #---- building discrete operators ----#
    # NOTE: knowing the non-zero patterns are the same
    # for the matrices I could be generating one large
    # matrix by doing operations on the values then constructing
    
    # generate diffusion (second derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]
    assemble_matrix!(Ne, Nbasis, p,
                     z, dlb, dlb, one,
                    I, J, Vdiff)
    K = sparse(I, J, Vdiff, N, N)

    # generate advective (first derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vadv = Float64[]
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, dlb, u,
                     I, J, Vadv)
    S = sparse(I, J, Vadv, N, N)
    
    # generate mass operator matrix (no derivatives)
    I = Int64[]
    J = Int64[]
    Vmass = Float64[]
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, lb, one,
                     I, J, Vmass)
    M = sparse(I, J, Vmass, N, N)

    
    # generate dissipation source term in cold region
    F = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, z, lb, a, one, F)

    #=
    # cold steady state test
    F = Pe_inv .* F
    A = S + Pe_inv .* K
    enforce_dirchlet!(A, F, Tsurf, 1)
    T .= A\F
    plot(T, z)
    display(plot!(cold_steady_test.(z), z))
    =#


    # advective cfl
    Δt = h/abs(u(1))

    # get divide index
    Γ_idx = partition_temp_cold(T[:,2])

    params = (z = z,
              K = K,
              S = S,
              M = M,
              F = F,
              Tsurf = Tsurf,
              Γ_idx = Γ_idx,
              Pe_inv = Pe_inv,
              δ = δ,)

    for i = 0:500
        timestep!(T, ϕ, Pc, params, Δt)
    end

    nothing
     
end
