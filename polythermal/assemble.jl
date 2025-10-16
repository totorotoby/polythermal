using ForwardDiff
using SparseArrays
using LinearAlgebra
using Statistics
using DataStructures


#=
gaussian integration of funcs multiplied together with args for each function
weights and abscissa pulled from: https://pomax.github.io/bezierinfo/legendre-gauss.html
element - list of at least the start and end nodes of the element to integrate over
=#
function gauss_integrate(element, p, type, funcs...)

    weights = nothing
    # 4th Order normal gaussian
    if type == 1

        weights = [0.6521451548625461
                   0.6521451548625461
                   0.3478548451374538
                   0.3478548451374538]

        abscissa = [-0.3399810435848563
    	            0.3399810435848563
    	            -0.8611363115940526
    	            0.8611363115940526]

    # GLL from second to 5th order
    elseif type == 2
        if p == 1
            weights = [1.0 1.0]
            abscissa = [-1.0 1.0]
        elseif p == 2
            weights = [1/3 4/3 1/3]
            abscissa = [-1.0 0.0 1.0]
        elseif p == 3
            weights = [1/6 5/6
                       5/6 1/6]
            abscissa = [-1.0 -0.4472135954999579
                        0.4472135954999579 1.0]
        elseif p == 4
            weights = [1/10 49/90 32/45
                       49/90 1/10]
            abscissa = [-1.0 -0.6546536707079771 0.0
                        -0.6546536707079771 1]
        end
    end
        
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
                nodes = EToX(e, p, x)
                v = gauss_integrate(nodes, p, 1, x -> func1(x, i, nodes) , x ->  func2(x, j, nodes), k)
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
        nodes = EToX(e, p, x)
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            F[row] += gauss_integrate(nodes, p, 1, x -> func1(x, i, nodes), forcing, func2)
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
 Element to coordinates and nodes:
 e - element number
 p - order of basis
 nodes - list of global nodes
 returns nodes in element =#
EToX(e, p, nodes) = nodes[(e-1)*p + 1 : (e-1)*p + p + 1]
EToN(e, p) = (e-1)*p + 1 : (e-1)*p + p + 1

function get_temp(H, T_m)
    return min.(T_m, H)
end

function get_porosity(H, T_m)
    return max.(T_m, H)
end

function get_temperature_ops(Ne, Nbasis, p, z, u, a, α)

    #---- building discrete operators ----#
    # NOTE: knowing the non-zero patterns are the same
    # for the matrices I could be generating one large
    # matrix by doing operations on the values then constructing

    N = p*Ne + 1
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

    return K, S, M, F
    
end

# TODO: probably don't need to recompute all the gaussian integration here
# can probably just generate new diagonals to multiply K M and F by
function get_compaction_ops(Ne, Nbasis, p, z, ϕ, α)

    N = p*Ne + 1

    ϕ = ϕ .+ .000001
    
    ϕinterp = Val -> expansion(Val, p, ϕ, z)
    ϕαinterp = Val -> expansion(Val, p, ϕ.^α, z)

    #display(plot(ϕαinterp.(z), z))
    # generate diffusion (second derivative) operator matrix
    # generate diffusion (second derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]
    assemble_matrix!(Ne, Nbasis, p,
                     z, dlb, dlb, ϕαinterp,
                    I, J, Vdiff)
    
    Kϕα = sparse(I, J, Vdiff, N, N)
    
    # generate mass matrix with porosity integrated
    I = Int64[]
    J = Int64[]
    Vmass = Float64[]
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, lb,
                     ϕinterp,
                     I, J, Vmass)
    
    Mϕ = sparse(I, J, Vmass, N, N)
    
    # compation equation forcing
    Fϕα = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, z, dlb, ϕαinterp, one, Fϕα)

    return Kϕα, Mϕ, Fϕα, ϕαinterp
    
end

function get_porosity_ops(Ne, Nbasis, p, z, u, a, Pc)

    N = p*Ne + 1
    
    Pcinterp = Val -> expansion(Val, p, Pc, z)
    
    # generate lumped mass matrix 
    I = Int64[]
    J = Int64[]
    Vmass = Float64[]
    diag = zeros(N)
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, lb,
                     one,
                     I, J, Vmass)
    
    for nz = 1:length(I)
        diag[I[nz]] += Vmass[nz]
    end
    for i = 1:N
        diag[i] = 1/diag[i]
    end
    Mlump = spdiagm(0 => diag)

    # generate mass with Pc matrix 
    I = Int64[]
    J = Int64[]
    Vmass = Float64[]
    diag = zeros(N)
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, lb,
                     Pcinterp,
                     I, J, Vmass)
    Mpc = sparse(I, J, Vmass, N, N)

    # generate advective (first derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vadv = Float64[]
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, dlb, u,
                     I, J, Vadv)
    S = sparse(I, J, Vadv, N, N)

    # melting source term
    F = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, z, lb, a, one, F)

    return Mlump, Mpc, S, F
    
end

function get_enth_ops(Ne, Nbasis, p, z, u, a, Pc)

    Pcinterp = Val -> expansion(Val, p, Pc, z)
    N = p*Ne + 1
    
    # generate lumped mass matrix 
    I = Int64[]
    J = Int64[]
    Vmass = Float64[]
    diag = zeros(N)
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, lb,
                     one,
                     I, J, Vmass)
    
    for nz = 1:length(I)
        diag[I[nz]] += Vmass[nz]
    end
    for i = 1:N
        diag[i] = 1/diag[i]
    end
    Mlump = spdiagm(0 => diag)

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
    
    # generate mass with Pc matrix 
    I = Int64[]
    J = Int64[]
    Vmass = Float64[]
    diag = zeros(N)
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, lb,
                     Pcinterp,
                     I, J, Vmass)
    Mpc = sparse(I, J, Vmass, N, N)

    # melting source term
    F = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, z, lb, a, one, F)

    return K, S, Mpc, Mlump, F
    
end


#=
function restrict_cold(Ne, N, Nbasis, p, z, u)
    
    N = size(M)[1]

    Mn = M[1, :]
    Kn = K[1, :]
    Sn = S[1, :]
    Fn = F[1]
    
    M = M[Nt + 1:end,Nt + 1:end]
    K = K[Nt + 1:end,Nt + 1:end]
    S = S[Nt + 1:end,Nt + 1:end]
    F = F[Nt + 1:end]

    
    M[1, :] = Mn[1 : N - Nt]
    K[1, :] = Kn[1 : N - Nt]
    S[1, :] = Sn[1 : N - Nt]
    F[1] = Fn
    
    return M, K, S, F 
end
=#
