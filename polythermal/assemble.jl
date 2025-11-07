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

    end
    #=    
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
        =#
        
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
Assembles a local over the reference element tensor with ψ_iψ_jψ_k, where ψ is lag or derivative of.
=#
function precompute_local_tensor(Nbasis, p, nodes, func1, func2, func3)
    # k_e[i,j,k] = ∫ φ_i φ_j φ_k dx on the reference element
    k_e = zeros(Nbasis, Nbasis, Nbasis)
    for i in 1:Nbasis, j in 1:Nbasis, k in 1:Nbasis
        k_e[i,j,k] = gauss_integrate(
            nodes, p, 1,
            x -> func1(x, i, nodes),
            x -> func2(x, j, nodes),
            x -> func3(x, k, nodes)
        )
    end
    k_e = reshape(k_e, Nbasis^2, Nbasis)
    return k_e
end

function get_sparsity(Ne, nnz, Nbasis, p)

    I = zeros(nnz)
    J = zeros(nnz)
    c = 1
    for e in 1:Ne
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            for j in 1:Nbasis
                col = (p*e) + (j-p)
                if i != Nbasis && j != Nbasis
                    I[c] = row
                    J[c] = col
                    c += 1
                end
            end
        end
    end
    return I, J
end

#=
Takes local element tensor and contracts to matrix with Σ_k g_k int(ψ_iψ_jψ_k)
where int(...) comes from assemble_local_tensor, and places entries into global matrix. that is g is length n
=#
function assemble_global_from_local_tensor!(Ne, nnz, Nbasis, p, g, t_e, V)

    c = 1
    for e in 1:Ne
        idx=EToN(e, p)
        glocal = @view g[idx]
        # do flattened tensor multiple giving flattened local 2d matrix
        k_e = t_e * glocal
        k_e = reshape(k_e, (Nbasis,Nbasis))
        for i in 1:Nbasis, j in 1:Nbasis
            # at starting element add to last element index,
            # because they are the same
            V[c] += k_e[i,j]
            c += 1
        end
        c -= 1
    end
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

function enforce_dirchlet!(A, F, v, index)
    A[index, index] = 1.0
    if index != 1
        A[index, 1:index-1] .= 0
    end
    if index != size(A)[2]
        A[index, index+1:end] .= 0
    end
    F[index] = v
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
NNZ(Ne, Nbasis) = Ne * (Nbasis)^2 - Ne + 1

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

function get_compaction_ops_temp(Ne, Nbasis, p, z, ϕ, α, ops)

    N = p*Ne + 1
    ϕ = ϕ .+ .0000001
    
    ϕα = ϕ.^α

    I = ops.PeI
    J = ops.PeJ
    ke = ops.ke
    me = ops.me
    
    # generate diffusion (second derivative) operator matrix
    Vdiff = zeros(nnz)
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

function get_enth_ops(Ne, N, Γ, Nt, Nbasis, p, z, u, a, Pc)

    Pcinterp = Val -> expansion(Val, p, Pc, z)
    Γc = Ne - Γ

    # generate mass operator matrix (no derivatives)
    I = Int64[]
    J = Int64[]
    Vmass = Float64[]
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, lb, one,
                     I, J, Vmass)
    #@show length(Vmass)
    #error()
    M = sparse(I, J, Vmass, N, N)

    
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
    assemble_matrix!(Γ, Nbasis, p,
                     z, lb, lb,
                     Pcinterp,
                     I, J, Vmass)
    Mpc = sparse(I, J, Vmass, N, N)

    # generate diffusion (second derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]
    assemble_matrix!(Γc, Nbasis, p,
                     z, dlb, dlb, one,
                     I, J, Vdiff)

    # this is tricky, its just moving the indices to the cold region,
    # but we generated with indices starting at 0 in temperate region
    I = I .+ (Nt - 1)
    J = J .+ (Nt - 1)
    K = sparse(I, J, Vdiff, N, N)

    
    Q = Mpc + K

    # melting source term
    F = zeros(N)
    assemble_forcing!(Ne, Nbasis, p, z, lb, a, one, F)

    return Q, S, M, Mlump, F
    
end


function get_lumped_mass(Ne, Nbasis, p, z, N)
    
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

    return Mlump
end

function get_diffusion_matrix(Γc, Nt, Nbasis, p, z, N)
    # generate diffusion (second derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]
    assemble_matrix!(Γc, Nbasis, p,
                     z, dlb, dlb, one,
                     I, J, Vdiff)

    # this is tricky, its just moving the indices to the cold region,
    # but we generated with indices starting at 0 in temperate region
    I = I .+ (Nt - 1)
    J = J .+ (Nt - 1)
    K = sparse(I, J, Vdiff, N, N)
    return K
end

function get_advection_matrix(Ne, Nbasis, p, z, u, N)
    # generate advective (first derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vadv = Float64[]
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, dlb, u,
                     I, J, Vadv)
    S = sparse(I, J, Vadv, N, N)
    
    return S
end
