using ForwardDiff
using SparseArrays
using LinearAlgebra
using Statistics
using DataStructures
using FastGaussQuadrature

function get_mesh(Ne, p, L, N, he, ref_nodes)
    
    mesh = zeros(N)
    for e in 0:Ne-1
        bidx = e*p
        for i in 1:p
            mesh[bidx + i] = (he*ref_nodes[i] + he*(e + 1) + he*e)/2
        end
    end
    mesh[end] = L
    return mesh
    
end


#=
gaussian integration of funcs multiplied together with args for each function
weights and abscissa pulled from: https://pomax.github.io/bezierinfo/legendre-gauss.html
element - list of at least the start and end nodes of the element to integrate over
=#
function gauss_integrate(bounds, p, type, funcs...)

    weights = nothing
    if type == 1
        weights = [ 0.1894506104550685,	
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
                    0.0271524594117541 ]


        abscissa = [-0.0950125098376374,
                    0.0950125098376374, 
                    -0.2816035507792589,
                    0.2816035507792589, 
                    -0.4580167776572274,
                    0.4580167776572274, 
                    -0.6178762444026438,
                    0.6178762444026438, 
                    -0.7554044083550030,
                    0.7554044083550030, 
                    -0.8656312023878318,
                    0.8656312023878318, 
                    -0.9445750230732326,
                    0.9445750230732326, 
                    -0.9894009349916499,
                    0.9894009349916499]
    end
        
    val = 0.0
    scale = (bounds[end] - bounds[1]) * .5
    c = (bounds[end] + bounds[1]) * .5
    for l in 1:length(weights)
        val += weights[l] * 
            reduce(*, [f(scale * abscissa[l] + c) for f in funcs])
    end
    return scale *  val
end

function precompute_local_mat(Nbasis, p, nodes, func1, func2)
    # k_e[i,j] = ∫ φ_i φ_j dx on the reference element
    k_e = zeros(Nbasis, Nbasis)
    for i in 1:Nbasis, j in 1:Nbasis
        k_e[i,j] = gauss_integrate(
            nodes, p, 1,
            x -> func1(x, i, nodes),
            x -> func2(x, j, nodes)
        )
    end
    return k_e
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

    I = zeros(Int, nnz)
    J = zeros(Int, nnz)
    c = 1
    for e in 1:Ne
        for i in 1:Nbasis
            row = (p*e) + (i-p)
            for j in 1:Nbasis
                col = (p*e) + (j-p)
                if (i != Nbasis || j != Nbasis)
                    I[c] = row
                    J[c] = col
                    c += 1
                end
            end
        end
    end
    I[end] = (p*Ne) + 1
    J[end] = (p*Ne) + 1
    return I, J
end

function assemble_global_vec_from_local_mat!(Ne, Nbasis, p, g, t_e, F)

    F[:] .= 0
    for e in 1:Ne
        idx=EToN(e, p)
        glocal = @view g[idx]
        # do flattened tensor multiple giving flattened local 2d matrix
        k_e = t_e * glocal
        for i in 1:Nbasis
            # at starting element add to last element index,
            # because they are the same
            F[idx[i]] += k_e[i]
        end
    end
end
#=
Takes local element tensor and contracts to matrix with Σ_k g_k int(ψ_iψ_jψ_k)
where int(...) comes from assemble_local_tensor, and places entries into global matrix. that is g is length n
=#
function assemble_global_from_local_tensor!(
        Ne, Nbasis, p, g, t_e, V::Vector{Float64})

    c = 1
    for e in 1:Ne
        idx = EToN(e, p)
        glocal = @view g[idx]
        k_e = zeros(Nbasis, Nbasis)
        for i in 1:Nbasis, j in 1:Nbasis
            for k in 1:Nbasis
                k_e[i,j] += t_e[(i-1)*Nbasis + j, k] * glocal[k]
            end
        end
        for i in 1:Nbasis, j in 1:Nbasis
            V[c] += k_e[i,j]
            c += 1
        end
        c -= 1
    end
end

function assemble_global_from_local_tensor!(Ne, Nbasis, p, g, t_e,
                                            M::SparseMatrixCSC{Float64, Int64})

    M[:] .= 0
    for e in 1:Ne
        idx=EToN(e, p)
        glocal = @view g[idx]
        # do flattened tensor multiple giving flattened local 2d matrix
        k_e = zeros(Nbasis,Nbasis)
        for i in 1:Nbasis, j in 1:Nbasis
            for k in 1:Nbasis
                k_e[i,j] += t_e[(i-1)*Nbasis+j,k] * glocal[k]
            end
        end
        for i in 1:Nbasis, j in 1:Nbasis
            # at starting element add to last element index,
            # because they are the same
            M[idx[i], idx[j]] += k_e[i,j]
        end
    end
end


function assemble_interface!(Γ, H, z, Nbasis, p, Pc, mt, Q)

    Γ_nodes = EToN(Γ, p)
    z_nodes = EToX(Γ, p, z)
    
    # linear interp interface within element
    nb, ne = Γ_nodes[1], Γ_nodes[end]
    zb, ze = z[nb], z[ne]
    Δz = ze - zb
    h_frac = (0 - H[nb])/(H[ne] - H[nb])
    Γ_interior = zb + h_frac * Δz
    
    
    
    for i in 1:Nbasis, j in 1:Nbasis
        v = gauss_integrate([zb, Γ_interior],
                            p,
                            1,
                            z -> lb(z, i, z_nodes),
                            z -> lb(z, j, z_nodes),
                            Val -> expansion(Val, p, Pc, z))
        
    end
    
    @show Γ_interior
    @show z[nb], z[ne]
    quit()
    
    
end

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


function interpolate_lagrangian_global(x_nodes, u_nodes, p, x_plot)
    u_plot = zeros(length(x_plot))
    
    for (i, xp) in enumerate(x_plot)
        e, local_nodes = XToN(xp, p, x_nodes)
        
        local_inds = EToN(e, p)
        local_u = u_nodes[local_inds]
        
        val = 0.0
        for j in 1:length(local_nodes)
            val += local_u[j] * lb(xp, j, local_nodes)
        end
        u_plot[i] = val
    end
    
    return u_plot
end

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

EToX(e, p, nodes) = nodes[(e-1)*p + 1 : (e-1)*p + p + 1]
EToN(e, p) = (e-1)*p + 1 : (e-1)*p + p + 1
NNZ(Ne, Nbasis) = Ne * (Nbasis)^2 - Ne + 1

function get_temp(H, T_m)
    return min.(T_m, H)
end

function get_porosity(H, T_m)
    return max.(T_m, H)
end

function get_temperate_ops(Ne, N, nnzt, Nbasis, p,
                           ϕ, α, mt, kt, dm, It, Jt)

    ϕtemp = ϕ .+ .000001
    
    VKϕ = zeros(nnzt)
    VMϕ = zeros(nnzt)
    Fϕ = zeros(N)

    assemble_global_from_local_tensor!(Ne, Nbasis, p, ϕtemp, mt, VMϕ)
    assemble_global_from_local_tensor!(Ne, Nbasis, p, ϕtemp.^α, kt, VKϕ)    
    assemble_global_vec_from_local_mat!(Ne, Nbasis, p, ϕtemp.^α, dm, Fϕ)

    Kϕ = sparse(It, Jt, VKϕ, N, N)
    Mϕ = sparse(It, Jt, VMϕ, N, N)

    return Kϕ, Mϕ, Fϕ
    
end

function update_ϕ_ops!(Γ, ϕ, Nt, params, t_ops)

    ϕtemp = ϕ .+ .000001
    
    assemble_global_from_local_tensor!(Γ, params.Nbasis, params.p,
                                       ϕtemp.^(params.α), t_ops.kt, t_ops.Kϕ)
    assemble_global_from_local_tensor!(Γ, params.Nbasis, params.p,
                                       ϕtemp, t_ops.mt, t_ops.Mϕ)
    assemble_global_vec_from_local_mat!(Γ, params.Nbasis, params.p,
                                        ϕtemp.^(params.α), t_ops.dm, t_ops.Fϕ)
end
    
function update_Q!(H, Γ, Nt, Pc, params, t_ops, g_ops)

    #split element integration
    assemble_interface!(Γ, H, params.z, params.Nbasis,
                        params.p, Pc, t_ops.mt,
                        g_ops.Q)
    
    # reintegrate the compaction on the temperate side
    assemble_global_from_local_tensor!(Γ, params.Nbasis,
                                       params.p, Pc, t_ops.mt,
                                       g_ops.Q)
    
    # add on the diffusion on the cold side
    g_ops.Q[Nt:end, Nt:end] += g_ops.Kc[Nt:end, Nt:end]

end

function get_lumped_mass(Ne, Nbasis, p, z, N)
    
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
    M =  sparse(I, J, Vmass, N, N)
    
    return Mlump, M
end
    
function get_diffusion_matrix(Γc, Nt, Nbasis, p, z, N)
    # generate diffusion (second derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]
    assemble_matrix!(Γc, Nbasis, p,
                     z, dlb, dlb, one,
                     I, J, Vdiff)
    I = I .+ (Nt - 1)
    J = J .+ (Nt - 1)
    Kc = sparse(I,J, Vdiff, N,N)
    return Kc
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
