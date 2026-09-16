using ForwardDiff
using SparseArrays
using LinearAlgebra
using Statistics
using DataStructures
using FastGaussQuadrature

function get_mesh(Ne, p, B, L, N, he, ref_nodes, DG)

    mesh = zeros(N)
    if DG
        sref = (he .* ref_nodes)/2
        cref = B - sref[1]
        href = sref[end] - sref[1]
        for e in 0:Ne-1
            mesh[e*(p+1) + 1 : (e+1) * (p+1)] = cref .+ e .* href .+ sref
        end

    else
        for e in 0:Ne-1
            bidx = e*p
            for i in 1:p
                mesh[bidx + i] = (he*ref_nodes[i] + he*(e + 1) + he*e)/2
            end
        end
        mesh[end] = L
    end
    return mesh
end

#=
gaussian integration of funcs multiplied together with args for each function
weights and abscissa pulled from: https://pomax.github.io/bezierinfo/legendre-gauss.html
element - list of at least the start and end nodes of the element to integrate over
=#
function gauss_integrate(element, p, type, funcs...)

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
    scale = (element[end] - element[1]) * .5
    c = (element[end] + element[1]) * .5
    for l in 1:length(weights)
        val += weights[l] * 
            reduce(*, [f(scale * abscissa[l] + c) for f in funcs])
    end
    return scale *  val
end

function precompute_local_vec(Nbasis, p, nodes, func1)
    
    k_e = zeros(Nbasis)
    for i in 1:Nbasis
        k_e[i] = gauss_integrate(nodes, p ,1,
                                 x -> func1(x, i, nodes)
                                 )
    end
    return k_e
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


function assemble_global_static_vec_from_local_vec!(Ne, Nbasis, p, g, t_e, F, addition)

    if !addition
        F[:] .= 0
    end
    
    for e in 1:Ne
        idx = EToN(e,p)
        for i in 1:Nbasis
            F[idx[i]] += g * t_e[i]
        end
    end
end


function assemble_global_from_local_static_mat!(Ne, Nbasis, p, g, t_e, M, addition)

    if !addition
        M[:, :] .= 0
    end
    
    for e in 1:Ne
        idx=EToN(e, p)
        for i in 1:Nbasis, j in 1:Nbasis
            M[idx[i], idx[j]] += g * t_e[i,j]
        end
    end
end

function assemble_global_vec_from_local_mat!(Ne, Nbasis, p, g, t_e, F)

    F[:] .= 0
    for e in 1:Ne
        idx=EToN(e, p)
        glocal = @view g[idx]
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
# NOTE: NEEDS MAG JACOBIAN FOR NON-UNIFORM MESH
function assemble_global_from_local_tensor!(Ne, Nbasis, p, g, t_e, V::Vector{Float64})

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
                                            M::SparseMatrixCSC{Float64, Int64}, addition)
    if !addition
        M[:] .= 0
    end
    
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


function assemble_matrix!(Ne, Nbasis, p,
                          x, func1, func2, k,
                          I, J, V, DG)
    
    for e in 1:Ne
        for i in 1:Nbasis
            row = DG ? ((p+1)*(e-1) + i) : (p*e) + (i-p)
            for j in 1:Nbasis
                col = DG ? ((p+1)*(e-1) + j) : (p*e) + (j-p)
                nodes = DG ? x[(e - 1)*Nbasis+1:e*Nbasis] : EToX(e, p, x)
                v = gauss_integrate(nodes, p, 1, x -> func1(x, i, nodes) , x ->  func2(x, j, nodes), k)
                add_to_V!(I, J, V, v, row, col)
            end
        end
    end
end


function add_to_V!(I, J, V, v, row, col)
    
    idx = inCOO(I, J, row, col)
    if idx > 0 
        V[idx] += v
    else
        push!(I, row)
        push!(J, col)
        push!(V, v)
    end
end

function assemble_advective_flux!(Ne, Nbasis, p, z, I, J, u, advFlux, inflow, ϕbase, Tsurf)

    u = u(.5)

    for e in 1:Ne
        
        if e != 1
            # left faces of every element 
            #   idxL -> <- idxR
            # ---------*---------*---------
            #  e - 1      e       e + 1
            idxR = Nbasis * (e-1) + 1
            idxL = Nbasis * (e-1)

            add_to_V!(I, J, advFlux, -(1/2 * u - 1/2 * abs(u)), idxR, idxR)
            add_to_V!(I, J, advFlux, -(1/2 * u + 1/2 * abs(u)), idxR, idxL)
        end
        
        if e != Ne

            # right faces of every element
                      #   idxL -> <- idxR
            # ---------*---------*---------
            #  e - 1      e       e + 1
            
            idxL = Nbasis * (e)
            idxR = idxL + 1

            add_to_V!(I, J, advFlux, 1/2 * u + 1/2 * abs(u), idxL, idxL)
            add_to_V!(I, J, advFlux, 1/2 * u - 1/2 * abs(u), idxL, idxR)
        end

        # add outflow data
        if e == 1 && inflow
            add_to_V!(I, J, advFlux, -u, 1, 1)
        end
        if e == Ne && !inflow
            add_to_V!(I, J, advFlux, u, Ne*Nbasis, Ne*Nbasis)
        end
        
    end
end

function get_advective_boundary(N, Ne, Nbasis, u, inflow, Tsurf, ϕbase)
    u = u(.5)
    b = zeros(N)
    # u<0, inflow at z=1, data = Tsurf
    if inflow            
        b[Ne*Nbasis] = -u * Tsurf
    # u>0: inflow at z=0, data = ϕbase
    else
        b[1] = u * ϕbase
    end
    return b
end

function assemble_diffusive_flux!(Ne, Nbasis, p, z, I, J, Vdiff, k, σ, inflow, Tsurf, ϕbase)

    for e in 1:Ne

        # interior face shared by my element (m) and neighboring (n) element
        if e != 1
            
            # my and neighbor element global node indexing
            m_nodes = ((e - 2) * Nbasis + 1) : ((e - 1) * Nbasis)
            n_nodes = ((e - 1) * Nbasis + 1) : (e * Nbasis)
            
            # m and n element face index
            m_face = m_nodes[end]
            n_face = n_nodes[1]
            
            # m and n element coordinates
            z_m = z[m_nodes]
            z_n = z[n_nodes]
            
            # face coordinate
            z_face = z[(e - 1) * Nbasis + 1]

            # consistency: {dh/dz}[ψ]
            # for each side (m and n) this is -1/2 dh/dz (ψ = 1 only on the face),
            # which when adding to the matrix need the d/dz operator and h is
            # held in the vector meaning distrbute differentiation over the columns
            # and not rows. Distrubting out consistency gives 4 terms.
            
            for j in 1:Nbasis
                # left m face dependent on element m
                add_to_V!(I, J, Vdiff, - .5 *k * dlb(z_face, j, z_m), m_face, m_nodes[j])
                # left m face dependent on element n
                add_to_V!(I, J, Vdiff, - .5 *k * dlb(z_face, j, z_n), m_face, n_nodes[j])
                # right n element dependent on element n
                add_to_V!(I, J, Vdiff, .5 * k * dlb(z_face, j, z_m), n_face, m_nodes[j])
                # right n element dependent on element m
                add_to_V!(I, J, Vdiff, .5 * k * dlb(z_face, j, z_n), n_face, n_nodes[j])
            end

            # penalty: σ (h^m - h^n)
            # left face penalty coupling
            add_to_V!(I, J, Vdiff, σ, m_face, m_face)
            add_to_V!(I, J, Vdiff, -σ, m_face, n_face)
            # right face penalty coupling
            add_to_V!(I, J, Vdiff, σ, n_face, n_face)
            add_to_V!(I, J, Vdiff, -σ, n_face, m_face)

            # adjoint consistency: {dψ/dz}[h]
            # this is the opposite of consistency in the sense that we only pull have from
            # the single face node, but the derivative of the test function is distributed
            # over the rows

            for i in 1:Nbasis
                # left m element dependent on face m
                add_to_V!(I, J, Vdiff, - .5 * k * dlb(z_face, i, z_m), m_nodes[i], m_face)
                # left m element dependent on face n
                add_to_V!(I, J, Vdiff, .5 * k * dlb(z_face, i, z_m), m_nodes[i], n_face)
                # left n element dependent on face m
                add_to_V!(I, J, Vdiff, - .5 * k * dlb(z_face, i, z_n), n_nodes[i], m_face)
                # right n element dependent on face n
                add_to_V!(I, J, Vdiff, .5 * k * dlb(z_face, i, z_n), n_nodes[i], n_face)
            end
        end

        # dirichlet boundary faces
        if e == 1 && !inflow
            
            m_nodes = 1 : Nbasis
            m_z = z[m_nodes]
            f = m_nodes[1]
            
            for j in 1:Nbasis
                d = dlb(z_f, j, z_B)
                # consistency
                add_to_V!(I, J, Vdiff, k  * d, f, B_nodes[j])
                # adjoint consistency
                add_to_V!(I, J, Vdiff, k  * d, B_nodes[j], f)
            end
            add_to_V!(I, J, Vdiff, σ, f, f)
        end
        
        if e == Ne && inflow
            
            m_nodes = ((Ne-1)*Nbasis + 1) : (Ne*Nbasis)
            m_z = z[m_nodes]
            f = m_nodes[end]
            
            for j in 1:Nbasis
                d = dlb(z_f, j, m_z)
                # consistency
                add_to_V!(I, J, Vdiff, -k * d, f, m_nodes[j])
                # adjoint consistency
                add_to_V!(I, J, Vdiff, -k * d, m_nodes[j], f)
            end
            # penalty
            add_to_V!(I, J, Vdiff, σ, f, f)
        end
    end
end

# Assemble the full DG diffusion matrix K = (block volume stiffness) + (SIPG faces).
function get_diffusion_matrix_DG(Ne, Nbasis, p, z, k, N, σ, inflow, Tsurf, ϕbase)
    I = Int64[]
    J = Int64[]
    Vdiff = Float64[]
    
    assemble_matrix!(Ne, Nbasis, p, z, dlb, dlb, k, I, J, Vdiff, true)
    assemble_diffusive_flux!(Ne, Nbasis, p, z, I, J, Vdiff, k, σ, inflow, Tsurf, ϕbase)
    K = sparse(I, J, Vdiff, N, N)
    
    return K
end

function get_diffusion_boundary(N, Ne, Nbasis, p, z, k, σ, inflow, Tsurf, ϕbase)
    
    b = zeros(N)

    surf_nodes = (Ne - 1) * Nbasis + 1 : Ne * Nbasis
    surf_z = z[surf_nodes]
    surf_face = z[Ne * Nbasis]

    for i in 1:Nbasis
        b[surf_nodes[i]] = -k * dlb(surf_face, i, surf_nodes) * Tsurf
    end
    
    b[Ne * Nbasis] =+ σ * Tsurf

    if !inflow
        
        base_nodes = 1 : Nbasis
        base_z = z[base_nodes]
        base_face = z[1]

        for i in 1:Nbasis
            b[base_nodes[i]] = k * dlb(base_face, i, base_nodes) * ϕbase
        end

        b[1] =+ σ * ϕbase
        
    end
    
    return b
end

function assemble_compaction_flux!(Ne, Nbasis, p, z, I, J, Vpc, ϕ, α, σ, Pcbase)
    for e in 1:Ne
        if e != 1
            
            m_nodes = ((e-2)*Nbasis + 1) : ((e-1)*Nbasis)
            n_nodes = ((e-1)*Nbasis + 1) : ( e   *Nbasis)
            z_f     = z[(e-1)*Nbasis + 1]

        end
        
        # Pcbase boundary
        if e == 1
        end
    end
end

function get_compaction_boundary(N, Ne, Nbasis, p, z, ϕ, α, κ, δ, σ, Pcbase)
    b = zeros(N)
    return b
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
NNZDG(Ne, Nbasis) = Ne * (Nbasis)^2

function get_temp(H, T_m)
    return min.(T_m, H)
end

function get_porosity(H, T_m)
    return max.(T_m, H)
end

function get_temperate_ops(N)
                           
    Kϕ = spzeros(N,N)
    Mϕ = spzeros(N,N)
    Mχ = spzeros(N,N)
    Fϕ = zeros(N)
    return Kϕ, Mϕ, Mχ, Fϕ
    
end

function update_ϕ_ops!(Γ, ϕ, Nt, params, t_ops)

    ϕtemp = ϕ .+ params.ϵp
    
    assemble_global_from_local_tensor!(Γ, params.Nbasis, params.p,
                                       ϕtemp.^(params.α), t_ops.kt, t_ops.Kϕ, false)
    assemble_global_from_local_tensor!(Γ, params.Nbasis, params.p,
                                       ϕtemp, t_ops.mt, t_ops.Mϕ, false)
    assemble_global_vec_from_local_mat!(Γ, params.Nbasis, params.p,
                                        ϕtemp.^(params.α), t_ops.dm, t_ops.Fϕ)
end
    
function update_enthalpy_ops!(Γ, Nt, Pc, params, t_ops, g_ops)

    # reintegrate the compaction on the temperate side
    assemble_global_from_local_tensor!(Γ, params.Nbasis, params.p,
                                       Pc, t_ops.mt, g_ops.Q, false)
    
    if params.SUPG

        # add supg mass matrix to global mass matrix
        assemble_global_from_local_static_mat!(Γ, params.Nbasis, params.p,
                                          params.τ * params.u(.5),
                                          t_ops.dm, g_ops.Msupg, false)
        # add supg stiffness S_supg matrix to Q
        assemble_global_from_local_static_mat!(Γ, params.Nbasis, params.p,
                                               params.τ * params.u(.5) * params.u(.5),
                                               t_ops.km, g_ops.Q, true)
        
        # add supg compaction M_pe_supg matrix to Q
        assemble_global_from_local_tensor!(Γ, params.Nbasis, params.p, params.τ * params.u(.5)/params.η * Pc, t_ops.st,
                                           g_ops.Q, true)
        
        # add supg forcing to global F vector
        assemble_global_static_vec_from_local_vec!(Γ, params.Nbasis, params.p,
                                                   params.τ * params.u(.5) * params.a(.5),
                                                   t_ops.sv, g_ops.Fsupg, false)
    end
    # add on the diffusion on the cold side
    g_ops.Q[Nt:end, Nt:end] += g_ops.Kc[Nt:end, Nt:end]
end

function update_reg_ϕ_ops!(ϕ, χ, params, t_ops)
    
    ϕpos = max.(ϕ, 0.0)
    
    assemble_global_from_local_tensor!(params.Ne, params.Nbasis, params.p,
                                       ϕpos.^(params.α), t_ops.kt, t_ops.Kϕ, false)
    assemble_global_from_local_tensor!(params.Ne, params.Nbasis, params.p,
                                       ϕpos, t_ops.mt, t_ops.Mϕ, false)
    assemble_global_from_local_tensor!(params.Ne, params.Nbasis, params.p,
                                       (1 .- χ), t_ops.mt, t_ops.Mχ, false)
    assemble_global_vec_from_local_mat!(params.Ne, params.Nbasis, params.p,
                                        ϕpos.^(params.α), t_ops.dm, t_ops.Fϕ)
end

function update_dg_ϕ_ops!(ϕ, χ, params, t_ops)
    
    Ne = params.Ne
    Nbasis = params.Nbasis
    p = params.p
    z = params.z
    N = params.N
    α = params.α
    κ = params.κ
    δ = params.δ
    Pcbase = params.Pcbase

    # penalty parameter
    σ = 10.0 * p^2 / (z[Nbasis+1] - z[1])

    ϕpos = max.(ϕ, 0.0)

    # reuse regularized volume operators for Pc
    update_reg_ϕ_ops!(ϕ, χ, params, t_ops)

    # add SIPG face terms on
    I = Int64[]
    J = Int64[]
    Vpc = Float64[]
    
    assemble_compaction_flux!(Ne, Nbasis, p, z, I, J, Vpc, ϕpos, α, σ, Pcbase)
    t_ops.Kϕ = t_ops.Kϕ .+ sparse(I, J, Vpc, N, N)

    t_ops.bpc = get_compaction_boundary(N, Ne, Nbasis, p, z, ϕpos, α, κ, δ, σ, Pcbase)
end

function update_reg_ethalpy_ops!(H, Pc, params, t_ops, g_ops)


    χ = params.χ.(H)
    u = params.u(.5)
    τ = params.τ
    η = params.η

    # conduction
    assemble_global_from_local_tensor!(params.Ne, params.Nbasis, params.p,
                                       (1 .- χ), t_ops.kt, g_ops.Q, false)
    # melt/compaction
    assemble_global_from_local_tensor!(params.Ne, params.Nbasis, params.p,
                                       (χ.^2 .* Pc) ./ η, t_ops.mt, g_ops.Q, true)

    if params.SUPG
        # supg mass
        assemble_global_from_local_tensor!(params.Ne, params.Nbasis, params.p,
                                           (τ*u) .* χ, t_ops.st, g_ops.Msupg, false)
        # supg streamline
        assemble_global_from_local_tensor!(params.Ne, params.Nbasis, params.p,
                                           (τ*u*u) .* χ, t_ops.kt, g_ops.Q, true)
        # supg melt
        assemble_global_from_local_tensor!(params.Ne, params.Nbasis, params.p,
                                           (τ*u/η) .* (χ.^2 .* Pc), t_ops.st, g_ops.Q, true)
        # supg forcing
        assemble_global_vec_from_local_mat!(params.Ne, params.Nbasis, params.p,
                                            (τ*u*params.a(.5)) .* χ, t_ops.dm, g_ops.Fsupg)
    end
end

function get_lumped_mass(Ne, Nbasis, p, z, N, DG)
    
    I = Int64[]
    J = Int64[]
    Vmass = Float64[]
    diag = zeros(N)
    assemble_matrix!(Ne, Nbasis, p,
                     z, lb, lb,
                     one,
                     I, J, Vmass, DG)

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

function get_advection_matrix(Ne, Nbasis, p, z, u, N, DG, inflow, ϕbase, Tsurf)
    # generate advective (first derivative) operator matrix
    I = Int64[]
    J = Int64[]
    Vadv = Float64[]
    if !DG
        assemble_matrix!(Ne, Nbasis, p,
                         z, lb, dlb, u,
                         I, J, Vadv, DG)

        S = sparse(I, J, Vadv, N, N)
        
    else
        assemble_matrix!(Ne, Nbasis, p,
                         z, dlb, lb, u,
                         I, J, Vadv, DG)

        S = sparse(I, J, Vadv, N, N)
        
        I = Int64[]
        J = Int64[]
        advFlux = Float64[]
        assemble_advective_flux!(Ne, Nbasis, p, z, I, J, u, advFlux, inflow, ϕbase, Tsurf)
        Sflux = sparse(I, J, advFlux, N, N)
        S .= S .- Sflux
        
    end

    return S
end
