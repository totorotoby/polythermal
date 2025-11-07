using Plots
include("assemble.jl")

# number of elements
Ne = 3
# basis order
p = 2
# number basis functions per element
Nbasis = p+1
# number of nodes
N = p*Ne + 1
# domain boundarys [L, B]
L = 1.0
B = 0
# length between nodes
h = (L-B)/(N-1)
# spatial varying "test" function
gf(x) = x^.2#lb(x, 1, [0, h, 2*h])
# domain
x = 0:h:1
# function vector
g = gf.(x)
#--- New version ---#
I, J = get_sparsity(Ne, Nbasis, p)
nnz = length(I)
V = zeros(nnz)
k_e = precompute_local_tensor(Nbasis, p, [-1, 0, 1], lb, lb, lb)
@time assemble_global_from_local_tensor!(Ne, nnz, Nbasis, p, g, k_e, V)

#--- old version ---#

@time begin
    I_old = Int64[]
    J_old = Int64[]
    V_old = Float64[]
    gf_expan = Val -> expansion(Val, p, g, x)
    assemble_matrix!(Ne, Nbasis, p, x, lb, lb, gf_expan, I_old, J_old, V_old) 
end

#display(V)
#display(V_old)
#display(V - V_old)
display(V ./ V_old)


@assert I ≈ I_old
@assert J ≈ J_old
@assert V ≈ V_old



nothing
