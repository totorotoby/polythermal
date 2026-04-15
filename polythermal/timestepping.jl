using Smoothing
using Arpack
using Printf
include("assemble.jl")
include("sol_tests.jl")


# regularization function
function χfunc(H, params)
    return .5 .* (1 .+ tanh.(H ./ params.ϵ))
end

function χpfunc(H, params)
    return params.ϵp .+ (1 .- params.ϵp) .* χfunc(H, params)
end

function timestep(H, Pc, params, eops, Δt)

    p = params.p
    Tsurf = params.Tsurf
    ϕbase = params.ϕbase
    z = params.z
    inflow = params.inflow
    implicit = params.implicit
    
    #--- new solver ---#

    # do enthalpy either implicitly
    if implicit == true
        picard!(H, Pc,
                params, eops,
                z, inflow,
                Δt, Tsurf, ϕbase,
                1e-8, 200)
    else
        # or explicitly
        H[:] = RK4(H, Δt, Nt, params, g_ops, enthalpy_rhs)
    end
        
    #--- re-partition ---#
    T = get_temp(H, 0.0)
    ϕ = get_porosity(H, 0.0)
    update_ϕ_ops!(ϕ, Nt, params, t_ops)

    plot(Pc[1:Nt], z[1:Nt], label="Pc")
    display(plot!(H[:], z, label="H"))
    
    return (H, Pc)

end

function solve_Pc!(Pc, params, eops)

    κ = params.κ
    δ = params.δ
    η = params.η
    g = params.g
    Pcbase = params.Pcbase

    enforce_dirchlet!(A, R, Pcbase, 1)
    Pc .= A\R
    
end

function eval_Hχϕ!(H, eops, params)

    z = params.z
    p = params.p
    Ne = params.Ne
    Nq = params.Nq
    Nbasis = params.Nbasis
    Hq = eops.Hq
    χq = eops.χq
    ϕq = eops.ϕq
    B = eops.B

    for e in 1:Ne
        idx = EToN(e, p)
        Hloc = H[idx]
        Hqloc = B * Hloc
        Hq[(e-1)*Nq + 1: e*Nq] .= Hqloc
    end

    χq = χfunc(Hq, params)
    ϕq = χq .* Hq

    #plot(H, z, label="H")
    plot(Hq, χq, label="χ")
    display(plot!(Hq, ϕq, label="ϕ"))
    quit()
    
end

#explict timestepping for enthalpy method
function enthalpy_rhs(h, inflow, ϕbase, Nt, g_ops)

    Q = g_ops.Q
    S = g_ops.S
    Mlump = g_ops.Mlump
    F = g_ops.F

    A = Mlump * (- S - Q)
    RHS = A * h + Mlump * F
    RHS[end] = 0.0
    if inflow == true
        RHS[Nt] = 0.0
    else
        RHS[1] = ϕbase
    end
        
    return RHS
end

function RK4(u, Δt, Nt, params, g_ops, rhs)

    k1 = Δt * rhs(u[:], params.inflow, params.ϕbase, Nt, g_ops)
    k2 = Δt * rhs(u[:] + k1/2, params.inflow, params.ϕbase, Nt, g_ops)
    k3 = Δt * rhs(u[:] + k2/2, params.inflow, params.ϕbase, Nt, g_ops)
    k4 = Δt * rhs(u[:] + k3, params.inflow, params.ϕbase, Nt, g_ops)

    u_raw = u[:] + (k1 + 2k2 + 2k3 + k4) / 6
    u_smooth = Smoothing.binomial(u_raw, 1)

    return u_raw
    
end

function picard!(H, Pc, params, eops,
                 z, inflow, Δt, Tsurf, ϕbase,
                 tol, maxiter)

    z = params.z
    
    eps = 1e-12
    iter = 0
    err = Inf

    # get quadtrature enthalpy
    eval_Hχϕ!(H, eops, params)
    solve_Pc!(Pc, params, eops)
    
    #picard loop
    while err > tol && iter < maxiter
        # get previous iteration k
        H_prev .= H_iter
        Pc_prev .= Pc_iter

        # compute new H and Pc k+1
        T = get_temp(H_iter, 0.0)
        ϕ = get_porosity(H_iter, 0.0)
        update_ϕ_ops!(ϕ, params, t_ops)
        solve_Pc!(Pc_iter, params, t_ops)
        update_enthalpy_ops!(Pc_iter, params, t_ops, g_ops)
        Q_new = g_ops.Q

        if params.SUPG
            Msupg = g_ops.Msupg
            Fsupg = g_ops.Fsupg
            A = (M + Msupg) + (Δt/2) * (S + Q_new)
            R = ((M + Msupg) - (Δt/2) * (S + Q_old)) * H_old + Δt * (F + Fsupg)
        else
            A = M + (Δt/2) * (S + Q_new)
            R = (M - (Δt/2) * (S + Q_old)) * H_old + Δt * F
        end
        
        # set boundary conditions
        enforce_dirchlet!(A, R, Tsurf, size(A,1))
        if inflow
            enforce_dirchlet!(A, R, 0.0, Nt)
        else
            enforce_dirchlet!(A, R, ϕbase, 1)
        end
        H_iter .= A \ R

        # convergence check
        err_H  = norm(H_iter - H_prev) / (norm(H_prev) + eps)
        err_Pc = norm(Pc_iter - Pc_prev) / (norm(Pc_prev) + eps)
        err = max(err_H, err_Pc)
        iter += 1
    end

    if iter == maxiter
        @printf "Maximum non-linear iterations excited\n"
    end
        
    H .= H_iter
    Pc .= Pc_iter

end
