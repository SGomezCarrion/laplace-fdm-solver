module LaplaceSolver

using LinearAlgebra
using SparseArrays
using Plots
using LaTeXStrings

export solve_laplace_mixed_bcs

"""
    build_1d_operators(n::Int, h::Float64)

Constructs the 1D finite difference matrices for Dirichlet (D) and Neumann (N) 
boundary conditions using central differences.
"""
function build_1d_operators(n::Int, h::Float64)
    # Interior matrix
    L_D = spdiagm(n-1, n-1, -1 => -ones(n-2), 0 => 2*ones(n-1), 1 => -ones(n-2))
    
    # Base finite difference matrix
    D_h = spzeros(n+1, n+1)
    D_h[2,1] = -1
    D_h[n,n+1] = -1
    D_h[2:n, 2:n] = L_D
    
    # Dirichlet 1D Operator
    D = (1/(h^2)) * D_h
    D[1,1] = 1
    D[n+1,n+1] = 1
    
    # Neumann 1D Operator (using 2nd order Taylor approximation at boundaries)
    N_h = copy(D_h)
    N_h[1,1] = 2
    N_h[1,2] = -2
    N_h[n+1,n+1] = 2
    N_h[n+1,n] = -2
    N = (1/(h^2)) * N_h
    
    return D, N
end

"""
    build_boundary_indicators(n::Int)

Creates sparse diagonal indicator matrices to identify corners, 
Neumann regions, and Dirichlet regions along the boundary.
"""
function build_boundary_indicators(n::Int)
    b1 = ceil(Int64, ((n+1)/4) + 1)
    b2 = floor(Int64, 3*(n+1)/4)
    
    i = [b1:b2;]
    neumann = sparse(i, i, ones(length(i)), n+1, n+1)
    corner = sparse([1, n+1], [1, n+1], [1, 1], n+1, n+1)
    dirichlet = sparse(I, n+1, n+1) - neumann - corner
    
    return neumann, corner, dirichlet
end

"""
    assemble_2d_system(n::Int, uplus::Float64, uminus::Float64)

Assembles the full 2D sparse matrix A_h and the right-hand side vector f_h 
using Kronecker products for efficient memory allocation.
"""
function assemble_2d_system(n::Int, uplus::Float64, uminus::Float64)
    h = 1/n
    D, N = build_1d_operators(n, h)
    neumann, corner, dirichlet = build_boundary_indicators(n)
    
    # Central differences for Neumann boundaries
    D_central = (1/h^2) * spdiagm(1 => -ones(n), 0 => 2*ones(n+1), -1 => -ones(n))
    D_neumann = neumann * D_central
    
    # Assemble using Kronecker products
    A_x = kron(dirichlet, D) + kron(neumann, N) + kron(corner, D_neumann)
    A_y = kron(D, dirichlet) + kron(N, neumann) + kron(D_neumann, corner)
    A_corner = kron(corner, corner)
    
    A_h = A_x + A_y + A_corner
    
    # Assemble RHS vector
    b1 = ceil(Int64, ((n+1)/4) + 1)
    b2 = floor(Int64, 3*(n+1)/4)
    F_h = spzeros(n+1, n+1)
    
    for j in 1:b1-1
        F_h[1,j] = uplus; F_h[n+1,j] = uminus
        F_h[j,1] = uplus; F_h[j,n+1] = uminus
    end
    for j in b2+1:n+1
        F_h[1,j] = uminus; F_h[n+1,j] = uplus
        F_h[j,1] = uminus; F_h[j,n+1] = uplus
    end
    
    f_h = collect(reshape(F_h, (n+1)*(n+1)))
    
    return A_h, f_h, h
end

"""
    solve_laplace_mixed_bcs(; n::Int=100, uplus::Float64=1.0, uminus::Float64=-100.0)

Main routine to assemble, solve, and plot the 2D Laplace equation.
"""
function solve_laplace_mixed_bcs(; n::Int=100, uplus::Float64=1.0, uminus::Float64=-100.0)
    println("Assembling system for n = $n...")
    A_h, f_h, h = assemble_2d_system(n, uplus, uminus)
    
    println("Solving linear system...")
    u_vec = A_h \ f_h
    u_mat = reshape(u_vec, n+1, n+1)
    
    println("Generating plots...")
    x = 0:h:1
    y = 0:h:1
    
    p1 = surface(x, y, u_mat, xlabel="x", ylabel="y", zlabel=L"u(x,y)", 
                 title="Surface Plot", legend=false, camera=(45,45))
                 
    p2 = contour(x, y, u_mat, xlabel="x", ylabel="y", 
                 title="Contour Plot", legend=false)
                 
    plot(p1, p2, layout=(1, 2), size=(900, 450))
end

end # module