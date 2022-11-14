# Call Packages
using Random, NLsolve

# Set seeds
Random.seed!(1704)

# Initialization
N    = 3
x0   = ones(N)
A    = rand(N, N)
a    = [0.5, 0.4, 1.0]

# Define objective function
function obj_fun!(F, x, a) 
    F[1] = x[1]^2 - a[1]
    F[2] = log(x[2]) - a[2]
    F[3] = x[3]^3 - x[1] - a[3]
end

# Run optimization
res = nlsolve((F,x) -> obj_fun!(F, x, a), x0, autodiff = :forward)

# Display solution
@show res.zero

# For problems with constraints, see "Mixed complementarity problems"
# https://github.com/JuliaNLSolvers/NLsolve.jl 
