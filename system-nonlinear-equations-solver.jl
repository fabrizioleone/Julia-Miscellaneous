# Call Packages
using Random, Optim, NLsolve

# Set seeds
Random.seed!(1704)

# Initialization
N    = 3
x0   = ones(N)
A    = rand(N, N)
a    = [0.5, 0.4, 1.0]

#--------------------------------------------------------#
# Option 1: use Optim.jl
#--------------------------------------------------------#

# Define objective function: declare a scalar
obj_fun(x) = 1.0

# Define constraint
function con_c!(c, x, a) 
    c[1] = x[1]^2 - a[1]
    c[2] = log(x[2]) - a[2]
    c[3] = x[3]^3 - x[1] - a[3]
end

# Define constraints on x 
lx   = fill(-Inf, N)
ux   = fill(Inf, N)

# Define constraints on c
lc   = fill(0.0, N)
uc   = fill(0.0, N)

# Run optimization
obj  = TwiceDifferentiable(obj_fun, x0)
con  = TwiceDifferentiableConstraints((c,x) -> con_c!(c, x, a), lx, ux, lc, uc)
res  = optimize(obj, con, x0, IPNewton(); autodiff = :forward)

#--------------------------------------------------------#
# Option 1: use NLsolve.jl
#--------------------------------------------------------#

# Define objective function
function obj_fun!(F, x, a) 
    F[1] = x[1]^2 - a[1]
    F[2] = log(x[2]) - a[2]
    F[3] = x[3]^3 - x[1] - a[3]
end

# Run optimization
res1 = nlsolve((F,x) -> obj_fun!(F, x, a), x0, autodiff = :forward)

#--------------------------------------------------------#
# Display solution
#--------------------------------------------------------#
[res.minimizer res1.zero]
#@show con_c!(fill(NaN, N), res.minimizer, a) 
