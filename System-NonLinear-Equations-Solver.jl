# Call Packages
using Random, Optim

# Set seeds
Random.seed!(1704)

# Initialization
N    = 3
x0   = ones(N)
A    = rand(N, N)
a    = [0.5, 0.4, 1.0]

# Define objective function: declare a scalar
obj_fun(x) = 1.0

# Define constraint
function con_c!(c, x, a) 
    c[1] = x[1]^2 - a[1]
    c[2] = log(x[2]) - a[2]
    c[3] = x[3]^3 - x[1] - a[3]
    c
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

# Display solution
@show res.minimizer
@show con_c!(fill(NaN, N), res.minimizer, a) 

