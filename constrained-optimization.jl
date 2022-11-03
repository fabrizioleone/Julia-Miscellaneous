# Call Packages
using Random, Optim

# Set seeds
Random.seed!(1704)

# Initialization
N    = 3
x0   = ones(N)
w    = 1.0
A    = rand(N, N)
a    = 2.0

# Define objective function: x = A * x
obj_fun(x, A) = sum(abs2.(x .- A * x))

# Define constraint: a * ∑ x 
con_c!(c, x, a) = (c[1] = a*sum(x))

# Define constraints on x 
lx   = fill(0, N)
ux   = fill(Inf, N)

# Define constraints on a * ∑ x 
lc   = [w]
uc   = [w]
# -> This way lc <= a * ∑ x <= uc --> ∑ x = w
# see also: https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#generic-nonlinear-constraints

# Run optimization
obj  = TwiceDifferentiable(x -> obj_fun(x, A), x0)
con  = TwiceDifferentiableConstraints((c,x) -> con_c!(c, x, a), lx, ux, lc, uc)
res  = optimize(obj, con, x0, IPNewton(); autodiff = :forward)

res.minimizer
