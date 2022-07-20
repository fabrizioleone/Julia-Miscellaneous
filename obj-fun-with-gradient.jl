# Call Packages
using Optim, BenchmarkTools

# Parameters
x0 = [0.1, 0.1]
a  = [1.0, 3.5]
b  = [0.1, 0.5]
lb = [0.0, 0.0]
ub = [10.0, 10.0]

# Objective fun
function objfun(x::Vector, a::Vector, b::Vector) 
    f = sum((2*x .- a.*b).^2)
    return f
end

# Gradient
function grafun(x::Vector, a::Vector, b::Vector)
    G = 4 * (2*x .- a.*b)
    return G
end

# Objective fun & gradient together
function fg!(F, G, x::Vector, a::Vector, b::Vector)
    
    # Common calculations
    common = (2*x .- a.*b)

    # Define gradient
    if G != nothing
      grad = 4 * common
      [G[i] = grad[i] for i in 1:length(grad)]
    end

    # Define objective function
    if F != nothing
      F = sum(common.^2)
      return F
    end

end

# Check that all methods yield the same solution
hcat(optimize(x -> objfun(x,a,b), x0).minimizer, 
     optimize(x -> objfun(x,a,b), x -> grafun(x,a,b), x0; inplace = false).minimizer,
     optimize(Optim.only_fg!((F, G, x) -> fg!(F, G, x, a, b)), x0).minimizer)

# Benchmark
#@benchmark optimize(x -> objfun(x,a,b), x0; autodiff = :forward)
#@benchmark optimize(x -> objfun(x,a,b), x -> grafun(x,a,b), x0; inplace = false)
#@benchmark optimize(Optim.only_fg!((F, G, x) -> fg!(F, G, x, a, b)), [0., 0.], LBFGS())
#@benchmark optimize(Optim.only_fg!((F, G, x) -> fg!(F, G, x, a, b)), [0., 0.], GradientDescent())
#@benchmark optimize(Optim.only_fg!((F, G, x) -> fg!(F, G, x, a, b)), [0., 0.], ConjugateGradient())

    
