
## Load the packages:
# -------------------

using SCS, JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools, DiffOpt, SparseArrays, KNITRO

## Load the pivoted Cholesky finder
# ---------------------------------
include("code_to_compute_pivoted_cholesky.jl")


## Some helper functions
# ======================

# construct e_i in R^n
function e_i(n, i)
    e_i_vec = zeros(n, 1)
    e_i_vec[i] = 1
    return e_i_vec
end

# this symmetric outer product is used when a is constant, b is a JuMP variable
function ⊙(a,b)
    return ((a*b') .+ transpose(a*b')) ./ 2
end

# this symmetric outer product is for computing ⊙(a,a) where a is a JuMP variable
function ⊙(a)
    return a*transpose(a)
end

# function to compute cardinality of a vector
function compute_cardinality(x, ϵ_sparsity)
    n = length(x)
    card_x = 0
    for i in 1:n
        if abs(x[i]) >=  ϵ_sparsity
            card_x = card_x + 1
        end
    end
    return card_x
end

# function to compute rank of a matrix
function compute_rank(X, ϵ_sparsity)
    eigval_array_X = eigvals(X)
    rnk_X = 0
    n = length(eigval_array_X)
    for i in 1:n
        if abs(eigval_array_X[i]) >= ϵ_sparsity
            rnk_X = rnk_X + 1
        end
    end
    return rnk_X
end

# Index set creator function for the dual variables λ that are associated with the

struct i_j_idx # correspond to (i,j) pair, where i,j ∈ I_N_⋆
    i::Int64 # corresponds to index i
    j::Int64 # corresponds to index j
end

# Custom structure for saving the worst-case function
struct worst_case_function
	x_array::OffsetMatrix{Float64, Matrix{Float64}}
	g_array::OffsetMatrix{Float64, Matrix{Float64}}
	f_array::OffsetVector{Float64, Vector{Float64}}
end

# We have dual variable λ={λ_ij}_{i,j} where i,j ∈ I_N_star
# The following function creates the maximal index set for λ

function index_set_constructor_for_dual_vars_full(I_N_star)

    # construct the index set for λ
    idx_set_λ = i_j_idx[]
    for i in I_N_star
        for j in I_N_star
            if i!=j
                push!(idx_set_λ, i_j_idx(i,j))
            end
        end
    end

    return idx_set_λ

end

# The following function will return the effective index set of a known λ i.e., those indices of  that are  λ  that are non-zero.

function effective_index_set_finder(λ ; ϵ_tol = 0.0005)

    # the variables λ are of the type DenseAxisArray whose index set can be accessed using _.axes and data via _.data syntax

    idx_set_λ_current = (λ.axes)[1]

    idx_set_λ_effective = i_j_idx[]

    # construct idx_set_λ_effective

    for i_j_λ in idx_set_λ_current
        if abs(λ[i_j_λ]) >= ϵ_tol # if λ[i,j] >= ϵ, where ϵ is our cut off for accepting nonzero
            push!(idx_set_λ_effective, i_j_λ)
        end
    end

    return idx_set_λ_effective

end

## The following function will construct the effective index set of λ based on the observation

function idx_set_nz_λ_constructor(N)

	idx_set_nz_λ = []

	for i in 0:N
		idx_set_nz_λ  = [idx_set_nz_λ; i_j_idx(-1, i)]
	end

  for i in 0:N-1
		idx_set_nz_λ = [idx_set_nz_λ;  i_j_idx(i, i+1)]
	end

	return idx_set_nz_λ

end



## Data generator function for NCG PEP relaxed line search version
# ================================================================

function data_generator_function(N, β, χ;
	ξ_input_data_generator = 0 # controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if ξ = 0 then we just use a bound of the form
	# ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
	)

	# define all the bold vectors
	# ---------------------------

	# define the 𝐠 vectors
	# 𝐠 = [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]

	dim_𝐠 = 2*N + 3

	𝐠 = OffsetArray(zeros(dim_𝐠, N+2), 1:dim_𝐠, -1:N)

	𝐠[:, -1] = zeros(dim_𝐠, 1)

	for i in 0:N
		𝐠[:, i] = e_i(dim_𝐠 , i+2)
	end

	# define the 𝐟 vectors
	# 𝐟 = [𝐟_⋆ 𝐟_0 𝐟_1 ... 𝐟_N]

	dim_𝐟 = N+1

	𝐟 = OffsetArray(zeros(dim_𝐟, N+2), 1:dim_𝐟, -1:N)

	𝐟[:, -1] = zeros(dim_𝐟, 1)

	for i in 0:N
		𝐟[:, i] = e_i(dim_𝐟, i+1)
	end

	# define the pseudogradient vectors 𝐩
	# 𝐩 = [𝐩_⋆ 𝐩_0 𝐩_1 𝐩_2 ... 𝐩_{N}] ∈ 𝐑^(N+2 × dim_𝐩)

	dim_𝐩 = dim_𝐠

	𝐩 = OffsetArray(Matrix{Any}(undef, dim_𝐩, N+2), 1:dim_𝐩, -1:N)

	𝐩[:, -1] = zeros(dim_𝐩, 1)

  if ξ_input_data_generator == 0 # ξ_input_data_generator == 0 => no restart
    	𝐩[:, 0] = e_i(dim_𝐩, 1)
	elseif ξ_input_data_generator == 1 # ξ_input_data_generator == 1 => restart
	  	𝐩[:, 0] = 𝐠[:, 0]
	else
		  @error "ξ_input_data_generator has to be equal to 0 or 1"
	  	return
	end

	if N == 2
		for i in 1:N-1
	   	𝐩[:, i] = 𝐠[:,i] + χ[0, i] .* 𝐩[:,0]
		end
	end

	if N>=3
		for i in 1:N-1
			if i == 1
				𝐩[:, i] = 𝐠[:,i] + χ[0, i] .* 𝐩[:,0]
			elseif i >= 2
				𝐩[:, i] = 𝐠[:,i] + ( sum( χ[j, i] .* 𝐠[:,j] for j in 1:i-1) ) + χ[0, i] .* 𝐩[:,0]
			else
				@error "we need N>=2"
				return
			end
		end
	end

	# 𝐱 = [𝐱_⋆ 𝐱_0 𝐱_1 𝐱_2 ... 𝐱_{N}]

	dim_𝐱 = dim_𝐠

	𝐱  = OffsetArray(Matrix{Any}(undef,dim_𝐱, N+2), 1:dim_𝐱, -1:N)

	𝐱[:,-1] = zeros(dim_𝐱, 1)

	for i in 0:N
		𝐱[:, i] = e_i(dim_𝐱, (N+2)+(i+1))
	end

	return 𝐱, 𝐠, 𝐟, 𝐩


end


# N = 2
# m_data_generator_function = Model()
# @variable(m_data_generator_function, χ[0:N-2, 1:N-1])
# @variable(m_data_generator_function, β[0:N-2])
# 𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function(N, β, χ)
# @show [𝐩[:,0] 𝐠[:,0:N] 𝐱[:,0:N]] == I



## Define the encoder matrices for NCG PEP relaxed line search
# ============================================================
A_mat(i, j, 𝐠, 𝐱) = ⊙(𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
B_mat(i, j, 𝐱) = ⊙(𝐱[:,i]-𝐱[:,j], 𝐱[:,i]-𝐱[:,j])
C_mat(i, j, 𝐠) = ⊙(𝐠[:,i]-𝐠[:,j], 𝐠[:,i]-𝐠[:,j])
C_tilde_mat(i, j, χ, 𝐩) = ⊙(𝐩[:,i]-𝐩[:,j], 𝐩[:,i]-𝐩[:,j])
D_mat(i, j, 𝐠) = ⊙(𝐠[:,i], 𝐠[:,j])
D_tilde_mat(i, j, χ, 𝐠, 𝐩) = ⊙(𝐠[:,i], 𝐩[:,j])
E_mat(i, j, 𝐠, 𝐱) = ⊙(𝐠[:,i] - 𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
D_bar_mat(i, j, 𝐠, 𝐩) = ⊙(𝐠[:,i] - 𝐩[:,j], 𝐠[:,i] - 𝐩[:,j])
a_vec(i, j, 𝐟) = 𝐟[:, j] - 𝐟[:, i]


## Feasible Solution Generator
# This generates a feasible solution to NCG-PEP by applying the NCG to a randomly generated quadratic function.

function feasible_sol_generator_NCG_PEP(N, μ, L, η)

  R = 1
	d = (2*N) + 3

	# generate the eigenvalues μ = λ[1] <= λ[2] <= ... <= λ[d]=L
	eigvals_f = zeros(d)
	F1 = μ + 1e-6
	F2 = L - 1e-6
	eig_vals_f = sort([μ; F1 .+ (F2-F1)*(1 .- rand(d-2)); L])

	# create the diagonal matrix that defines f(x) = (1/2)*x'*Q_f*x
	Q_f = diagm(eig_vals_f)

	# generate the starting point
	# We could use this as an alternative
	# x_0_tilde = zeros(d)
	# x_0_tilde[1] = 1/μ
	# x_0_tilde[d] = 1/L
	x_0_tilde = randn(d)
	x_0_tilde = x_0_tilde/norm(x_0_tilde,2)
	R_tilde_sqd = 0.5*x_0_tilde'*Q_f*x_0_tilde
	R_tilde = sqrt(R_tilde_sqd)
	x_0 = x_0_tilde*(R/R_tilde)
	f_0 = 0.5*x_0'*Q_f*x_0

	## generate the elements of α, x, g, β, p
	α = OffsetVector(zeros(N), 0:N-1)
	β = OffsetVector(zeros(N), 0:N-1)

	# Declare H
	H = zeros(d, (2*N)+3)

	# Declare Ft
	Ft = zeros(1, N+1)

	# array of all x's
	x_array = OffsetArray(zeros(d,N+1), 1:d, 0:N)
	x_array[:, 0] = x_0

	# array of all g's
	g_array = OffsetArray(zeros(d,N+1), 1:d, 0:N)
	g_array[:, 0] = Q_f*x_array[:, 0]

	# array of all f's
	f_array = OffsetVector(zeros(N+1), 0:N)
	f_array[0] = f_0

	# array of all pseudo-gradients p's
	p_array = OffsetArray(zeros(d,N+1), 1:d, 0:N)
	p_array[:, 0] = Q_f*x_array[:, 0]

	# Putting the entries of α, x_array, g_array, and f_array one by one

	for k in 0:N-1
		# generate α[k]
		α[k] = (g_array[:,k]'*p_array[:,k])/(p_array[:,k]'*Q_f*p_array[:,k])
		# α[k] = (x_array[:, k]'*Q_f^2*x_array[:, k])/(x_array[:, k]'*Q_f^3*x_array[:, k])
		# generate x[k+1]
		x_array[:, k+1] = x_array[:, k] - α[k]*p_array[:,k]
		# generate g[k+1]
		g_array[:, k+1] = Q_f*x_array[:,k+1]# g_array[:, k] - α[k]*Q_f*p_array[:,k]
		# generate β[k]
		β[k] = ((g_array[:,k+1]'*g_array[:,k+1])- η*(g_array[:,k+1]'*g_array[:,k]))/(g_array[:,k]'*g_array[:,k])
		# generate p[k+1]
		p_array[:,k+1] = g_array[:,k+1] + β[k]*p_array[:, k]
		# generate f[i+1]
		f_array[k+1] = 0.5*x_array[:, k+1]'*Q_f*x_array[:, k+1]
	end

	for k in 0:N-1
		if abs(p_array[:,k]'*g_array[:,k+1]) >= 10^-6
			@error "conjugacy condition is not satisfied"
			return
		end
	end

	# Filling the entries of H and Ft one by one now

	H[:, 1] = p_array[:,0]

	for i in 2:N+2
		H[:, i] = g_array[:, i-2]
	end

	for i in (N+2)+1:(2*N)+3
		H[:, i] = x_array[:, i-(N+3)]
	end

	for i in 1:N+1
		Ft[1, i] = f_array[i-1]
	end

	# Generate G

	G = H'*H

	# Generate L_cholesky

	L_cholesky =  compute_pivoted_cholesky_L_mat(G)

	if norm(G - L_cholesky*L_cholesky', Inf) > 1e-6
		@info "checking the norm bound for feasible {G, L_cholesky}"
		@warn "||G - L_cholesky*L_cholesky^T|| = $(norm(G - L_cholesky*L_cholesky', Inf))"
	end

	# time to generate χ and α_tilde
	χ = OffsetArray(zeros(N-1,N-1), 0:N-2, 1:N-1)

	for k in 1:N-1
		χ[k-1, k] = β[k-1]
	end

	for k in 1:N-1
		for j in 0:k-2
			χ[j,k] = χ[j,k-1]*β[k-1]
		end
	end

	α_tilde = OffsetArray(zeros(N,N), 1:N, 0:N-1)

	for i in 1:N
		α_tilde[i,i-1] = α[i-1]
	end

	for i in 1:N
		for j in 0:i-2
			α_tilde[i,j] = α[j] + sum(α[k]*χ[j,k] for k in j+1:i-1)
		end
	end

	# verify if x and g array match

	x_test_array = OffsetArray(zeros(d,N+1), 1:d, 0:N)

	x_test_array[:,0] = x_array[:,0]

	p_test_array = OffsetArray(zeros(d,N), 1:d, 0:N-1)

	p_test_array[:,0] = p_array[:,0]

	for i in 1:N
		x_test_array[:,i] = x_array[:,0] - sum(α_tilde[i,j]*g_array[:,j] for j in 0:i-1)
	end

	for i in 1:N-1
		p_test_array[:, i] = g_array[:,i] + sum(χ[j,i]*g_array[:,j] for j in 0:i-1)
	end

	if norm(p_array[:,0:N-1] - p_test_array[:,0:N-1]) > 1e-10 || norm(x_array - x_test_array) > 1e-10
		@error "somehting went wrong during the conversion process of the feasible solution generation"
		return
	else
		@info "norm(p_feas_array - p_test_feas_array)=$(norm(p_array[:,0:N-1] - p_test_array[:,0:N-1]))"
		@info "norm(x_feas_array - x_test_feas_array)=$(norm(x_array - x_test_array))"
	end

	# shorten β

	β = OffsetVector([β[i] for i in 0:N-2], 0:N-2)

	return G, Ft, L_cholesky, α,  χ, α_tilde, β

end


# ## Warm start
# N = 10
# η = 1
# μ = 0.5
# L = 1
# G_ws, Ft_ws, L_cholesky_ws, α_ws,  χ_ws, α_tilde_ws, β_ws = feasible_sol_generator_NCG_PEP(N, μ, L, η)


## Constructs bound on the decision variable

function bound_M_input_construtor(G, β, χ, N)
	bound_M_input_1 = maximum(G[i,i] for i in 1:size(G,1))
	# bound_M_input_2 = maximum(abs(β[i]) for i in 0:N-2)
	bound_M_input_3 = maximum(abs.(χ))
	bound_M = 2*maximum([bound_M_input_1; bound_M_input_3])
	return bound_M
end

function bound_M_χ_input_construtor(χ, N)
	bound_M_input_3 = maximum(abs.(χ))
	bound_M = 2*bound_M_input_3
	return bound_M
end

## For FR, generates the c_k sequence satisfying
# ||g_k||^2 <= ||p_k||^2 <= c_k*||g_k||^2

function c_k_generator_FR(N, c_0, μ)
	c_k_array = OffsetVector(zeros(N-1), 0:N-2)
	c_k_array[0] = c_0
	for k in 1:N-2
		c_k_array[k] = ((L - μ)^2 + 4*c_k_array[k-1]*L*μ + 4*(L - μ)*sqrt((-1 + c_k_array[k-1])*L*μ))/(4*L*μ)
	end
	return c_k_array
end

## Generates valid bounds on the entries of β for Fletcher-Reeves

function bound_number_β_FR_generator(N, c_0, μ)
	bound_number_β_FR_array = OffsetVector(zeros(N-1), 0:N-2)
	c_k_array = c_k_generator_FR(N, c_0, μ)
	for k in 0:N-2
		bound_number_β_FR_array[k] = (L^2 - 6*L*μ + 4*c_k_array[k]*L*μ + μ^2 +
		4*L*sqrt((-1 + c_k_array[k])*L*μ) - 4*μ*sqrt((-1 + c_k_array[k])*L*μ))/
		(4*c_k_array[k]*L*μ)
	end
	return bound_number_β_FR_array
end



## N-point NCG-PEP Solver with relaxed line search
# ================================================

function N_point_NCG_PEP_solver(
  ## paramters
	# ----------
	N, μ, L,
  G_ws, Ft_ws, L_cholesky_ws, χ_ws, β_ws;
	## options
	# --------
  η = 1, # controls PRP or FR, if η = 1 => PRP, and if η = 0 => FR
	ξ = 0, # # controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if not we just use a bound of the form ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
	c_0 = Inf, # decides ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
	solution_type = :find_locally_optimal, # options are :find_locally_optimal and :find_globally_optimal
	ρ_lower_bound = 0, # Lower bound on the contraction factor
	ρ_upper_bound = Inf, # Upper bound on the contraction factor
  impose_pattern = :on, # options are :on and :off, (i) if :on then we use the interpolation inequalities corresponding to the effective index set of λ, and (ii) if :off, we use all the interpolation inequalities
	show_output = :on, # options are :on and :off
	impose_bound_implied = :on, # options are :on and :off, if it is :on then we use the fact that f(x_N)-f(x_⋆) <= f(x_0) - f(x_⋆), which we know from our 1-point analysis
	impose_bound_heuristic = :on, # options are :on and :off, we use bound computation heuristic based on the locally optimal solution
	PSDness_modeling = :exact, # options are :exact and :through_ϵ and :lazy_constraint_callback
	ϵ_tol_feas = 1e-6, # feasiblity tolerance for minimum eigenvalue of G
	ϵ_tol_feas_Gurobi = 1e-6, # Feasibility of primal constraints, i.e., whether ⟨a;x⟩ ≤ b holds for the primal solution. More precisely, ⟨a;x⟩ ≤ b  will be considered to hold if ⟨a;x⟩ - b ≤  FeasibilityTol. We have a polishing mechanism to improve the tolerance, so the parameter can be set as low as 1e-3
	max_cut_count = 1e6, # number of lazy cuts when we use lazy constraints for modeling G = L_cholesky*L_cholesky
	scaling_factor = 1, # scaling factor for G and Ft,
	bound_M_sf_g_1 = 1 # bound on the entries of G and Ft when scaling_factor > 1
	)

	## Number of points etc
	# ---------------------
	I_N_star = -1:N
	dim_G = (2*N)+3
	dim_Ft = N+1
	⋆ = -1

	## Scale the warm-start points appropriately

	G_ws = G_ws/scaling_factor

	Ft_ws = Ft_ws/scaling_factor

	L_cholesky_ws = L_cholesky_ws/sqrt(scaling_factor)

	## Solution type

	if solution_type == :find_globally_optimal

		@info "[🐌 ] globally optimal solution finder activated, solution method: spatial branch and bound"

		NCG_PEP_model =  Model(Gurobi.Optimizer)
		# using direct_model results in smaller memory allocation
		# we could also use
		# Model(Gurobi.Optimizer)
		# but this requires more memory

		set_optimizer_attribute(NCG_PEP_model, "NonConvex", 2)
		# "NonConvex" => 2 tells Gurobi to use its nonconvex algorithm

		set_optimizer_attribute(NCG_PEP_model, "MIPFocus", 3)
		# If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
		# If you believe the solver is having no trouble finding the optimal solution, and wish to focus more
		# attention on proving optimality, select MIPFocus=2.
		# If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

		# 🐑: other Gurobi options one can play with
		# ------------------------------------------

		# turn off all the heuristics (good idea if the warm-starting point is near-optimal)
		# set_optimizer_attribute(NCG_PEP_model, "Heuristics", 0)
		# set_optimizer_attribute(NCG_PEP_model, "RINS", 0)

		# other termination epsilons for Gurobi
		# set_optimizer_attribute(NCG_PEP_model, "MIPGapAbs", 1e-4)

		set_optimizer_attribute(NCG_PEP_model, "MIPGap", 1e-2) # 99% optimal solution, because Gurobi will provide a result associated with a global lower bound within this tolerance, by polishing the result, we can find the exact optimal solution by solving a convex SDP

		# set_optimizer_attribute(NCG_PEP_model, "FuncPieceRatio", 0) # setting "FuncPieceRatio" to 0, will ensure that the piecewise linear approximation of the nonconvex constraints lies below the original function

		# set_optimizer_attribute(NCG_PEP_model, "Threads", 20) # how many threads to use at maximum
		#
		set_optimizer_attribute(NCG_PEP_model, "FeasibilityTol", ϵ_tol_feas_Gurobi)
		#
		# set_optimizer_attribute(NCG_PEP_model, "OptimalityTol", 1e-4)

	elseif solution_type == :find_locally_optimal

		@info "[🐰 ] locally optimal solution finder activated, solution method: interior point method"

		NCG_PEP_model = Model(
		optimizer_with_attributes(
		KNITRO.Optimizer,
		"convex" => 0,
		"strat_warm_start" => 1,
		# the last settings below are for larger N
		# you can comment them out if preferred but not recommended
		"honorbnds" => 1,
		# "bar_feasmodetol" => 1e-3,
		"feastol" => ϵ_tol_feas,
		"maxit" => 50000,
		# "ncvx_qcqp_init" => 4,
		# "infeastol" => 1e-12,
		"opttol" =>  ϵ_tol_feas)
		)


	end

	## Define the variables


	# add the variables
	# -----------------

	@info "[🎉 ] defining the variables"

	# construct G ⪰ 0
	@variable(NCG_PEP_model, G[1:dim_G, 1:dim_G], Symmetric)

	# define the cholesky matrix of Z: L_cholesky
	@variable(NCG_PEP_model, L_cholesky[1:dim_G, 1:dim_G])

	# construct Ft (this is just transpose of F)
	@variable(NCG_PEP_model, Ft[1:dim_Ft] >= 0) # Because without loss of generality f_⋆ = 0, all the entries of Ft will be non-negative

	# Construct χ
	@variable(NCG_PEP_model, χ[0:N-2, 1:N-1])

	# Construct β
	@variable(NCG_PEP_model, β[0:N-2])

	# construct the hypograph variables
	@variable(NCG_PEP_model,  t >= ρ_lower_bound/scaling_factor)

	if ρ_upper_bound  <= 2*ρ_lower_bound
		@info "[🃏 ] Adding upper bound on objective to be maximized"
		set_upper_bound(t, ρ_upper_bound/scaling_factor)
	end

	# Number of interpolation constraints

	if impose_pattern == :off
		# define λ over the full index set
		idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)
	elseif impose_pattern == :on
		@info "[🎣 ] imposing pattern on effective index set of λ"
	    idx_set_λ = idx_set_nz_λ_constructor(N)
	else
		@error "impose_pattern should be either :on or :off"
	end

	## Create the data generator

	@info "[🎍 ] adding the data generator function to create 𝐱, 𝐠, 𝐟"

	𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function(N, β, χ; ξ_input_data_generator = ξ)

	## Add the constraints
	# --------------------

	@info "[🎢 ] adding the constraints"

	## Constraint
	#  ||g_0||^2 <= ||p_0||^2
	# ⇔ tr G C_{0,⋆} <= tr G C_tilde_{0,⋆}

	@constraint(NCG_PEP_model, con_p0_sqd_lb, tr(G*C_mat(0, ⋆ , 𝐠)) <= tr(G*C_tilde_mat(0, ⋆, χ, 𝐩)))

	## Constraint
	# ||p_0||^2 <= c_0 ||g_0||^2
	# ⇔ tr G C_tilde_{0,⋆} <= c_0*tr G C_{0,⋆}

	@constraint(NCG_PEP_model, con_p0_sqd_ub, tr(G*C_tilde_mat(0, ⋆, χ, 𝐩)) <= c_0*tr(G*C_mat(0, ⋆, 𝐠)))

	## Constraint
	# ⟨ g_1 ; p_0⟩ = 0
	# ⇔ tr G D_tilde_{1,0} = 0

	@constraint(NCG_PEP_model, con_line_search_init_1, tr(G*D_tilde_mat(1, 0, χ, 𝐠, 𝐩)) == 0)

	## Constraint
	# ⟨ g_1 ; x_0 - x_1⟩ = 0
	# ⇔ tr G A_{0,1} == 0

	@constraint(NCG_PEP_model, con_line_search_init_2, tr(G*A_mat(0, 1, 𝐠, 𝐱)) == 0)

	## Constraint
	# ⟨ g_0 ; p_0⟩ = ||g_0||^2
	# ⇔ tr G D_tilde_{0,0} = tr G C_{0,⋆}

	@constraint(NCG_PEP_model, con_g0_p0_relation, tr(G*D_tilde_mat(0, 0, χ, 𝐠, 𝐩)) == tr(G*C_mat(0, ⋆, 𝐠)))

	## Constraint
	# ξ*||g_0 - p_0||^2
	# ⇔ tr G * (ξ * D_bar_{0,0}) == 0

	# @constraint(NCG_PEP_model, con_restart, ξ*tr(G*D_bar_mat(0,0, 𝐠, 𝐩)) == 0)

	## Constraint
	# ∀ i ∈ [1:N-1] β_{i-1}*||g_{i-1}||^2 = ||g_i||^2 - η* ⟨ g_i ∣ g_{i-1} ⟩
	# ⇔ i ∈ [1:N-1] β_{i-1}* tr G C_{i-1, ⋆} = tr G (C_{i,⋆} - η*D_{i,i-1})

	@constraint(NCG_PEP_model, con_β_formula[i in 1:N-1], β[i-1] * tr(G*C_mat(i-1, ⋆, 𝐠)) == tr(G * (C_mat(i, ⋆, 𝐠) - η*D_mat(i, i-1, 𝐠))) )

	## Constraint ∀i ∈ [1:N-1], j ∈ [0:i-2] χ[j,i] = χ[j,i-1]*β[i-1]

	@constraint(NCG_PEP_model, con_χ_formula_1[i in 1:N-1, j in 0:i-2], χ[j,i] == χ[j,i-1]*β[i-1])

	## Constraint ∀i ∈ [1:N-1] χ[i-1,i] = β[i-1]

	@constraint(NCG_PEP_model, con_χ_formula_2[i in 1:N-1], χ[i-1, i] == β[i-1])

	## Constraint
	# ∀ i ∈ [1:N-1] ⟨ g_{i+1} ; p_{i}⟩ = 0
	# ⇔ tr G D_tilde_{i+1, i} = 0

	@constraint(NCG_PEP_model, con_line_search_i_1[i in 1:N-1], tr(G*D_tilde_mat(i+1, i, χ, 𝐠, 𝐩)) == 0)

	## Constraint
	# ∀ i ∈ [1:N-1] ⟨ g_{i+1} ; x_i - x_{i+1}⟩ = 0
	# ⇔ ∀ i ∈ [1:N-1] tr G A_{i, i+1} = 0

	@constraint(NCG_PEP_model, con_line_search_i_2[i in 1:N-1], tr(G*A_mat(i, i+1, 𝐠, 𝐱)) == 0)

	## Constraint
	# ∀ i ∈ [1:N-1]  ⟨ g_{i} ; p_{i}⟩ = ||g_{i}||^2
	# ⇔ ∀ i ∈ [1:N-1]  tr G D_tilde_{i, i} = tr G C_{i, ⋆}

	@constraint(NCG_PEP_model, con_gi_pi_relation[i in 1:N-1], tr(G*D_tilde_mat(i, i, χ, 𝐠, 𝐩)) == tr(G*C_mat(i, ⋆, 𝐠)))

	## Constraint
	# f_0 - f_⋆ = 1
	# ⇔ F a_{⋆, 0} = 1

	@constraint(NCG_PEP_model, con_init, Ft'*a_vec(⋆,0, 𝐟) == 1/scaling_factor)

	## Interpolation constraint
	## f_i >= f_j + ⟨ g_j ; x_i - x_j ⟩ + 1/(2*(1-q)) ( 1/L ||g_i - g_j||^2 + μ ||x_i - x_j||^2 - 2 q ⟨ g_i - g_j ; x_i - x_j⟩), where q = μ/L

	q = μ/L

	con_interpol = map(1:length(idx_set_λ)) do ℓ
		i_j_λ = idx_set_λ[ℓ]
		@info "coninterpol ℓ = $(i_j_λ)"
		@constraint(NCG_PEP_model,
		Ft'*a_vec(i_j_λ.i,i_j_λ.j,𝐟)
		+ tr(G * (
		A_mat(i_j_λ.i, i_j_λ.j, 𝐠, 𝐱)
		+ ( 1/(2*(1-q)) )*(
		(1/(L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠)
		+	(μ*B_mat(i_j_λ.i, i_j_λ.j, 𝐱))
		- (2*q*E_mat(i_j_λ.i, i_j_λ.j, 𝐠, 𝐱) ) ) ) ) <= 0)
	end # end the do map

	## Hypograph constraint

	@constraint(NCG_PEP_model, hypographCon, t <= Ft'*a_vec(⋆,N, 𝐟))

	## Constraints to model G = L_cholesky*L_cholesky'

	if PSDness_modeling == :exact

		# direct modeling through definition and vectorization
		# ---------------------------------------------------
		@constraint(NCG_PEP_model, vectorize(G - (L_cholesky * L_cholesky'), SymmetricMatrixShape(dim_G)) .== 0)

	elseif PSDness_modeling == :through_ϵ

		# definition modeling through vectorization and ϵ_tol_feas

		# part 1: models Z-L_cholesky*L_cholesky <= ϵ_tol_feas*ones(dim_G,dim_G)
		@constraint(NCG_PEP_model, vectorize(G - (L_cholesky * L_cholesky') - ϵ_tol_feas*ones(dim_G,dim_G), SymmetricMatrixShape(dim_G)) .<= 0)

		# part 2: models Z-L_cholesky*L_cholesky >= -ϵ_tol_feas*ones(dim_G,dim_G)

		@constraint(NCG_PEP_model, vectorize(G - (L_cholesky * L_cholesky') + ϵ_tol_feas*ones(dim_G,dim_G), SymmetricMatrixShape(dim_G)) .>= 0)

	elseif PSDness_modeling == :lazy_constraint_callback

		# set_optimizer_attribute(NCG_PEP_model, "FuncPieces", -2) # FuncPieces = -2: Bounds the relative error of the approximation; the error bound is provided in the FuncPieceError attribute. See https://www.gurobi.com/documentation/9.1/refman/funcpieces.html#attr:FuncPieces

		# set_optimizer_attribute(NCG_PEP_model, "FuncPieceError", 0.1) # relative error

		set_optimizer_attribute(NCG_PEP_model, "MIPFocus", 1) # focus on finding good quality feasible solution

		# add initial cuts
		num_cutting_planes_init = (5*dim_G^2)
		cutting_plane_array = [eigvecs(G_ws) rand(Uniform(-1,1), dim_G, num_cutting_planes_init)] # randn(dim_G,num_cutting_planes_init)
		num_cuts_array_rows, num_cuts = size(cutting_plane_array)
		for i in 1:num_cuts
			d_cut = cutting_plane_array[:,i]
			d_cut = d_cut/norm(d_cut,2) # normalize the cutting plane vector
			@constraint(NCG_PEP_model, tr(G*(d_cut*d_cut')) >= 0)
		end

		cut_count = 0

		# add the lazy callback function
		# ------------------------------
		function add_lazy_callback(cb_data)
			if cut_count <= max_cut_count
				G0 = zeros(dim_G,dim_G)
				for i=1:dim_G
					for j=1:dim_G
						G0[i,j]=callback_value(cb_data, G[i,j])
					end
				end

				eig_abs_max_G0 = max(abs(eigmax(G0)), abs(eigmin(G0)))

				if (eigvals(G0)[1]) <= -ϵ_tol_feas # based on the normalized smallest eigenvalue that matters
					u_t = eigvecs(G0)[:,1]
					u_t = u_t/norm(u_t,2)
					con3 = @build_constraint(tr(G*u_t*u_t') >=0.0)
					MOI.submit(NCG_PEP_model, MOI.LazyConstraint(cb_data), con3)
				end
				cut_count+=1
			end
		end

		# submit the lazy constraint
		# --------------------------
		MOI.set(NCG_PEP_model, MOI.LazyConstraintCallback(), add_lazy_callback)

	else

		@error "something is not right in PSDness_modeling"

		return

	end

	## List of valid constraints for G

	# diagonal components of G are non-negative

	@constraint(NCG_PEP_model, conNonNegDiagG[i=1:dim_G], G[i,i] >= 0)


	# the off-diagonal components satisfy:
	# (∀i,j ∈ dim_G: i != j) -(0.5*(G[i,i] + G[j,j])) <= G[i,j] <=  (0.5*(G[i,i] + G[j,j]))

	for i in 1:dim_G
		for j in 1:dim_G
			if i != j
				@constraint(NCG_PEP_model, G[i,j] <= (0.5*(G[i,i] + G[j,j])) )
				@constraint(NCG_PEP_model, -(0.5*(G[i,i] + G[j,j])) <= G[i,j] )
			end
		end
	end

	# Two constraints to define the matrix L_cholesky to be a lower triangular matrix
	# -------------------------------------------------

	# upper off-diagonal terms of L_cholesky are zero

	for i in 1:dim_G
	 for j in 1:dim_G
		 if i < j
			 fix(L_cholesky[i,j], 0; force = true)
		 end
	 end
	end

	# diagonal components of L_cholesky are non-negative

	for i in 1:dim_G
		@constraint(NCG_PEP_model, L_cholesky[i,i] >= 0)
	end

	## Add the implied bounds on the entries of G

	# Constraint
	# ∀ i ∈ [0:N] 2μ F[i+1] <= G[i+2, i+2] <= 2L F[i+1]

	@constraint(NCG_PEP_model, con_gi_sqd_lb_i[i in 0:N], 2*μ*Ft[i+1] <= G[i+2, i+2])

	@constraint(NCG_PEP_model, con_gi_sqd_ub_i[i in 0:N], G[i+2, i+2] <= 2*L*Ft[i+1])

	# Constraint
	# ∀ i ∈ [0:N] (2/L) F[i+1] <= G[(N+2)+(i+1), (N+2)+(i+1)] <= (2/μ) F[i+1]

	@constraint(NCG_PEP_model, con_xi_sqd_lb_i[i in 0:N], (2/L)*Ft[i+1] <= G[N+2+i+1, N+2+i+1])

	@constraint(NCG_PEP_model, con_xi_sqd_ub_i[i in 0:N], G[N+2+i+1, N+2+i+1] <= (2/μ)*Ft[i+1])

	## Add the implied bounds on the entries of β

	if η == 1 #  if it is PRP
	   # ub_β = (((L - μ)*(L - μ + 2*Sqrt((-1 + c)*L*μ)))/(4*c*L*μ))
	   # lb_β = -(((L - μ)*(L - μ + 2*Sqrt((-1 + c)*L*μ)))/(4*c*L*μ))
	   bound_number_β_PRP = (((L - μ)*(L - μ + 2*sqrt((-1 + c_0)*L*μ)))/(4*c_0*L*μ))
	   set_lower_bound.(β, -bound_number_β_PRP)
	   set_upper_bound.(β, bound_number_β_PRP)
	elseif η == 0 # if it is FR
	   bound_number_β_FR_array = bound_number_β_FR_generator(N, c_0, μ)
	   for k in 0:N-2
		   set_lower_bound.(β[k], -bound_number_β_FR_array[k])
		   set_upper_bound.(β[k], bound_number_β_FR_array[k])
	   end
	end

	## Add the objective

	@objective(NCG_PEP_model, Max, t*scaling_factor)

	## Time to warm-start all the variables

	@info "[👲 ] warm-start values for all the variables"

	# warm-start G

	for i in 1:dim_G
		for j in 1:dim_G
			set_start_value(G[i,j], G_ws[i,j])
		end
	end

	# warm-start Ft

	for i in 1:dim_Ft
		set_start_value(Ft[i], Ft_ws[i])
	end

	# warm-start χ

	for k in 1:N-1
		for j in 0:N-2
			set_start_value(χ[j,k], χ_ws[j,k])
		end
	end

	# warm-start β

	for i in 0:N-2
		set_start_value(β[i], β_ws[i])
	end

	# warm-start L_cholesky

	for i in 1:dim_G
		for j in 1:dim_G
			set_start_value(L_cholesky[i,j], L_cholesky_ws[i,j])
		end
	end

	# warm start t

	t_ws =  Ft_ws[end] #(c_f*(Ft_ws*a_vec(-1,N, 𝐟)))[1]

	set_start_value(t, t_ws)

	if any(isnothing, start_value.(all_variables(NCG_PEP_model))) == true
		@error "all the variables are not warm-started"
	else
		@info "😃 all the variables are warm-started"
	end

	## Impose bounds on the decision variables

	if impose_bound_implied == :on
		for i in 1:dim_Ft
			set_upper_bound(Ft[i], 1.01)
		end
	end

	if impose_bound_heuristic == :on && scaling_factor == 1

		bound_M = bound_M_input_construtor(G_ws, β_ws, χ_ws, N)

		bound_M_cholesky = sqrt(bound_M)

		# set bound for χ
		# ---------------
		set_lower_bound.(χ, -bound_M)
		set_upper_bound.(χ, bound_M)

		# set bound for β ( we have better bound now)
		# ---------------
		# set_lower_bound.(β, -bound_M)
		# set_upper_bound.(β, bound_M)

		# set bound for G
		# ---------------
		for i in 1:dim_G
				for j in 1:dim_G
					if i != j
						set_lower_bound(G[i,j], -bound_M)
						set_upper_bound(G[i,j], bound_M)
					elseif i == j
						set_lower_bound(G[i,j], 0)
						set_upper_bound(G[i,j], bound_M)
					end
				end
		end


		# set bound for L_cholesky
		# ------------------------

		# need only upper bound for the diagonal compoments, as the lower bound is zero from the model
		for i in 1:dim_G
			    set_lower_bound(L_cholesky[i,i], 0)
				set_upper_bound(L_cholesky[i,i], bound_M_cholesky)
		end
		# need to bound only components, L_cholesky[i,j] with i > j, as for i < j, we have zero, due to the lower triangular structure
		for i in 1:dim_G
				for j in 1:dim_G
						if i > j
								set_lower_bound(L_cholesky[i,j], -bound_M_cholesky)
								set_upper_bound(L_cholesky[i,j], bound_M_cholesky)
						end
				end
		end

	elseif impose_bound_heuristic == :on && scaling_factor > 1

		bound_M = bound_M_χ_input_construtor(χ_ws, N)

		# set bound for χ
		# ---------------
		set_lower_bound.(χ, -bound_M)
		set_upper_bound.(χ, bound_M)

		# set bound for β ( we have better bound now)
		# ---------------
		# set_lower_bound.(β, -bound_M)
		# set_upper_bound.(β, bound_M)

		# set bound for G
		# ---------------
		for i in 1:dim_G
				for j in 1:dim_G
					if i != j
						set_lower_bound(G[i,j], -bound_M_sf_g_1)
						set_upper_bound(G[i,j], bound_M_sf_g_1)
					elseif i == j
						set_lower_bound(G[i,j], 0)
						set_upper_bound(G[i,j], bound_M_sf_g_1)
					end
				end
		end

		for i in 1:dim_Ft
			set_upper_bound(Ft[i], bound_M_sf_g_1)
		end


		# set bound for L_cholesky
		# ------------------------

		# need only upper bound for the diagonal compoments, as the lower bound is zero from the model
		for i in 1:dim_G
					set_lower_bound(L_cholesky[i,i], 0)
				set_upper_bound(L_cholesky[i,i], sqrt(bound_M_sf_g_1))
		end
		# need to bound only components, L_cholesky[i,j] with i > j, as for i < j, we have zero, due to the lower triangular structure
		for i in 1:dim_G
				for j in 1:dim_G
						if i > j
								set_lower_bound(L_cholesky[i,j], -sqrt(bound_M_sf_g_1))
								set_upper_bound(L_cholesky[i,j], sqrt(bound_M_sf_g_1))
						end
				end
		end

	end # end impose_bound_heuristic

	## time to optimize
	# ----------------

	@info "[🙌 	🙏 ] model building done, starting the optimization process"

	if show_output == :off
		set_silent(NCG_PEP_model)
	end

	optimize!(NCG_PEP_model)

	@info "NCG_PEP_model has termination status = " termination_status(NCG_PEP_model)

	solve_time_NCG_PEP = solve_time(NCG_PEP_model)

	## Bound violation checker

	if impose_bound_heuristic == :on && scaling_factor > 1

			bound_violation_checker_NCG_PEP(
			    # input point
			    # -----------
			    value.(G), value.(Ft), value.(L_cholesky),
			    # input bounds
			    # ------------
			    -bound_M_sf_g_1, bound_M_sf_g_1, 0, bound_M_sf_g_1, -sqrt(bound_M_sf_g_1), sqrt(bound_M_sf_g_1);
			    # options
			    # -------
			    show_output = :on
			    )

	end

	## Time to store the solution

	if (solution_type == :find_locally_optimal && (termination_status(NCG_PEP_model) == LOCALLY_SOLVED || termination_status(NCG_PEP_model) == SLOW_PROGRESS || termination_status(NCG_PEP_model) == ITERATION_LIMIT))  || (solution_type ==:find_globally_optimal && termination_status(NCG_PEP_model) == OPTIMAL )

		if termination_status(NCG_PEP_model) == SLOW_PROGRESS
			@warn "[💀 ] termination status of NCG_PEP_model is SLOW_PROGRESS"
		end

		# store the solutions and return
		# ------------------------------

		@info "[😻 ] optimal solution found done, store the solution"

		# store G_opt

		G_opt = scaling_factor*value.(G)

		# store L_cholesky_opt

		L_cholesky_opt = compute_pivoted_cholesky_L_mat(G_opt) # 💀 Do not need to scale this one, as we have already rescaled G_opt

		# store F_opt

		Ft_opt = scaling_factor*value.(Ft)

		# L_cholesky_opt = value.(L_cholesky)

		if norm(G_opt - L_cholesky_opt*L_cholesky_opt', Inf) > 10^-4
			@warn "||G - L_cholesky*L_cholesky^T|| = $(norm(G_opt -  L_cholesky_opt*L_cholesky_opt', Inf))"
		end

		# store χ_k_opt

		χ_opt = value.(χ)

		# store β_kM1_opt

	  β_opt = value.(β)

		contraction_factor_opt = scaling_factor*value.(t)

		@info "[💹 ] warm-start objective value = $(Ft_ws[end]*scaling_factor), and objective value of found solution = $(contraction_factor_opt)"


	else

		@error "[🙀 ] could not find an optimal solution"

	end

	## Time to return the solution

	return G_opt, Ft_opt, L_cholesky_opt, χ_opt, β_opt, contraction_factor_opt, solve_time_NCG_PEP


end



# Finally have a function to check if any of the bounds are violated for sanity check
function bound_violation_checker_NCG_PEP(
    # input point
    # -----------
    G_sol, Ft_sol, L_cholesky_sol,
    # input bounds
    # ------------
    G_lb, G_ub, Ft_lb, Ft_ub, L_cholesky_lb, L_cholesky_ub;
    # options
    # -------
    show_output = :on
    )

    if show_output == :on
			  @info "[🐅] Checking bounds for G_scaled, L_cholesky_scaled"
				@info "================================================="
        @show [G_lb minimum(G_sol)  maximum(G_sol) G_ub]
        @show [L_cholesky_lb  minimum(L_cholesky_sol)  maximum(L_cholesky_sol) L_cholesky_ub]
				# @show [χ_lb  minimum(χ)  maximum(χ) χ_ub]
				# @show [β_lb  minimum(β)  maximum(β) β_ub]
    end

    # bound satisfaction flag

    bound_satisfaction_flag = 1

    # verify bound for χ
		# if !(χ_lb -  1e-8 < minimum(χ_sol) && maximum(χ_sol) < χ_ub + 1e-8)
		# 		@error "found χ is violating the input bound"
		# 		bound_satisfaction_flag = 0
		# end

		# verify bound for β
		# if !(β_lb -  1e-8 < minimum(β_sol) && maximum(β_sol) < β_ub + 1e-8)
		# 		@error "found β is violating the input bound"
		# 		bound_satisfaction_flag = 0
		# end

    # verify bound for G
    if !(G_lb -  1e-8 < minimum(G_sol) && maximum(G_sol) < G_ub + 1e-8)
        @error "found G is violating the input bound"
        bound_satisfaction_flag = 0
    end

    # verify bound for L_cholesky
    if !(L_cholesky_lb -  1e-8 < minimum(L_cholesky_sol) && maximum(L_cholesky_sol) < L_cholesky_ub +  1e-8)
        @error "found L_cholesky is violating the input bound"
        bound_satisfaction_flag = 0
    end

    if bound_satisfaction_flag == 0
        @error "[💀 ] some bound is violated, increase the bound intervals"
				return
    elseif bound_satisfaction_flag == 1
        @info "[😅 ] all bounds are satisfied by the input point, rejoice"
    end

    return bound_satisfaction_flag

end


## Helper functions for exact line search

# another important function to find proper index of Θ given (i,j) pair
index_finder_Θ(i,j,idx_set_λ) = findfirst(isequal(i_j_idx(i,j)), idx_set_λ)



## Create the data generator for exact line search
# ================================================

function data_generator_function_NCG_PEP_exact_line_search(N, γ, α, β, χ; 	ξ_input_data_generator = 0 # controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if ξ = 0 then we just use a bound of the form
	# ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
)

	# define all the bold vectors
	# --------------------------

	# define 𝐱_0 and 𝐱_star

	dim_𝐠 = N+3

	dim_𝐱 = N+3

	dim_𝐩 = N+3

	dim_𝐟 = N+1

	𝐱_0 = e_i(dim_𝐱, 1)

	𝐱_star = zeros(dim_𝐱, 1)

	# define 𝐠_0, 𝐠_1, …, 𝐠_N

	# first we define the 𝐠 vectors,
	# index -1 corresponds to ⋆, i.e.,  𝐟[:,-1] =  𝐟_⋆ = 0

	# 𝐠= [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]

	𝐠 = OffsetArray(zeros(dim_𝐠, N+2), 1:dim_𝐠, -1:N)

	for i in 0:N
		𝐠[:,i] = e_i(dim_𝐠, i+2)
	end

	# time to define the 𝐟 vectors

	# 𝐟 = [𝐟_⋆ 𝐟_0, 𝐟_1, …, 𝐟_N]

	𝐟 = OffsetArray(zeros(dim_𝐟, N+2), 1:dim_𝐟, -1:N)

	for i in 0:N
		𝐟[:,i] = e_i(dim_𝐟, i+1)
	end

	# 𝐩 corresponds to the pseudo-gradient g̃
	# 𝐩 = [𝐩_⋆ 𝐩_0 𝐩_1 𝐩_2 ... 𝐩_{N-1}] ∈ 𝐑^(N+2 × N+1)

	𝐩 = OffsetArray(Matrix{Any}(undef, dim_𝐩, N+1), 1:dim_𝐩, -1:N-1)

	# assign values next using our formula for 𝐩_k
	𝐩[:,-1] = zeros(dim_𝐩, 1)

	if ξ_input_data_generator == 0 # ξ_input_data_generator == 0 => no restart
			𝐩[:, 0] = e_i(dim_𝐩, N+3)
	elseif ξ_input_data_generator == 1 # ξ_input_data_generator == 1 => restart
			𝐩[:, 0] = 𝐠[:, 0]
	else
			@error "ξ_input_data_generator has to be equal to 0 or 1"
			return
	end
	# 𝐩[:,0] = 𝐠[:,0]

	if N == 2
		for i in 1:N-1
			𝐩[:, i] = 𝐠[:,i] + χ[0, i] .* 𝐩[:,0]
		end
	end

	if N>=3
		for i in 1:N-1
			if i == 1
				𝐩[:, i] = 𝐠[:,i] + χ[0, i] .* 𝐩[:,0]
			elseif i >= 2
				𝐩[:, i] = 𝐠[:,i] + ( sum( χ[j, i] .* 𝐠[:,j] for j in 1:i-1) ) + χ[0, i] .* 𝐩[:,0]
			else
				@error "we need N>=2"
				return
			end
		end
	end

	# 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}] ∈ 𝐑^(N+2 × N+2)
	𝐱  = OffsetArray(Matrix{Any}(undef, dim_𝐱, N+2), 1:dim_𝐱, -1:N)

	# assign values next using our formula for 𝐱_k
	𝐱[:,-1] = 𝐱_star

	𝐱[:,0] = 𝐱_0

	for i in 1:N
		if i == 1
		   𝐱[:, i] = 𝐱[:,0] - α[i,0] .* 𝐩[:,0]
		 elseif i >= 2
			 	𝐱[:, i] = 𝐱[:,0] - ( sum( α[i,j] .* 𝐠[:,j] for j in 1:i-1) )- α[i,0] .* 𝐩[:,0]
		 else
			 @error "we need N >= 2"
		 end
	end

	return 𝐱, 𝐠, 𝐟, 𝐩

end


## Define the encoder matrices for exact line search
# ==================================================

A_mat_exact_line_search(i,j,α,𝐠,𝐱) = ⊙(𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
B_mat_exact_line_search(i,j,α,𝐱) = ⊙(𝐱[:,i]-𝐱[:,j], 𝐱[:,i]-𝐱[:,j])
C_mat_exact_line_search(i,j,𝐠) = ⊙(𝐠[:,i]-𝐠[:,j], 𝐠[:,i]-𝐠[:,j])
# D_mat_exact_line_search(i, 𝐠) = ⊙(𝐠[:,i], 𝐠[:,i-1])
D_mat_exact_line_search(i, j, 𝐠) = ⊙(𝐠[:,i], 𝐠[:,j])
# D_tilde_mat_exact_line_search(i,𝐠,𝐩,χ) = ⊙(𝐠[:,i], 𝐩[:,i-1])
D_tilde_mat_exact_line_search(i, j, χ, 𝐠, 𝐩) = ⊙(𝐠[:,i], 𝐩[:,j])
E_mat_exact_line_search(i,j,α,𝐠,𝐱) = ⊙(𝐠[:,i] - 𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
a_vec_exact_line_search(i,j,𝐟) = 𝐟[:, j] - 𝐟[:, i]
C_tilde_mat_exact_line_search(i, j, χ, 𝐩) = ⊙(𝐩[:,i]-𝐩[:,j], 𝐩[:,i]-𝐩[:,j])


## Find a feasible solution to compute lower bound for NCG PEP exact line search

function feasible_sol_generator_NCG_PEP_exact_line_search(N, μ, L, η_2)

	d = N+3

	# generate the eigenvalues μ = λ[1] <= λ[2] <= ... <= λ[d]=L
	eigvals_f = zeros(d)
	F1 = μ + 1e-6
	F2 = L - 1e-6
	eig_vals_f = sort([μ; F1 .+ (F2-F1)*(1 .- rand(d-2)); L])

	# create the diagonal matrix that defines f(x) = (1/2)*x'*Q_f*x
	Q_f = diagm(eig_vals_f)

	# generate the starting point
	# x_0_tilde = zeros(d)
	# x_0_tilde[1] = 1/μ
	# x_0_tilde[d] = 1/L
	x_0_tilde = randn(d)
	x_0_tilde = x_0_tilde/norm(x_0_tilde,2)
	R_tilde_sqd = 0.5*x_0_tilde'*Q_f*x_0_tilde
	R_tilde = sqrt(R_tilde_sqd)
	x_0 = x_0_tilde*(R/R_tilde)
	f_0 = 0.5*x_0'*Q_f*x_0

	## generate the elements of γ, x, g, β, p
	γ = OffsetVector(zeros(N), 0:N-1)
	β = OffsetVector(zeros(N), 0:N-1)

	# Declare H
	H = zeros(d, N+3)

	# Declare Ft
	Ft = zeros(1, N+1)

	# array of all x's
	x_array = OffsetArray(zeros(d,N+1), 1:d, 0:N)
	x_array[:, 0] = x_0

	# array of all g's
	g_array = OffsetArray(zeros(d,N+1), 1:d, 0:N)
	g_array[:, 0] = Q_f*x_array[:, 0]

	# array of all f's
	f_array = OffsetVector(zeros(N+1), 0:N)
	f_array[0] = f_0

	# array of all pseudo-gradients p's
	p_array = OffsetArray(zeros(d,N+1), 1:d, 0:N)
	p_array[:, 0] = Q_f*x_array[:, 0]

	# Putting the entries of γ, x_array, g_array, and f_array one by one

	for k in 0:N-1
		# generate γ[k]
		γ[k] = (g_array[:,k]'*p_array[:,k])/(p_array[:,k]'*Q_f*p_array[:,k])
		# γ[k] = (x_array[:, k]'*Q_f^2*x_array[:, k])/(x_array[:, k]'*Q_f^3*x_array[:, k])
		# generate x[k+1]
		x_array[:, k+1] = x_array[:, k] - γ[k]*p_array[:,k]
		# generate g[k+1]
		g_array[:, k+1] = Q_f*x_array[:,k+1]# g_array[:, k] - γ[k]*Q_f*p_array[:,k]
		# generate β[k]
		β[k] = ((g_array[:,k+1]'*g_array[:,k+1])- η_2*(g_array[:,k+1]'*g_array[:,k]))/(g_array[:,k]'*g_array[:,k])
		# generate p[k+1]
		p_array[:,k+1] = g_array[:,k+1] + β[k]*p_array[:, k]
		# generate f[i+1]
		f_array[k+1] = 0.5*x_array[:, k+1]'*Q_f*x_array[:, k+1]
	end

	for k in 0:N-1
		if abs(p_array[:,k]'*g_array[:,k+1]) >= 10^-6
			@error "conjugacy condition is not satisfied"
			return
		end
	end

	# Filling the entries of H and Ft one by one now

	H[:, 1] = x_array[:,0]

	for i in 2:N+2
		H[:, i] = g_array[:, i-2]
	end

	H[:, N+3] = p_array[:, 0]

	for i in 1:N+1
		Ft[1, i] = f_array[i-1]
	end

	# Generate G

	G = H'*H

	# Generate L_cholesky

	L_cholesky =  compute_pivoted_cholesky_L_mat(G)

	if norm(G - L_cholesky*L_cholesky', Inf) > 1e-6
		@info "checking the norm bound for feasible {G, L_cholesky}"
		@warn "||G - L_cholesky*L_cholesky^T|| = $(norm(G - L_cholesky*L_cholesky', Inf))"
	end

	# time to generate χ and α
	χ = OffsetArray(zeros(N-1,N-1), 0:N-2, 1:N-1)

	for k in 1:N-1
		χ[k-1, k] = β[k-1]
	end

	for k in 1:N-1
		for j in 0:k-2
			χ[j,k] = χ[j,k-1]*β[k-1]
		end
	end

	α = OffsetArray(zeros(N,N), 1:N, 0:N-1)

	for i in 1:N
		α[i,i-1] = γ[i-1]
	end

	for i in 1:N
		for j in 0:i-2
			α[i,j] = γ[j] + sum(γ[k]*χ[j,k] for k in j+1:i-1)
		end
	end

	# verify if x and g array match

	x_test_array = OffsetArray(zeros(d,N+1), 1:d, 0:N)

	x_test_array[:,0] = x_array[:,0]

	p_test_array = OffsetArray(zeros(d,N), 1:d, 0:N-1)

	p_test_array[:,0] = p_array[:,0]

	for i in 1:N
		x_test_array[:,i] = x_array[:,0] - sum(α[i,j]*g_array[:,j] for j in 0:i-1)
	end

	for i in 1:N-1
		p_test_array[:, i] = g_array[:,i] + sum(χ[j,i]*g_array[:,j] for j in 0:i-1)
	end

	if norm(p_array[:,0:N-1] - p_test_array[:,0:N-1]) > 1e-10 || norm(x_array - x_test_array) > 1e-10
		@error "somehting went wrong during the conversion process of the feasible solution generation"
		return
	else
		@info "norm(p_feas_array - p_test_feas_array)=$(norm(p_array[:,0:N-1] - p_test_array[:,0:N-1]))"
		@info "norm(x_feas_array - x_test_feas_array)=$(norm(x_array - x_test_array))"
	end

	# shorten β

	β = OffsetVector([β[i] for i in 0:N-2], 0:N-2)

	return G, Ft, L_cholesky, γ,  χ, α, β

end


## Parameters
# -----------
# N = 5
# L = 1
# μ = 0.5
# R = 1
# η_2 = 1
# G_feas, Ft_feas, L_cholesky_feas, γ_feas,  χ_feas, α_feas, β_feas = feasible_sol_generator_NCG_PEP_exact_line_search(N, μ, L, η_2)


## Solve the NCG PEP for exact line search
# ========================================

function NCG_PEP_exact_line_search_solver(
	# different parameters to be used
	# -------------------------------
	N, μ, L, R, idx_set_λ_ws_effective,
	# warm-start values for the variables
	G_ws, Ft_ws, L_cholesky_ws, γ_ws,  χ_ws, α_ws, β_ws
  # G_up_bd, Ft_up_bd, L_cholesky_up_bd, χ_up_bd, β_up_bd
	;
	# options
	# -------
	# fix_some_variables = :off,
	η = 1, # controls PRP or FR, if η = 1 => PRP, and if η = 0 => FR
	ξ = 0, # # controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if not we just use a bound of the form ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
	c_0 = Inf, # decides ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
	bound_impose = :off, # other option :on
	# algorithm_type = :PRP, # other option FR (Fletcher-Reeves)
	solution_type =  :find_locally_optimal, # other option :find_globally_optimal
	show_output = :on, # other option :on
	local_solver = :knitro, # other option :knitro
	knitro_multistart = :off, # other option :on (only if :knitro solver is used)
	knitro_multi_algorithm = :off, # other option on (only if :knitro solver is used)
	upper_bound_opt_val = polyak_contraction_factor^N*R^2, # this is coming from Borys Polyak's paper
	lower_bound_opt_val = 0,
	reduce_index_set_for_λ = :off,
	find_feasible_sol_only = :off,
	PSDness_modeling = :exact, # options are :exact and :through_ϵ and :lazy_constraint_callback
	ϵ_tol_feas = 1e-4, # feasiblity tolerance for minimum eigenvalue of G
	maxCutCount=1e6, # number of lazy cuts when we use lazy constraints for modeling G = L_cholesky*L_cholesky
	# options for reduce_index_set_for_λ
	# (i) :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective),
	# (ii) :off , this will define λ and warm-start over the full index set
	# (iii) :for_warm_start_only , this option is the same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
  fix_β = :off, # options are :on and :off
	β_fixed = []
	)

	# if algorithm_type == :PRP
	# 	η_2 = 1
	# elseif algorithm_type == :FR
	# 	η_2 = 0
	# else
	# 	@error("algorithm_type must be either :PRP or :FR")
	# end

	## Some hard-coded parameter

	⋆ = -1

	## Number of points etc
	# ---------------------
	I_N_star = -1:N
	dim_G = N+3
	dim_Ft = N+1
	dim_𝐱 = N+3
	𝐱_0 = e_i(dim_𝐱, 1)
	𝐱_star = zeros(dim_𝐱, 1)

	## Solution type


	if solution_type == :find_globally_optimal

		@info "[🐌 ] globally optimal solution finder activated, solution method: spatial branch and bound"

		NCG_PEP_exact_model =  Model(Gurobi.Optimizer)
		# using direct_model results in smaller memory allocation
		# we could also use
		# Model(Gurobi.Optimizer)
		# but this requires more memory

		set_optimizer_attribute(NCG_PEP_exact_model, "NonConvex", 2)
		# "NonConvex" => 2 tells Gurobi to use its nonconvex algorithm

		set_optimizer_attribute(NCG_PEP_exact_model, "MIPFocus", 3)
		# If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
		# If you believe the solver is having no trouble finding the optimal solution, and wish to focus more
		# attention on proving optimality, select MIPFocus=2.
		# If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

		# 🐑: other Gurobi options one can play with
		# ------------------------------------------

		# turn off all the heuristics (good idea if the warm-starting point is near-optimal)
		# set_optimizer_attribute(NCG_PEP_exact_model, "Heuristics", 0)
		# set_optimizer_attribute(NCG_PEP_exact_model, "RINS", 0)

		# other termination epsilons for Gurobi
		# set_optimizer_attribute(NCG_PEP_exact_model, "MIPGapAbs", 1e-4)

		set_optimizer_attribute(NCG_PEP_exact_model, "MIPGap", 1e-2) # 99% optimal solution, because Gurobi will provide a result associated with a global lower bound within this tolerance, by polishing the result, we can find the exact optimal solution by solving a convex SDP

		# set_optimizer_attribute(NCG_PEP_exact_model, "FuncPieceRatio", 0) # setting "FuncPieceRatio" to 0, will ensure that the piecewise linear approximation of the nonconvex constraints lies below the original function

		# set_optimizer_attribute(NCG_PEP_exact_model, "Threads", 20) # how many threads to use at maximum
		#
		set_optimizer_attribute(NCG_PEP_exact_model, "FeasibilityTol", 1e-3)
		#
		# set_optimizer_attribute(NCG_PEP_exact_model, "OptimalityTol", 1e-4)

	elseif solution_type == :find_locally_optimal

		@info "[🐙 ] locally optimal solution finder activated, solution method: interior point method"

		if local_solver == :knitro

			@info "[🚀 ] activating KNITRO"

			# NCG_PEP_exact_model = Model(optimizer_with_attributes(KNITRO.Optimizer, "convex" => 0,  "strat_warm_start" => 1))

			NCG_PEP_exact_model = Model(
			optimizer_with_attributes(
			KNITRO.Optimizer,
			"convex" => 0,
			"strat_warm_start" => 1,
			# the last settings below are for larger N
			# you can comment them out if preferred but not recommended
			"honorbnds" => 1,
			# "bar_feasmodetol" => 1e-3,
			"feastol" => ϵ_tol_feas,
			# "infeastol" => 1e-12,
			# "opttol" => 1e-4
			)
			)

			if knitro_multistart == :on
				set_optimizer_attribute(NCG_PEP_exact_model, "ms_enable", 1)
				set_optimizer_attribute(NCG_PEP_exact_model, "par_numthreads", 8)
				set_optimizer_attribute(NCG_PEP_exact_model, "par_msnumthreads", 8)
				# set_optimizer_attribute(NCG_PEP_exact_model, "ms_maxsolves", 200)
			end

			if knitro_multi_algorithm == :on
				set_optimizer_attribute(NCG_PEP_exact_model, "algorithm", 5)
				set_optimizer_attribute(NCG_PEP_exact_model, "ma_terminate", 0)
			end

		elseif local_solver == :ipopt

			@info "[🎃 ] activating IPOPT"

			NCG_PEP_exact_model = Model(Ipopt.Optimizer)

		end
	end

	## Define all the variables

	# add the variables
	# -----------------

	@info "[🎉 ] defining the variables"

	# construct G ⪰ 0
	@variable(NCG_PEP_exact_model, G[1:dim_G, 1:dim_G], Symmetric)

	# define the cholesky matrix of Z: L_cholesky
	# -------------------------------------------
	@variable(NCG_PEP_exact_model, L_cholesky[1:dim_G, 1:dim_G])

	# construct Ft (this is just transpose of F)
	@variable(NCG_PEP_exact_model, 0 <= Ft[1:dim_Ft])

  # Construct γ
  @variable(NCG_PEP_exact_model, γ[0:N-1])

	# Construct χ
	@variable(NCG_PEP_exact_model, χ[0:N-2, 1:N-1])

	# Construct α
	@variable(NCG_PEP_exact_model, α[1:N, 0:N-1])

	# Construct β
	@variable(NCG_PEP_exact_model, β[0:N-2])

	# construct the hypograph variables
	@variable(NCG_PEP_exact_model,  t >= lower_bound_opt_val)

	if upper_bound_opt_val <= 10
		@info "[🃏 ] Adding upper bound on objective to be maximized"
		set_upper_bound(t, upper_bound_opt_val)
	end

	# Number of interpolation constraints

	if reduce_index_set_for_λ == :off
		# define λ over the full index set
		idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)
	elseif reduce_index_set_for_λ == :on
		# @error "this part is yet to be written"
		idx_set_λ = idx_set_nz_λ_constructor(N)
		# define λ over a reduced index set, idx_set_λ_ws_effective, which is the effective index set of λ_ws
		# idx_set_λ = idx_set_λ_ws_effective
	end

	# Define Θ[i,j] entries
	# Define Θ[i,j] matrices such that for i,j ∈ I_N_star, we have Θ[i,j] = ⊙(𝐱[:,i] -𝐱[:,j], 𝐱[:,i] - 𝐱[:,j])

	Θ = NCG_PEP_exact_model[:Θ] = reshape(
	hcat([
	@variable(NCG_PEP_exact_model, [1:dim_𝐱, 1:dim_𝐱], Symmetric, base_name = "Θ[$i_j_λ]")
	for i_j_λ in idx_set_λ]...), dim_𝐱, dim_𝐱, length(idx_set_λ))

	# create the data generator

	@info "[🎍 ] adding the data generator function to create 𝐱, 𝐠, 𝐟"

	𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function_NCG_PEP_exact_line_search(N, γ, α, β, χ; 	ξ_input_data_generator = ξ # controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if ξ = 0 then we just use a bound of the form
		# ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
	)

	## Time to add the constraints one by one

	@info "[🎢 ] adding the constraints"

	## add the constraint related to Θ
	# -------------------------------
	# add the constraints corresponding to Θ: (∀(i,j) ∈ idx_set_λ)
	# Θ[:,:,position_of_(i,j)_in_idx_set_λ] ==  ⊙(𝐱[:,i] -𝐱[:,j], 𝐱[:,i] - 𝐱[:,j])
	# --------------------------------
	# Note
	# ----
	# we can access Θ[i,j] by calling
	# Θ[:,:,index_finder_Θ(i, j, idx_set_λ)]

	conΘ = map(1:length(idx_set_λ)) do ℓ
		i_j_λ = idx_set_λ[ℓ]
		@constraint(NCG_PEP_exact_model, vectorize(
		Θ[:,:,ℓ] - ⊙(𝐱[:,i_j_λ.i]-𝐱[:,i_j_λ.j], 𝐱[:,i_j_λ.i]-𝐱[:,i_j_λ.j]),
		SymmetricMatrixShape(dim_𝐱)) .== 0)
	end

	# Implement symmetry in Θ, i.e., Θ[i,j]=Θ[j,i]

	conΘSymmetry = map(1:length(idx_set_λ)) do ℓ
		i_j_λ = idx_set_λ[ℓ]
		if i_j_idx(i_j_λ.j, i_j_λ.i) in idx_set_λ
			@constraint(NCG_PEP_exact_model, vectorize(
			Θ[:,:,index_finder_Θ(i_j_λ.i, i_j_λ.j, idx_set_λ)]
			-
			Θ[:,:,index_finder_Θ(i_j_λ.j, i_j_λ.i, idx_set_λ)],
			SymmetricMatrixShape(dim_𝐱)
			) .== 0 )
		end
	end # end the do map

	## Add the constraints related to G

	if PSDness_modeling == :exact

			# direct modeling through definition and vectorization
			# ---------------------------------------------------
			@constraint(NCG_PEP_exact_model, vectorize(G - (L_cholesky * L_cholesky'), SymmetricMatrixShape(dim_G)) .== 0)

	elseif PSDness_modeling == :through_ϵ

			# definition modeling through vectorization and ϵ_tol_feas

			# part 1: models Z-L_cholesky*L_cholesky <= ϵ_tol_feas*ones(dim_G,dim_G)
			@constraint(NCG_PEP_exact_model, vectorize(G - (L_cholesky * L_cholesky') - ϵ_tol_feas*ones(dim_G,dim_G), SymmetricMatrixShape(dim_G)) .<= 0)

			# part 2: models Z-L_cholesky*L_cholesky >= -ϵ_tol_feas*ones(dim_G,dim_G)

			@constraint(NCG_PEP_exact_model, vectorize(G - (L_cholesky * L_cholesky') + ϵ_tol_feas*ones(dim_G,dim_G), SymmetricMatrixShape(dim_G)) .>= 0)

	elseif PSDness_modeling == :lazy_constraint_callback

				# set_optimizer_attribute(NCG_PEP_exact_model, "FuncPieces", -2) # FuncPieces = -2: Bounds the relative error of the approximation; the error bound is provided in the FuncPieceError attribute. See https://www.gurobi.com/documentation/9.1/refman/funcpieces.html#attr:FuncPieces

				# set_optimizer_attribute(NCG_PEP_exact_model, "FuncPieceError", 0.1) # relative error

				set_optimizer_attribute(NCG_PEP_exact_model, "MIPFocus", 1) # focus on finding good quality feasible solution

				# add initial cuts
				num_cutting_planes_init = (5*dim_G^2)
				cutting_plane_array = [eigvecs(G_ws) rand(Uniform(-1,1), dim_G, num_cutting_planes_init)] # randn(dim_G,num_cutting_planes_init)
				num_cuts_array_rows, num_cuts = size(cutting_plane_array)
				for i in 1:num_cuts
						d_cut = cutting_plane_array[:,i]
						d_cut = d_cut/norm(d_cut,2) # normalize the cutting plane vector
						@constraint(NCG_PEP_exact_model, tr(G*(d_cut*d_cut')) >= 0)
				end

				cutCount=0
				# maxCutCount=1e3

				# add the lazy callback function
				# ------------------------------
				function add_lazy_callback(cb_data)
						if cutCount<=maxCutCount
								G0 = zeros(dim_G,dim_G)
								for i=1:dim_G
										for j=1:dim_G
												G0[i,j]=callback_value(cb_data, G[i,j])
										end
								end

								eig_abs_max_G0 = max(abs(eigmax(G0)), abs(eigmin(G0)))

								if (eigvals(G0)[1]) <= -ϵ_tol_feas # based on the normalized smallest eigenvalue that matters
								# if (eigvals(G0)[1]) <= -1e-4
										u_t = eigvecs(G0)[:,1]
										u_t = u_t/norm(u_t,2)
										con3 = @build_constraint(tr(G*u_t*u_t') >=0.0)
										MOI.submit(NCG_PEP_exact_model, MOI.LazyConstraint(cb_data), con3)
										## try adding a random cut as well
										# u_t_rand = u_t + rand(Uniform(-1,1),length(u_t))
										# u_t_rand = u_t_rand/norm(u_t_rand,2)
										# con3rand = @build_constraint(tr(G*u_t_rand*u_t_rand') >=0.0)
										# MOI.submit(NCG_PEP_exact_model, MOI.LazyConstraint(cb_data), con3rand)
										# noPSDCuts+=1
								end
								# if eigvals(G0)[2]<=-1e-2
								# 		u_t = eigvecs(G0)[:,2]
								# 		u_t = u_t/norm(u_t,2)
								# 		con4 = @build_constraint(tr(G*u_t*u_t') >=0.0)
								# 		MOI.submit(NCG_PEP_exact_model, MOI.LazyConstraint(cb_data), con4)
								# end
								cutCount+=1
						end
				end

				# submit the lazy constraint
				# --------------------------
				MOI.set(NCG_PEP_exact_model, MOI.LazyConstraintCallback(), add_lazy_callback)

	else

			@error "something is not right in PSDness_modeling"

			return

	end

	# conG = @constraint(NCG_PEP_exact_model, vectorize(G - L_cholesky*L_cholesky', SymmetricMatrixShape(dim_G)) .== 0)

	# Add the constraint realted to the initial condition
	# B_mat_exact_line_search_0_minus1 = ⊙(𝐱_0-𝐱_star, 𝐱_0-𝐱_star)

	# Initial condition: f(x_0) - f(x_⋆) == 1

	@constraint(NCG_PEP_exact_model, conInit,
	(Ft'*a_vec_exact_line_search(⋆,0, 𝐟))
	== R^2
	)

	## Constraint
	#  ||g_0||^2 <= ||p_0||^2
	# ⇔ tr G C_{0,⋆} <= tr G C_tilde_{0,⋆}

	@constraint(NCG_PEP_exact_model, con_p0_sqd_lb, tr(G*C_mat_exact_line_search(0, ⋆ , 𝐠)) <= tr(G*C_tilde_mat_exact_line_search(0, ⋆, χ, 𝐩)))

	## Constraint
	# ||p_0||^2 <= c_0 ||g_0||^2
	# ⇔ tr G C_tilde_{0,⋆} <= c_0*tr G C_{0,⋆}

	@constraint(NCG_PEP_exact_model, con_p0_sqd_ub, tr(G*C_tilde_mat_exact_line_search(0, ⋆, χ, 𝐩)) <= c_0*tr(G*C_mat_exact_line_search(0, ⋆, 𝐠)))

	## Constraint
	# ⟨ g_1 ; p_0⟩ = 0
	# ⇔ tr G D_tilde_{1,0} = 0

	@constraint(NCG_PEP_exact_model, con_line_search_init_1, tr(G*D_tilde_mat_exact_line_search(1, 0, χ, 𝐠, 𝐩)) == 0)

	## Constraint
	# ⟨ g_1 ; x_0 - x_1⟩ = 0
	# ⇔ tr G A_{0,1} == 0

	@constraint(NCG_PEP_exact_model, con_line_search_init_2, tr(G*A_mat_exact_line_search(0, 1, α, 𝐠, 𝐱)) == 0)

	## Constraint
	# ⟨ g_0 ; p_0⟩ = ||g_0||^2
	# ⇔ tr G D_tilde_{0,0} = tr G C_{0,⋆}

	@constraint(NCG_PEP_exact_model, con_g0_p0_relation, tr(G*D_tilde_mat_exact_line_search(0, 0, χ, 𝐠, 𝐩)) == tr(G*C_mat_exact_line_search(0, ⋆, 𝐠)))


	## Constraints connecting χ and β

	## Constraint ∀i ∈ [1:N-1], j ∈ [0:i-2] χ[j,i] = χ[j,i-1]*β[i-1]

	@constraint(NCG_PEP_exact_model, con_χ_formula_1[i in 1:N-1, j in 0:i-2], χ[j,i] == χ[j,i-1]*β[i-1])

	## Constraint ∀i ∈ [1:N-1] χ[i-1,i] = β[i-1]

	@constraint(NCG_PEP_exact_model, con_χ_formula_2[i in 1:N-1], χ[i-1, i] == β[i-1])

	## Constraints connecting α, γ, and χ

	# first constraint connecting α and γ
	# ∀i∈[1:N] α[i,i-1] = γ[i-1]

	@constraint(NCG_PEP_exact_model, con_α_γ_1[i=1:N], α[i,i-1] == γ[i-1])

	# second constraint connecting α, γ, and χ
	# ∀i∈[1:N] ∀j∈[0:i-2] α[i,j] = γ[j] + sum(γ[k]*χ[j,k] for k in j+1:i-1)

	@constraint(NCG_PEP_exact_model, con_α_γ_2[i=1:N, j=0:i-2], α[i,j] == γ[j] + sum(γ[k]*χ[j,k] for k in j+1:i-1))

	# Constraints that defines β

	## Constraint
	# ∀ i ∈ [1:N-1] β_{i-1}*||g_{i-1}||^2 = ||g_i||^2 - η* ⟨ g_i ∣ g_{i-1} ⟩
	# ⇔ i ∈ [1:N-1] β_{i-1}* tr G C_{i-1, ⋆} = tr G (C_{i,⋆} - η*D_{i,i-1})

	@constraint(NCG_PEP_exact_model, con_β_formula[i in 1:N-1], β[i-1] * tr(G*C_mat_exact_line_search(i-1, ⋆, 𝐠)) == tr(G * (C_mat_exact_line_search(i, ⋆, 𝐠) - η*D_mat_exact_line_search(i, i-1, 𝐠))) )

	## Constraint
	# ∀ i ∈ [1:N-1] ⟨ g_{i+1} ; p_{i}⟩ = 0
	# ⇔ tr G D_tilde_{i+1, i} = 0

	@constraint(NCG_PEP_exact_model, con_line_search_i_1[i in 1:N-1], tr(G*D_tilde_mat_exact_line_search(i+1, i, χ, 𝐠, 𝐩)) == 0)

	## Constraint
	# ∀ i ∈ [1:N-1] ⟨ g_{i+1} ; x_i - x_{i+1}⟩ = 0
	# ⇔ ∀ i ∈ [1:N-1] tr G A_{i, i+1} = 0

	@constraint(NCG_PEP_exact_model, con_line_search_i_2[i in 1:N-1], tr(G*A_mat_exact_line_search(i, i+1, α, 𝐠, 𝐱)) == 0)

	## Constraint
	# ∀ i ∈ [1:N-1]  ⟨ g_{i} ; p_{i}⟩ = ||g_{i}||^2
	# ⇔ ∀ i ∈ [1:N-1]  tr G D_tilde_{i, i} = tr G C_{i, ⋆}

	@constraint(NCG_PEP_exact_model, con_gi_pi_relation[i in 1:N-1], tr(G*D_tilde_mat_exact_line_search(i, i, χ, 𝐠, 𝐩)) == tr(G*C_mat_exact_line_search(i, ⋆, 𝐠)))

	## interpolation constraint

	## Interpolation constraint
	## f_i >= f_j + ⟨ g_j ; x_i - x_j ⟩ + 1/(2*(1-q)) ( 1/L ||g_i - g_j||^2 + μ ||x_i - x_j||^2 - 2 q ⟨ g_i - g_j ; x_i - x_j⟩), where q = μ/L

	q = μ/L

	conInterpol = map(1:length(idx_set_λ)) do ℓ
		i_j_λ = idx_set_λ[ℓ]
		@constraint(NCG_PEP_exact_model,
		Ft'*a_vec_exact_line_search(i_j_λ.i,i_j_λ.j,𝐟)
		+ tr(G * (
		A_mat_exact_line_search(i_j_λ.i,i_j_λ.j,α,𝐠,𝐱)
		+ ( 1/(2*(1-q)) )*(
		(1/(L))*C_mat_exact_line_search(i_j_λ.i,i_j_λ.j,𝐠)
		+	(μ*Θ[:,:,index_finder_Θ(i_j_λ.i, i_j_λ.j, idx_set_λ)])
		- (2*q*E_mat_exact_line_search(i_j_λ.i,i_j_λ.j,α,𝐠,𝐱))
		)
		)
		)
		<= 0)
	end # end the do map

	## Hypograph constraint

	@constraint(NCG_PEP_exact_model, hypographCon, t - Ft'*a_vec_exact_line_search(-1,N, 𝐟) <= 0
	)

	## List of valid constraints for G

	# diagonal components of G are non-negative

	@constraint(NCG_PEP_exact_model, conNonNegDiagG[i=1:dim_G], G[i,i] >= 0)


	# the off-diagonal components satisfy:
	# (∀i,j ∈ dim_G: i != j) -(0.5*(G[i,i] + G[j,j])) <= G[i,j] <=  (0.5*(G[i,i] + G[j,j]))

	for i in 1:dim_G
		for j in 1:dim_G
			if i != j
				@constraint(NCG_PEP_exact_model, G[i,j] <= (0.5*(G[i,i] + G[j,j])) )
				@constraint(NCG_PEP_exact_model, -(0.5*(G[i,i] + G[j,j])) <= G[i,j] )
			end
		end
	end

	# Two constraints to define the matrix L_cholesky to be a lower triangular matrix
	# -------------------------------------------------

	# upper off-diagonal terms of L_cholesky are zero

	for i in 1:dim_G
		for j in 1:dim_G
			if i < j
				fix(L_cholesky[i,j], 0; force = true)
			end
		end
	end

	# diagonal components of L_cholesky are non-negative

	for i in 1:dim_G
		@constraint(NCG_PEP_exact_model, L_cholesky[i,i] >= 0)
	end

	## add the mathematical bound for the entries of G

	@constraint(NCG_PEP_exact_model, G[1,1] <= (2/μ)*Ft[1])
	@constraint(NCG_PEP_exact_model, G[1,1] >= (2/L)*Ft[1])

	for i in 2:N+2
		@constraint(NCG_PEP_exact_model, G[i,i] <= 2*L*Ft[i-1])
		@constraint(NCG_PEP_exact_model, G[i,i] >= 2*μ*Ft[i-1])
	end

	## Add the objective

	if find_feasible_sol_only == :off
		  @info "[🎇 ] adding objective"
    	@objective(NCG_PEP_exact_model, Max, t)
	elseif find_feasible_sol_only == :on
		  @info "finding a feasible solution only"
	end

	## Time to warm-start all the variables

	## Time to warm-start the variables

	@info "[👲 ] warm-start values for all the variables"

	# warm-start G

	for i in 1:dim_G
		for j in 1:dim_G
			set_start_value(G[i,j], G_ws[i,j])
		end
	end

	# warm-start Ft

	for i in 1:dim_Ft
		set_start_value(Ft[i], Ft_ws[i])
	end

	# warm_start γ

	for i in 0:N-1
		set_start_value(γ[i], γ_ws[i])
	end

	# warm-start χ

	for k in 1:N-1
		for j in 0:N-2
			set_start_value(χ[j,k], χ_ws[j,k])
		end
	end

	# warm-start α

	for i in 1:N
		for j in 0:N-1
			set_start_value(α[i,j], α_ws[i,j])
		end
	end

	# warm-start β

	for i in 0:N-2
		set_start_value(β[i], β_ws[i])
	end

	# warm-start L_cholesky

	for i in 1:dim_G
		for j in 1:dim_G
			set_start_value(L_cholesky[i,j], L_cholesky_ws[i,j])
		end
	end

	# warm start for Θ
	# ----------------

	# construct 𝐱_ws, 𝐠_ws, 𝐟_ws corresponding to γ_ws
	𝐱_ws, 𝐠_ws, 𝐟_ws = data_generator_function_NCG_PEP_exact_line_search(N, γ_ws, α_ws, β_ws, χ_ws)

	# construct Θ_ws step by step
	Θ_ws = zeros(dim_𝐱, dim_𝐱, length(idx_set_λ))

	for ℓ in 1:length(idx_set_λ)
		i_j_λ = idx_set_λ[ℓ]
		Θ_ws[:,:,ℓ] = ⊙(𝐱_ws[:,i_j_λ.i]-𝐱_ws[:,i_j_λ.j], 𝐱_ws[:,i_j_λ.i]-𝐱_ws[:,i_j_λ.j])
	end
	# setting the warm-start value for Θ_ws

	for ℓ in 1:length(idx_set_λ)
		i_j_λ = idx_set_λ[ℓ]
		set_start_value.(Θ[:,:,ℓ], Θ_ws[:,:,ℓ])
	end

	# warm start t

	t_ws =  Ft_ws[end] #(c_f*(Ft_ws*a_vec_exact_line_search(-1,N, 𝐟)))[1]

	set_start_value(t, t_ws)

	# Check if all the variables have been warm-started

	if any(isnothing, start_value.(all_variables(NCG_PEP_exact_model))) == true
		@error "all the variables are not warm-started"
	end

	if bound_impose == :on
			@info "[🌃 ] finding bound on the variables"

			# store the values

			λ_lb = 0
			λ_ub = M_λ
			τ_lb = 0
			τ_ub = M_τ
			η_lb = 0
			η_ub = M_η
			ν_lb = 0
			ν_ub = ν_ws
			Z_lb = -M_Z
			Z_ub = M_Z
			L_cholesky_lb = -M_L_cholesky
			L_cholesky_ub = M_L_cholesky
			α_lb = -M_α
			α_ub = M_α
			Θ_lb = -M_Θ
			Θ_ub = M_Θ

			# set bound for λ
			# ---------------
			# set_lower_bound.(λ, λ_lb): done in definition
			set_upper_bound.(λ, λ_ub)

			# set bound for τ
			# set_lower_bound.(τ, τ_lb): done in definition
			set_upper_bound.(τ, τ_ub)

			# set bound for η
			#  set_lower_bound.(η, η_lb): done in definition
			set_upper_bound.(η, η_ub)

			# set bound for ν
			# ---------------
			# set_lower_bound.(ν, ν_lb): done in definition
			set_upper_bound(ν, ν_ub)

			# set bound for Z
			# ---------------
			for i in 1:dim_Z
					for j in 1:dim_Z
							set_lower_bound(Z[i,j], Z_lb)
							set_upper_bound(Z[i,j], Z_ub)
					end
			end

			# set bound for L_cholesky
			# ------------------------

			if find_global_lower_bound_via_cholesky_lazy_constraint == :off
					# need only upper bound for the diagonal compoments, as the lower bound is zero from the model
					for i in 1:N+2
							set_upper_bound(L_cholesky[i,i], L_cholesky_ub)
					end
					# need to bound only components, L_cholesky[i,j] with i > j, as for i < j, we have zero, due to the lower triangular structure
					for i in 1:N+2
							for j in 1:N+2
									if i > j
											set_lower_bound(L_cholesky[i,j], L_cholesky_lb)
											set_upper_bound(L_cholesky[i,j], L_cholesky_ub)
									end
							end
					end
			end

			# set bound for Θ
			# ---------------
			set_lower_bound.(Θ, Θ_lb)
			set_upper_bound.(Θ, Θ_ub)

			# set bound for α
			# ---------------
			set_lower_bound.(α, α_lb)
			set_upper_bound.(α, α_ub)

	end

	# fix β if fix_β == :on

	if fix_β == :on

		@info "[💁 ] Fixing β and finding the corresponding solution"

		for i in 0:N-2
		   fix.(β[i], β_fixed[i]; force = true)
		end

	end

	# time to optimize
	# ----------------

	@info "[🙌 	🙏 ] model building done, starting the optimization process"

	if show_output == :off
		set_silent(NCG_PEP_exact_model)
	end

	optimize!(NCG_PEP_exact_model)


	@info "NCG_PEP_exact_model has termination status = " termination_status(NCG_PEP_exact_model)

	## Time to store the solution

	if (solution_type == :find_locally_optimal && termination_status(NCG_PEP_exact_model) == LOCALLY_SOLVED) || (solution_type == :find_locally_optimal && termination_status(NCG_PEP_exact_model) == SLOW_PROGRESS) ||
	(solution_type == :find_locally_optimal && termination_status(NCG_PEP_exact_model) == ITERATION_LIMIT) || (solution_type ==:find_globally_optimal && termination_status(NCG_PEP_exact_model) == OPTIMAL )

		# store the solutions and return
		# ------------------------------

		@info "[😻 ] optimal solution found done, store the solution"

		# store G_opt

		G_opt = value.(G)

		# store F_opt

		Ft_opt = value.(Ft)

		# store L_cholesky_opt

		L_cholesky_opt = value.(L_cholesky)

		if norm(G_opt - L_cholesky_opt*L_cholesky_opt', Inf) > 10^-4
			@warn "||G - L_cholesky*L_cholesky^T|| = $(norm(G_opt -  L_cholesky_opt*L_cholesky_opt', Inf))"
		end

		# store γ_opt

		γ_opt = value.(γ)

		# store χ_opt

		χ_opt = value.(χ)

		# store α_opt

		α_opt = value.(α)

		# store β_opt

		β_opt = value.(β)

		@info "[💹 ] warm-start objective value = $t_ws, and objective value of found solution = $(Ft_opt[end])"

		t_opt = value.(t)

	else

		@error "[🙀 ] could not find an optimal solution"

	end

	## Time to return the solution

	return G_opt, Ft_opt, L_cholesky_opt, γ_opt,  χ_opt, α_opt, β_opt

end


# N = 3
# μ = 0.1
# L = 1
# R = 1
# bigM = 1e6
# η_2 = 1
# q = μ/L
# polyak_contraction_factor = (1 - (μ/L)* (1/(1 + (L/μ)^2)))
# G_ws, Ft_ws, L_cholesky_ws, γ_ws,  χ_ws, α_ws, β_ws = feasible_sol_generator_NCG_PEP_exact_line_search(N, μ, L, η_2)
#
# G_opt, Ft_opt, L_cholesky_opt, γ_opt,  χ_opt, α_opt, β_opt = NCG_PEP_solver(
# 	# different parameters to be used
# 	# -------------------------------
# 	N, μ, L, R, idx_set_λ_ws_effective,
# 	G_ws, Ft_ws, L_cholesky_ws, γ_ws,  χ_ws, α_ws, β_ws;
# 	# options
# 	# -------
# 	solution_type =  :find_locally_optimal, # other option :find_globally_optimal
# 	show_output = :on, # other option :on
# 	local_solver = :knitro, # other option :knitro
# 	knitro_multistart = :off, # other option :on (only if :knitro solver is used)
# 	knitro_multi_algorithm = :off, # other option on (only if :knitro solver is used)
# 	upper_bound_opt_val = polyak_contraction_factor^N*R^2, # this is coming from Borys Polyak's paper
# 	lower_bound_opt_val = 0,
# 	reduce_index_set_for_λ = :off
# 	# options for reduce_index_set_for_λ
# 	# (i) :on (making it :on will make force λ[i,j] = 0, if (i,j) ∉ idx_set_λ_feas_effective),
# 	# (ii) :off , this will define λ and warm-start over the full index set
# 	# (iii) :for_warm_start_only , this option is the same as the :off option, however in this case we will define λ over the full index set, but warm-start from a λ_ws that has reduced index set
# 	)


## Time to generate the worst-case function for exact line search.
# ===============================================================

function generate_worst_case_function_NCG_PEP_exact_line_search(
N, G, Ft, L_cholesky, γ,  χ, α, β;
ξ = 0
# controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if ξ = 0 then we just use a bound of the form
# ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
)

		𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function_NCG_PEP_exact_line_search(N, γ, α, β, χ; ξ_input_data_generator = ξ)

		H = convert(Array{Float64,2}, LinearAlgebra.transpose(L_cholesky))

		d, _ = size(H)

		x_array = OffsetArray(zeros(d,N+2), 1:d, -1:N)

		g_array = OffsetArray(zeros(d,N+2), 1:d, -1:N)

		f_array = OffsetVector(zeros(N+2), -1:N)

		for i in 0:N
		    x_array[:, i] =  H*𝐱[:, i]
		    g_array[:, i] = H*𝐠[:, i]
		    f_array[i] = Ft[i+1]
		end

		wf = worst_case_function(x_array, g_array, f_array)

		return wf

end


## Data generator for solving SDP with fixed stepsize for exact line search
# =========================================================================

function data_generator_function_for_SDP_with_fixed_stepsize_NCG_PEP_exact_line_search(N, γ, β; 	ξ_input_data_generator = 0 # controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if ξ = 0 then we just use a bound of the form
	# ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
)

	# define all the bold vectors
	# --------------------------

	# define 𝐱_0 and 𝐱_star

	dim_𝐠 = N+3

	dim_𝐱 = N+3

	dim_𝐩 = N+3

	dim_𝐟 = N+1

	𝐱_0 = e_i(dim_𝐱, 1)

	𝐱_star = zeros(dim_𝐱, 1)

	# define 𝐠_0, 𝐠_1, …, 𝐠_N

	# first we define the 𝐠 vectors,
	# index -1 corresponds to ⋆, i.e.,  𝐟[:,-1] =  𝐟_⋆ = 0

	# 𝐠= [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]

	𝐠 = OffsetArray(zeros(dim_𝐠, N+2), 1:dim_𝐠, -1:N)

	for i in 0:N
		𝐠[:,i] = e_i(dim_𝐠, i+2)
	end

	# time to define the 𝐟 vectors

	# 𝐟 = [𝐟_⋆ 𝐟_0, 𝐟_1, …, 𝐟_N]

	𝐟 = OffsetArray(zeros(dim_𝐟, N+2), 1:dim_𝐟, -1:N)

	for i in 0:N
		𝐟[:,i] = e_i(dim_𝐟, i+1)
	end

	# 𝐩 corresponds to the pseudo-gradient g̃
	# 𝐩 = [𝐩_⋆ 𝐩_0 𝐩_1 𝐩_2 ... 𝐩_{N-1}] ∈ 𝐑^(N+2 × N+1)

	𝐩 = OffsetArray(Matrix{Any}(undef, dim_𝐩, N+1), 1:dim_𝐩, -1:N-1)

	# assign values next using our formula for 𝐩_k
	𝐩[:,-1] = zeros(dim_𝐩, 1)

	if ξ_input_data_generator == 0 # ξ_input_data_generator == 0 => no restart
			𝐩[:, 0] = e_i(dim_𝐩, N+3)
	elseif ξ_input_data_generator == 1 # ξ_input_data_generator == 1 => restart
			𝐩[:, 0] = 𝐠[:, 0]
	else
			@error "ξ_input_data_generator has to be equal to 0 or 1"
			return
	end

	# 𝐩[:,0] = 𝐠[:,0]

  for i in 0:N-2
		𝐩[:, i+1] = 𝐠[:, i+1] + β[i]*𝐩[:,i]
	end

	# 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}] ∈ 𝐑^(N+2 × N+2)
	𝐱  = OffsetArray(Matrix{Any}(undef, dim_𝐱, N+2), 1:dim_𝐱, -1:N)

	# assign values next using our formula for 𝐱_k
	𝐱[:,-1] = 𝐱_star

	𝐱[:,0] = 𝐱_0

  for i in 0:N-1
		𝐱[:, i+1] = 𝐱[:, i] - γ[i]*𝐩[:, i]
	end

	return 𝐱, 𝐠, 𝐟, 𝐩

end


# Time to solve the SDP with the stepsize as an input
# ===================================================

function SDP_with_fixed_stepsize_NCG_PEP_exact_line_search(
	        # Inputs
					# ======
					N, μ, L, R, γ,  χ, α, β,
          # Warm-start points
					G_ws_els, Ft_ws_els; # [🐯 ] cost coefficients
					# Options
					# =======
					η = 1, # controls PRP or FR, if η = 1 => PRP, and if η = 0 => FR
					ξ = 0, # # controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if not we just use a bound of the form ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
					c_0 = Inf, # decides ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
					show_output = :on, # options are :on and :off
					ϵ_tolerance = 1e-8,
					reduce_index_set_for_λ = :off
          )

	## Some hard-coded parameter

	⋆ = -1

	## Number of points etc
	# ---------------------
	I_N_star = -1:N
	dim_G = N+3
	dim_Ft = N+1
	dim_𝐱 = N+3
	𝐱_0 = e_i(dim_𝐱, 1)
	𝐱_star = zeros(dim_𝐱, 1)

	# polyak_contraction_factor = (1 - (μ/L)* (1/(1 + (L/μ)^2)))

	## data generator
	# --------------

	𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function_for_SDP_with_fixed_stepsize_NCG_PEP_exact_line_search(N, γ, β;
	ξ_input_data_generator = ξ # controls restarting scheme: if ξ = 1 => NCG method is restarted at iteration k(=0 in code), if ξ = 0 then we just use a bound of the form
		# ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
	)

	# define the model
	# ================

	# model_primal_PEP_with_known_stepsizes = Model(optimizer_with_attributes(SCS.Optimizer))


	# model_primal_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer))

	model_primal_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer, "INTPNT_CO_TOL_DFEAS" => ϵ_tolerance, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => ϵ_tolerance))

	@info "[🎉 ] defining the variables"

	# construct G ⪰ 0
	@variable(model_primal_PEP_with_known_stepsizes, G[1:dim_G, 1:dim_G], PSD)

	# construct Ft (this is just transpose of F)
	@variable(model_primal_PEP_with_known_stepsizes, Ft[1:dim_Ft])

	# Number of interpolation constraints

	# We already have a feasible point from the locally optimal solution
	# @constraint(model_primal_PEP_with_known_stepsizes, Ft[end] >= 0.99*Ft_ws_els[end])

	# Number of interpolation constraints

	if reduce_index_set_for_λ == :off
		# define λ over the full index set
		idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)
	elseif reduce_index_set_for_λ == :on
		# @error "this part is yet to be written"
		idx_set_λ = idx_set_nz_λ_constructor(N)
		# define λ over a reduced index set, idx_set_λ_ws_effective, which is the effective index set of λ_ws
		# idx_set_λ = idx_set_λ_ws_effective
	end

	## Time to add the constraints one by one

	@info "[🎢 ] adding the constraints"

	## Initial condition

	@constraint(model_primal_PEP_with_known_stepsizes, conInit,
	(Ft'*a_vec_exact_line_search(⋆,0, 𝐟))
	<= R^2
	)

	## Constraint
	#  ||g_0||^2 <= ||p_0||^2
	# ⇔ tr G C_{0,⋆} <= tr G C_tilde_{0,⋆}

	@constraint(model_primal_PEP_with_known_stepsizes, con_p0_sqd_lb, tr(G*C_mat_exact_line_search(0, ⋆ , 𝐠)) <= tr(G*C_tilde_mat_exact_line_search(0, ⋆, χ, 𝐩)))

	## Constraint
	# ||p_0||^2 <= c_0 ||g_0||^2
	# ⇔ tr G C_tilde_{0,⋆} <= c_0*tr G C_{0,⋆}

	@constraint(model_primal_PEP_with_known_stepsizes, con_p0_sqd_ub, tr(G*C_tilde_mat_exact_line_search(0, ⋆, χ, 𝐩)) <= c_0*tr(G*C_mat_exact_line_search(0, ⋆, 𝐠)))

	## Constraint
	# ⟨ g_1 ; p_0⟩ = 0
	# ⇔ tr G D_tilde_{1,0} = 0

	@constraint(model_primal_PEP_with_known_stepsizes, con_line_search_init_1, tr(G*D_tilde_mat_exact_line_search(1, 0, χ, 𝐠, 𝐩)) == 0)

	## Constraint
	# ⟨ g_1 ; x_0 - x_1⟩ = 0
	# ⇔ tr G A_{0,1} == 0

	@constraint(model_primal_PEP_with_known_stepsizes, con_line_search_init_2, tr(G*A_mat_exact_line_search(0, 1, α, 𝐠, 𝐱)) == 0)

	## Constraint
	# ⟨ g_0 ; p_0⟩ = ||g_0||^2
	# ⇔ tr G D_tilde_{0,0} = tr G C_{0,⋆}

	@constraint(model_primal_PEP_with_known_stepsizes, con_g0_p0_relation, tr(G*D_tilde_mat_exact_line_search(0, 0, χ, 𝐠, 𝐩)) == tr(G*C_mat_exact_line_search(0, ⋆, 𝐠)))

	## β update parameter

	@constraint(model_primal_PEP_with_known_stepsizes, con_β_formula[i in 1:N-1], β[i-1] * tr(G*C_mat_exact_line_search(i-1, ⋆, 𝐠)) == tr(G * (C_mat_exact_line_search(i, ⋆, 𝐠) - η*D_mat_exact_line_search(i, i-1, 𝐠))) )

	## Line search condition 1

	@constraint(model_primal_PEP_with_known_stepsizes, con_line_search_i_1[i in 1:N-1], tr(G*D_tilde_mat_exact_line_search(i+1, i, χ, 𝐠, 𝐩)) == 0)

	## Constraint
	# ∀ i ∈ [1:N-1] ⟨ g_{i+1} ; x_i - x_{i+1}⟩ = 0
	# ⇔ ∀ i ∈ [1:N-1] tr G A_{i, i+1} = 0

	@constraint(model_primal_PEP_with_known_stepsizes, con_line_search_i_2[i in 1:N-1], tr(G*A_mat_exact_line_search(i, i+1, α, 𝐠, 𝐱)) == 0)

	## Constraint
	# ∀ i ∈ [1:N-1]  ⟨ g_{i} ; p_{i}⟩ = ||g_{i}||^2
	# ⇔ ∀ i ∈ [1:N-1]  tr G D_tilde_{i, i} = tr G C_{i, ⋆}

	@constraint(model_primal_PEP_with_known_stepsizes, con_gi_pi_relation[i in 1:N-1], tr(G*D_tilde_mat_exact_line_search(i, i, χ, 𝐠, 𝐩)) == tr(G*C_mat_exact_line_search(i, ⋆, 𝐠)))

	# Interpolation constraint

	## Time for the interpolation constraint

	q = μ/L

	conInterpol = map(1:length(idx_set_λ)) do ℓ
		i_j_λ = idx_set_λ[ℓ]
		@constraint(model_primal_PEP_with_known_stepsizes,
		Ft'*a_vec_exact_line_search(i_j_λ.i,i_j_λ.j,𝐟)
		+ tr(G * (
		A_mat_exact_line_search(i_j_λ.i,i_j_λ.j,α,𝐠,𝐱)
		+ ( 1/(2*(1-q)) )*(
		(1/(L))*C_mat_exact_line_search(i_j_λ.i,i_j_λ.j,𝐠)
		+	(μ*B_mat_exact_line_search(i_j_λ.i, i_j_λ.j, α, 𝐱))
		- (2*q*E_mat_exact_line_search(i_j_λ.i,i_j_λ.j,α,𝐠,𝐱))
		)
		)
		)
		<= 0)
	end # end the do map


	## Add the objective and the associated hypograph constraint

	# @info "[🎇 ] adding objective"
	# =============================

	@objective(model_primal_PEP_with_known_stepsizes, Max,  Ft[end])

	## Time to warmt-start
	@info "[👲 ] warm-start values for all the variables"

	# warm-start G

	# set_start_value.(G, G_ws_els)
	#
	# # warm-start Ft
	#
	# @show Ft
	#
	# @show Ft_ws_els
	#
	# set_start_value.(Ft, Ft_ws_els)

	## time to optimize!

	@info "Time to optimize"

	optimize!(model_primal_PEP_with_known_stepsizes)

	## store and return the solution
	# -----------------------------

	if termination_status(model_primal_PEP_with_known_stepsizes) == MOI.OPTIMAL || termination_status(model_primal_PEP_with_known_stepsizes) == MOI.SLOW_PROGRESS
		@info "[😈 ] Optimal solution found for solve_primal_with_known_stepsizes"
	else
			@error "model_primal_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_primal_PEP_with_known_stepsizes)
	end

	p_star = objective_value(model_primal_PEP_with_known_stepsizes)

	G_star = value.(G)

	Ft_star = value.(Ft)

	L_cholesky_star =  compute_pivoted_cholesky_L_mat(G_star)

	if norm(G_star - L_cholesky_star*L_cholesky_star', Inf) > 1e-6
		@info "checking the norm bound for feasible {G, L_cholesky}"
		@warn "||G - L_cholesky*L_cholesky^T|| = $(norm(G_star - L_cholesky_star*L_cholesky_star', Inf))"
	end

	@info "[🌞 ] PRIMALSDP-POLISHED ADAPEP contraction factor V_{k+1}/V_{k-1} = $(objective_value(model_primal_PEP_with_known_stepsizes)/R^2)"

	return p_star, G_star, Ft_star, L_cholesky_star

end

