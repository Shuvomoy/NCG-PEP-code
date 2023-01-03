
## Load the packages:
# -------------------
using JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools, DiffOpt, SparseArrays, KNITRO

## Load the pivoted Cholesky finder
# ---------------------------------
include("code_to_compute_pivoted_cholesky.jl")


## Some helper functions

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

# This function is used to generate the first sparse warm start point to compute sparse solution for the dual PEP with fixed stepsize only when we are using the Log-Det-Heuristics

function generate_dual_ws_for_sparse_sol_finder(N, I_N_star)

	idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)

	λ_ws_sparse = JuMP.Containers.DenseAxisArray{Float64}(undef, idx_set_λ)

	fill!(λ_ws_sparse, 1.0)

	# λ_ws_sparse = Dict{Any, Any}()
	#
	# for i_j in idx_set_λ
	# 	λ_ws_sparse[i_j] = 1.00
	# end

	τ_ws_sparse = Dict{Any, Any}()

	for i in 1:N
		τ_ws_sparse[i] = 1.00
	end

	ν_ws_sparse = 1.00

	return λ_ws_sparse, τ_ws_sparse, ν_ws_sparse

end

# usage λ_ws_sparse, τ_ws_sparse, ν_ws_sparse = generate_dual_ws_for_sparse_sol_finder(N, I_N_star)

# another important function to find proper index of Θ given (i,j) pair
index_finder_Θ(i,j,idx_set_λ) = findfirst(isequal(i_j_idx(i,j)), idx_set_λ)


function index_set_zero_entries_dual_variables(idx_set_λ)


    idx_set_nz_λ = [i_j_idx(-1, 0); i_j_idx(-1, 2); i_j_idx(0, 1); i_j_idx(1, 2)]

    idx_set_zero_λ = setdiff(idx_set_λ, idx_set_nz_λ ) # this is the index
# set where the entries of λ will be zero

    idx_set_zero_τ = [1]

		idx_set_zero_ρ = [3; 5]

    return idx_set_zero_λ, idx_set_zero_τ, idx_set_zero_ρ

end


# create a math index to Julia index dictionary as follows
# math index <=> julia index
# *          <=>      -1
# k-1        <=>       0
# k          <=>       1
# so technically we can set k = 1, and the offset array mechanism will work just fine

function data_generator_function_Lyapunov(γ_kM1, β_kM1)

	k = 1

	# define all the bold vectors
	# --------------------------

	# 𝐠= [𝐠_⋆ 𝐠_{k-1} 𝐠_{k} 𝐠_{k+1}]

	dim_𝐠 = 4

	𝐠 = OffsetArray(zeros(dim_𝐠, 3), 1:dim_𝐠, -1:k)

	𝐠[:, -1] = zeros(dim_𝐠, 1)

	𝐠[:, k-1] = e_i(dim_𝐠 , 2)

	𝐠[:, k] = e_i(dim_𝐠 , 3)

	# time to define the 𝐟 vectors

	# 𝐟 = [𝐟_⋆ 𝐟_{k-1} 𝐟_{k}, 𝐟_{k+1}]

	dim_𝐟 = 2

	𝐟 = OffsetArray(zeros(dim_𝐟, 3), 1:dim_𝐟, -1:k)

	𝐟[:, -1] = zeros(dim_𝐟, 1)

	𝐟[:, k-1] = e_i(dim_𝐟, 1)

	𝐟[:, k] = e_i(dim_𝐟, 2)

	# time construct the vectors of pseudo-gradients

	dim_𝐩 = 4

    𝐩 = OffsetArray(Matrix{Any}(undef, dim_𝐩, 3), 1:dim_𝐩, -1:k)

	𝐩[:, -1] = zeros(dim_𝐩, 1)

	𝐩[:, k-1] = e_i(dim_𝐩 , 4)

	𝐩[:, k] = 𝐠[:,k] + β_kM1*𝐩[:, k-1]

	# 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{k-1} ∣ 𝐱_{k} ∣ x_{k+1}] ∈ 𝐑^6×4

	dim_𝐱 = 4

	𝐱  = OffsetArray(Matrix{Any}(undef,dim_𝐱, 3), 1:dim_𝐱, -1:k)

	# assign values next using our formula for 𝐱_k

	𝐱[:, -1] = zeros(dim_𝐱, 1)

	𝐱[:, k-1] = e_i(dim_𝐱, 1)

	𝐱[:, k] = 𝐱[:,k-1] - γ_kM1*𝐩[:,k-1]

	# 𝐱[:,k+1] = 𝐱[:,k] - γ_k*𝐠[:,k] - χ_k*𝐩[:,k-1]

	return 𝐱, 𝐠, 𝐟, 𝐩

end


A_mat(i, j, γ_kM1, 𝐠, 𝐱) = ⊙(𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
B_mat(i, j, γ_kM1, 𝐱) = ⊙(𝐱[:,i]-𝐱[:,j], 𝐱[:,i]-𝐱[:,j])
C_mat(i, j, 𝐠) = ⊙(𝐠[:,i]-𝐠[:,j], 𝐠[:,i]-𝐠[:,j])
C_tilde_mat(i, j, β_kM1, 𝐩) = ⊙(𝐩[:,i]-𝐩[:,j], 𝐩[:,i]-𝐩[:,j])
D_mat(i, j, 𝐠) = ⊙(𝐠[:,i], 𝐠[:,j])
D_tilde_mat(i, j, β_kM1, 𝐠, 𝐩) = ⊙(𝐠[:,i], 𝐩[:,j])
E_mat(i, j, γ_kM1, 𝐠, 𝐱) = ⊙(𝐠[:,i] - 𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
a_vec(i, j, 𝐟) = 𝐟[:, j] - 𝐟[:, i]


function feasible_sol_generator_Lyapunov_Ada_PEP(μ, L, R, η_2;
c_f = 1, c_x = 1, c_g = 1, c_p = 1,
ι_f = 1, ι_x = 1, ι_g = 1, ι_p = 1
)

	k = 1

	d = 4

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

	if (c_f_input, c_x_input, c_g_input, c_p_input) ==  (0, 0, 0, 1) && (ι_f_input, ι_x_input, ι_g_input, ι_p_input) == (0, 0, 1, 0)
		@info "creating initial condition for pseudograd/grad ratio"
		x_0 = randn(d)
		g_0 = Q_f*x_0
		p_0 = g_0
		γ_0 = (p_0'*p_0)/(p_0'*Q_f*p_0)
		Q_f_new = Q_f - γ_0*Q_f^2
		R_tilde_new_sqd = x_0'*Q_f_new^2*x_0
		R_tilde_new = sqrt(R_tilde_new_sqd)
		x_0 = x_0*(R/R_tilde_new)
		f_0 = 0.5*x_0'*Q_f*x_0 # f_{k-1}
	end

	x_0_tilde = randn(d) # x_{k-1}
	x_0_tilde = x_0_tilde/norm(x_0_tilde,2) # x_{k-1} normalized
	Mat_effective = I
	R_tilde_sqd = x_0_tilde'*Mat_effective*x_0_tilde
	R_tilde = sqrt(R_tilde_sqd)

	x_0 = x_0_tilde*(R/R_tilde) # final x_{k-1} normalized
	f_0 = 0.5*x_0'*Q_f*x_0 # f_{k-1}

	## generate the elements of γ, x, g, β, p

	γ = OffsetVector(zeros(2), k-1:k)
	β = OffsetVector(zeros(2), k-1:k)

	# Declare H
	H = zeros(d, 4)

	# Declare Ft
	Ft = zeros(1, 2)

	# array of all x's
	x_array = OffsetArray(zeros(d,2), 1:d, k-1:k)
	x_array[:, k-1] = x_0

	# array of all g's
	g_array = OffsetArray(zeros(d,2), 1:d, k-1:k)
	g_array[:, k-1] = Q_f*x_array[:, k-1]

	# array of all f's
	f_array = OffsetVector(zeros(2), k-1:k)
	f_array[k-1] = f_0

	# array of all pseudo-gradients p's
	p_array = OffsetArray(zeros(d,2), 1:d, k-1:k)
	p_array[:, k-1] = Q_f*x_array[:, k-1] # will satisfy our constraints

	# Putting the entries of γ, x_array, g_array, and f_array one by one

	for i in k-1:k-1
		# generate γ[i]
		γ[i] = (g_array[:,i]'*p_array[:,i])/(p_array[:,i]'*Q_f*p_array[:,i])
		# γ[i] = (x_array[:, i]'*Q_f^2*x_array[:, i])/(x_array[:, i]'*Q_f^3*x_array[:, i])
		# generate x[i+1]
		x_array[:, i+1] = x_array[:, i] - γ[i]*p_array[:,i]
		# generate g[i+1]
		g_array[:, i+1] = Q_f*x_array[:,i+1]# g_array[:, i] - γ[i]*Q_f*p_array[:,i]
		# generate β[i]
		β[i] = ((g_array[:,i+1]'*g_array[:,i+1])- η_2*(g_array[:,i+1]'*g_array[:,i]))/(g_array[:,i]'*g_array[:,i])
		# generate p[i+1]
		p_array[:,i+1] = g_array[:,i+1] + β[i]*p_array[:, i]
		# generate f[i+1]
		f_array[i+1] = 0.5*x_array[:, i+1]'*Q_f*x_array[:, i+1]
	end


	for i in k-1:k-1
		if abs(p_array[:,i]'*g_array[:,i+1]) >= 10^-6
			@show abs(p_array[:,i]'*g_array[:,i+1])
			@error "conjugacy condition is not satisfied"
			return
		else
			@info "conjugacy condition is satisfied"
		end
	end



	# @show x_0'*Mat_effective*x_0

	# Filling the entries of H and Ft one by one now

	# scaling first

	H[:, 1] = x_array[:,k-1]

	H[:, 2] = g_array[:, k-1]

	H[:, 3] = g_array[:, k]

	H[:, 4] = p_array[:, k-1]

	Ft[1, 1] = f_array[k-1]

	Ft[1, 2] = f_array[k]

	# Generate G

	G = H'*H

	# Create β_kM1, χ_k, and γ_k

	β_kM1 = β[k-1]

	γ_kM1 = γ[k-1]

	# scale the G, Ft matrices with respect to the scaling factor

	G = G

	Ft = Ft

	# Generate L_cholesky

	L_cholesky =  compute_pivoted_cholesky_L_mat(G)

	if norm(G - L_cholesky*L_cholesky', Inf) > 1e-6
		@info "checking the norm bound for feasible {G, L_cholesky}"
		@warn "||G - L_cholesky*L_cholesky^T|| = $(norm(G - L_cholesky*L_cholesky', Inf))"
	end

	return G, Ft, L_cholesky, γ_kM1, β_kM1

end


# μ = 0.1
# L = 1
# R = 1
# η_2 = 1
# G_feas, Ft_feas, L_cholesky_feas, γ_kM1_feas, β_kM1_feas = feasible_sol_generator_Lyapunov_Ada_PEP(μ, L, R, η_2; c_f = 1, c_x = 1, c_g = 1, c_p = 1,
# ι_f = 1, ι_x = 1, ι_g = 1, ι_p = 1)




# function to set global subsolver as Gurobi
# ------------------------------------------
GRB_ENV  = Gurobi.Env()



## Global solver for AdaPEP

GlobalOptSolver = JuMP.optimizer_with_attributes(
                                        () -> Gurobi.Optimizer(GRB_ENV),
                                        # MOI.Silent() => true,
                                        "NonConvex" => 2, # means we are going to use Gurobi's spatial branch-and-bound algorithm
																				"MIPFocus" => 3, # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.
																				"MIPGap" => 1e-2, # 99% optimal solution, because Gurobi will provide a result associated with a global lower bound within this tolerance, by polishing the result, we can find the exact optimal solution by solving a convex SDP
																				"OBBT" => -1, # Value 0 disables Optimality-Based Bound Tightening (OBBT). Levels 1-3 describe the amount of work allowed for OBBT ranging from moderate to aggressive. The default -1 value is an automatic setting which chooses a rather moderate setting.
																				"Presolve" => 2 # aggressive application of presolve (2) takes more time, but can sometimes lead to a significantly tighter model.
                                        )
## Local solver for AdaPEP

LocalOptSolver = JuMP.optimizer_with_attributes(
									KNITRO.Optimizer,
									# MOI.Silent() => true,
									"convex" => 0,
									"strat_warm_start" => 1,
									# the last settings below are for larger N
									# you can comment them out if preferred but not recommended
									"honorbnds" => 1,
									# "bar_feasmodetol" => 1e-3,
									# "feastol" => ϵ_tol_feas,
									# "infeastol" => 1e-12,
									# "opttol" => 1e-4
									"algorithm" => 1
                                    )


function Ratio_of_Pseudograd_and_Grad_Model(
	# different parameters to be used
  # -------------------------------
  μ, L, R,
  # warm-start points
  # -----------------
  idx_set_λ_ws_effective, G_ws, Ft_ws, L_cholesky_ws, γ_kM1_ws,  β_kM1_ws;
	# different options
	# -----------------
	# [🐘 ] solver type
	solver = LocalOptSolver,
	# the solvers are:
	# GlobalOptSolver to find the globally optimal solution (uses gurobi)
	# LocalOptSolver to find the locally optimal solution (uses KNITRO)
	# [🐯 ] cost coefficients
	c_f = 0, c_x = 0, c_g = 0, c_p = 1,
	# [💁 ] initial condition coefficients
	ι_f = 0, ι_x = 0, ι_g = 0, ι_p = 1,
	show_output = :on, # options are :on and :off
	lower_bound_opt_val = 0,
	upper_bound_opt_val = Inf,
	impose_pattern_interpolation = :off,
	impose_pattern_noninterpolation = :off,
	bound_impose = :off, # # options are :generic, and :off
	bound_M = Inf,
	lower_bound_var = [], # lower bound vector on the decision variables
	upper_bound_var = [], # upper bound vector on the decision variables
	PSDness_modeling = :exact, # options are :exact and :through_ϵ and :lazy_constraint_callback
  ϵ_tol_feas = 1e-4, # feasiblity tolerance for minimum eigenvalue of G
  maxCutCount=1e6, # number of lazy cuts when we use lazy constraints for modeling G = L_cholesky*L_cholesky
	PRP_Plus = :off, # options are :positive_case, :negative_case and :off
	c_0_input_initial = 1 # it is (1+ (L/μ)^2) for PRP and can be taken to be any number greater than 1 for FR
	)

	## Number of points etc

	k = 1

	⋆ = -1 # Keep in mind that this ⋆ is the unicode charectar, which can be input as \star[TAB]

	I_N_star = [⋆; k-1; k]

	dim_G = 4

	dim_Ft = 2

	dim_𝐱 = 4

	dim_𝐩 = 4

	## Create model
	# -------------

	Ada_PEP_model =  Model(solver)

	## Define all the variables

	# add the variables
	# -----------------

	@info "[🎉 ] defining the variables"

	# construct G ⪰ 0
	@variable(Ada_PEP_model, G[1:dim_G, 1:dim_G], Symmetric)

	# define the cholesky matrix of Z: L_cholesky
	# -------------------------------------------
	@variable(Ada_PEP_model, L_cholesky[1:dim_G, 1:dim_G])

	# construct Ft (this is just transpose of F)
	@variable(Ada_PEP_model, Ft[1:dim_Ft])

	# Construct γ_k
	@variable(Ada_PEP_model, γ_kM1 >= 0)

	# Construct β
	@variable(Ada_PEP_model, β_kM1)

	# construct the hypograph variables
	@variable(Ada_PEP_model,  t >= lower_bound_opt_val)

	if impose_pattern_interpolation == :off
		# define λ over the full index set
		idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)
	elseif impose_pattern_interpolation == :on
		# define λ over a reduced index set, idx_set_λ_ws_effective, which is the effective index set of λ_ws
		@info "[🎣 ] imposing pattern on effective index set of λ"
		#
		# Full thing:
		# idx_set_λ = [i_j_idx(⋆, k-1); i_j_idx(⋆, k); i_j_idx(k-1, k);  i_j_idx(k, k-1); i_j_idx( k-1, ⋆); i_j_idx( k, ⋆)]
		# alternative
		idx_set_λ = [i_j_idx(k-1, k);  i_j_idx(k, k-1)]
		# idx_set_λ = [i_j_idx(k, k-1)]
	else
		@error "impose_pattern should be either :on or :off"
	end

	# Define Θ[i,j] entries
	# Define Θ[i,j] matrices such that for i,j ∈ I_N_star, we have Θ[i,j] = ⊙(𝐱[:,i] -𝐱[:,j], 𝐱[:,i] - 𝐱[:,j])

	Θ = Ada_PEP_model[:Θ] = reshape(
			hcat([
			@variable(Ada_PEP_model, [1:dim_𝐱, 1:dim_𝐱], Symmetric, base_name = "Θ[$i_j_λ]")
			for i_j_λ in idx_set_λ]...), dim_𝐱, dim_𝐱, length(idx_set_λ))

	@variable(Ada_PEP_model, B_mat_k_star[1:dim_𝐱, 1:dim_𝐱], Symmetric)

	@variable(Ada_PEP_model, C_tilde_mat_k_star[1:dim_𝐩, 1:dim_𝐩], Symmetric) # this models C_tilde_mat(k, ⋆, β_kM1, 𝐩)


	if bound_impose == :generic # options are :generic
		@info "[💣 ] imposing bound on the the entries of G, F based on Lyapunov function choice"

		# bound on Ft
		# -----------

		for i in 1:dim_Ft
			set_lower_bound(Ft[i], 0.0)
			set_upper_bound(Ft[i], bound_M)
		end

		# bound on G
		# ----------

		for i in 1:dim_G
			set_lower_bound(G[i,i], 0.0)
			set_upper_bound(G[i,i], bound_M)
		end

		# set bound for L_cholesky
		# ------------------------

		# need only upper bound for the diagonal compoments, as the lower bound is zero from the model
		for i in 1:dim_G
			set_lower_bound(L_cholesky[i,i], 0)
			set_upper_bound(L_cholesky[i,i], sqrt(bound_M))
		end
		# need to bound only components, L_cholesky[i,j] with i > j, as for i < j, we have zero, due to the lower triangular structure
		for i in 1:dim_G
			for j in 1:dim_G
				if i > j
					set_lower_bound(L_cholesky[i,j], -sqrt(bound_M))
					set_upper_bound(L_cholesky[i,j], sqrt(bound_M))
				end
			end
		end

		M_γ_kM1 = max(10*abs(γ_kM1_ws), 2)
		M_β_kM1 = max(10*abs(β_kM1_ws), 2)

		set_upper_bound(γ_kM1, M_γ_kM1)

	  set_lower_bound(β_kM1, -M_β_kM1)
		set_upper_bound(β_kM1, M_β_kM1)

	elseif bound_impose == :off

		@info "no bound imposed"

	else
		@error "bound_impose options are :generic, and :off"
		return
	end

	## create the data generator

	@info "[🎍 ] adding the data generator function to create 𝐱, 𝐠, 𝐟"

	𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function_Lyapunov(γ_kM1, β_kM1)

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
			@constraint(Ada_PEP_model, vectorize(
			Θ[:,:,ℓ] - ⊙(𝐱[:,i_j_λ.i]-𝐱[:,i_j_λ.j], 𝐱[:,i_j_λ.i]-𝐱[:,i_j_λ.j]),
			SymmetricMatrixShape(dim_𝐱)) .== 0)
	end

	# Implement symmetry in Θ, i.e., Θ[i,j]=Θ[j,i]

	conΘSymmetry = map(1:length(idx_set_λ)) do ℓ
			i_j_λ = idx_set_λ[ℓ]
			if i_j_idx(i_j_λ.j, i_j_λ.i) in idx_set_λ
				@constraint(Ada_PEP_model, vectorize(
				Θ[:,:,index_finder_Θ(i_j_λ.i, i_j_λ.j, idx_set_λ)]
			  -
				Θ[:,:,index_finder_Θ(i_j_λ.j, i_j_λ.i, idx_set_λ)],
				SymmetricMatrixShape(dim_𝐱)
				) .== 0 )
		 end
	end # end the do map

	## Add the constraints related to G

	# time to implement Z = L*L^T constraint
	# --------------------------------------

	if PSDness_modeling == :exact

			# direct modeling through definition and vectorization
			# ---------------------------------------------------
			@constraint(Ada_PEP_model, vectorize(G - (L_cholesky * L_cholesky'), SymmetricMatrixShape(dim_G)) .== 0)

	elseif PSDness_modeling == :through_ϵ

			# definition modeling through vectorization and ϵ_tol_feas

			# part 1: models Z-L_cholesky*L_cholesky <= ϵ_tol_feas*ones(dim_G,dim_G)
			@constraint(Ada_PEP_model, vectorize(G - (L_cholesky * L_cholesky') - ϵ_tol_feas*ones(dim_G,dim_G), SymmetricMatrixShape(dim_G)) .<= 0)

			# part 2: models Z-L_cholesky*L_cholesky >= -ϵ_tol_feas*ones(dim_G,dim_G)

			@constraint(Ada_PEP_model, vectorize(G - (L_cholesky * L_cholesky') + ϵ_tol_feas*ones(dim_G,dim_G), SymmetricMatrixShape(dim_G)) .>= 0)

	elseif PSDness_modeling == :lazy_constraint_callback

				# set_optimizer_attribute(Ada_PEP_model, "FuncPieces", -2) # FuncPieces = -2: Bounds the relative error of the approximation; the error bound is provided in the FuncPieceError attribute. See https://www.gurobi.com/documentation/9.1/refman/funcpieces.html#attr:FuncPieces

				# set_optimizer_attribute(Ada_PEP_model, "FuncPieceError", 0.1) # relative error

				set_optimizer_attribute(Ada_PEP_model, "MIPFocus", 1) # focus on finding good quality feasible solution

				# add initial cuts
				num_cutting_planes_init = (5*dim_G^2)
				cutting_plane_array = [eigvecs(G_ws) rand(Uniform(-1,1), dim_G, num_cutting_planes_init)] # randn(dim_G,num_cutting_planes_init)
				num_cuts_array_rows, num_cuts = size(cutting_plane_array)
				for i in 1:num_cuts
						d_cut = cutting_plane_array[:,i]
						d_cut = d_cut/norm(d_cut,2) # normalize the cutting plane vector
						@constraint(Ada_PEP_model, tr(G*(d_cut*d_cut')) >= 0)
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
									MOI.submit(Ada_PEP_model, MOI.LazyConstraint(cb_data), con3)
							end
							cutCount+=1
					end
				end

				# submit the lazy constraint
				# --------------------------
				MOI.set(Ada_PEP_model, MOI.LazyConstraintCallback(), add_lazy_callback)

	else
			@error "something is not right in PSDness_modeling"
			return
	end

	# add initial condition
	# ---------------------

	@constraint(Ada_PEP_model, conInit,
		# ι_f*(Ft'*a_vec(⋆,k-1, 𝐟)) +
		# ι_x*(tr(G*B_mat(k-1, ⋆, γ_kM1, 𝐱))) +
		# ι_g*(tr(G*C_mat(k, ⋆, 𝐠)))
		# + ι_p*(tr(G*C_tilde_mat(k-1, ⋆, β_kM1, 𝐩)))
		(tr(G*C_mat(k, ⋆, 𝐠)))
		== R^2
	)

	## All the conditions related to C, C_tilde, D, D_tilde
	# -----------------------------------------------------
	

	@constraint(Ada_PEP_model, tr(G*C_tilde_mat(k-1, ⋆, β_kM1, 𝐩)) == c_0_input_initial*tr(G*C_mat(k-1, ⋆, 𝐠)))

	# @constraint(Ada_PEP_model, tr(G*C_mat(k-1, ⋆, 𝐠)) <= tr(G*C_tilde_mat(k-1, ⋆, β_kM1, 𝐩)) )

	# if η_2 == 1
	# 	@constraint(Ada_PEP_model, tr(G*C_tilde_mat_k_star) >= 0.9*(L+μ)^2/(4*μ*L))
	# 	@constraint(Ada_PEP_model, tr(G*C_tilde_mat_k_star) <= 1.1*(L+μ)^2/(4*μ*L))
	# elseif η_2 == 0
	# 	@constraint(Ada_PEP_model, tr(G*C_tilde_mat_k_star) >= 0.9*(((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)) )
	# 	@constraint(Ada_PEP_model, tr(G*C_tilde_mat_k_star) <= 1.1*(((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)) )
	# end



	if impose_pattern_noninterpolation == :on

		# for i in k-1:k
		#
		# 	@constraint(Ada_PEP_model, (L_cholesky'*𝐠[:,i])[3] == 0)
		#
		# 	@constraint(Ada_PEP_model, (L_cholesky'*𝐠[:,i])[4] == 0)
		#
		# 	@constraint(Ada_PEP_model, (L_cholesky'*𝐱[:,i])[3] == 0)
		#
		# 	@constraint(Ada_PEP_model, (L_cholesky'*𝐱[:,i])[4] == 0)
		#
		#
	  # end

	  # we want to find the lower bound greater than this for finding the bad function
		if η_2 == 1
			@constraint(Ada_PEP_model, tr(G*C_tilde_mat_k_star) >= 0.9*(L+μ)^2/(4*μ*L))
			@constraint(Ada_PEP_model, tr(G*C_tilde_mat_k_star) <= 1.1*(L+μ)^2/(4*μ*L))
		elseif η_2 == 0
			@constraint(Ada_PEP_model, tr(G*C_tilde_mat_k_star) >= 0.9*(((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)) )
			@constraint(Ada_PEP_model, tr(G*C_tilde_mat_k_star) <= 1.1*(((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)) )
		end

  end



	# Constraint: tr G D_tilde{k-1, k-1} = tr G C_{k-1,*}  ⇔  ⟨ g_{k-1} ∣ p_{k-1} ⟩ = || g_{k-1} ||^2

	@constraint(Ada_PEP_model, tr(G*D_tilde_mat(k-1, k-1, β_kM1, 𝐠, 𝐩)) == tr(G*C_mat(k-1, ⋆, 𝐠)))

	# Constraint: tr G D_tilde{k, k-1} = 0 ⇔ ⟨ g_{k} ∣ p_{k-1} ⟩ = 0

	@constraint(Ada_PEP_model, tr(G*D_tilde_mat(k, k-1, β_kM1, 𝐠, 𝐩)) == 0)

	# Constraint: tr G A_{k-1,k} = 0 ⇔ ⟨ g_{k} ∣ x_{k-1} - x_{k} ⟩ = 0

	@constraint(Ada_PEP_model, tr(G*A_mat(k-1, k, γ_kM1, 𝐠, 𝐱))==0)

	# Constraint: CG update parameter
	# β_kM1 * tr G C_{k-1, *} = tr G (C_{k,*} - η_2*D_{k,k-1})

	@constraint(Ada_PEP_model, β_kM1 * tr(G*C_mat(k-1, ⋆, 𝐠)) == tr(G * (C_mat(k, ⋆, 𝐠) - η_2*D_mat(k, k-1, 𝐠))) )

	## Time for the interpolation constraint

	q = μ/L

	conInterpol = map(1:length(idx_set_λ)) do ℓ
			i_j_λ = idx_set_λ[ℓ]
			@info "coninterpol ℓ = $(i_j_λ)"
			if i_j_λ.i == k || i_j_λ.j == k
				@constraint(Ada_PEP_model,
				Ft'*a_vec(i_j_λ.i,i_j_λ.j,𝐟)
				+ tr(G * (
				A_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐠, 𝐱)
				+ ( 1/(2*(1-q)) )*(
				(1/(L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠)
				+	(μ*Θ[:,:,index_finder_Θ(i_j_λ.i, i_j_λ.j, idx_set_λ)])
				- (2*q*E_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐠, 𝐱) )
				)
				)
				)
				<= 0)
			elseif i_j_λ.i != k && i_j_λ.j != k
				@constraint(Ada_PEP_model,
				Ft'*a_vec(i_j_λ.i,i_j_λ.j,𝐟)
				+ tr(G * (
				A_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐠, 𝐱)
				+ ( 1/(2*(1-q)) )*(
				(1/(L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠)
				+	(μ*B_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐱))
				- (2*q*E_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐠, 𝐱) )
				)
				)
				)
				<= 0)
		   end
		end # end the do map

	## Time for the hypograph constraint

	# Now model
	# B_mat_k_star = ⊙(𝐱[:,k], 𝐱[:,k])
	# recall that  𝐱[:,-1] = 𝐱[:,⋆] = 0

	@constraint(Ada_PEP_model, conB_mat_k_star, vectorize(
	B_mat_k_star - ⊙(𝐱[:,k], 𝐱[:,k]),
	SymmetricMatrixShape(dim_𝐱)
	) .== 0)

	# Now model
	# C_tilde_mat_k_star = ⊙(p[:, k], p[:, k])
	# recall that p[:, -1] = p[:, ⋆] = 0

	@constraint(Ada_PEP_model, conC_tilde_mat_k_star, vectorize(
	C_tilde_mat_k_star - ⊙(𝐩[:, k], 𝐩[:, k]),
	SymmetricMatrixShape(dim_𝐩)
	) .== 0)

	## Time for the hypograph constraint

	@constraint(Ada_PEP_model, hypographCon, t <= tr(G*C_tilde_mat_k_star)
		#c_f*(Ft'*a_vec(⋆,k, 𝐟))
		# + c_x*(tr(G*B_mat_k_star))
		# +c_g*(tr(G*C_mat(k, ⋆, 𝐠)))
		# + c_p*(tr(G*C_tilde_mat_k_star))
	)

	## List of valid constraints for G

	# diagonal components of G are non-negative

	@constraint(Ada_PEP_model, conNonNegDiagG[i=1:dim_G], G[i,i] >= 0)


	# the off-diagonal components satisfy:
	# (∀i,j ∈ dim_G: i != j) -(0.5*(G[i,i] + G[j,j])) <= G[i,j] <=  (0.5*(G[i,i] + G[j,j]))

	for i in 1:dim_G
		for j in 1:dim_G
			if i != j
				@constraint(Ada_PEP_model, G[i,j] <= (0.5*(G[i,i] + G[j,j])) )
				@constraint(Ada_PEP_model, -(0.5*(G[i,i] + G[j,j])) <= G[i,j] )
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

	@constraint(Ada_PEP_model, conCholeskyDiag[i=1:dim_G], L_cholesky[i,i] >= 0)



	## add the mathematical bound for the entries of G

	@constraint(Ada_PEP_model, G[1,1] <= (2/μ)*Ft[1])
	@constraint(Ada_PEP_model, G[1,1] >= (2/L)*Ft[1])

	@constraint(Ada_PEP_model, G[2, 2] <= 2*L*Ft[1])
	@constraint(Ada_PEP_model, G[2, 2] >= 2*μ*Ft[1])

	@constraint(Ada_PEP_model, G[3, 3] <= 2*L*Ft[2])
	@constraint(Ada_PEP_model, G[3, 3] >= 2*μ*Ft[2])


	## Add the objective

	@info "[🎇 ] adding objective"

	@objective(Ada_PEP_model, Max, t)

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

	# warm_start γ_kM1

	set_start_value(γ_kM1, γ_kM1_ws)

	# warm-start β_kM1

	set_start_value(β_kM1, β_kM1_ws)

	# warm-start L_cholesky

	for i in 1:dim_G
		for j in 1:dim_G
			set_start_value(L_cholesky[i,j], L_cholesky_ws[i,j])
		end
	end


	# warm start for Θ
	# ----------------

	# construct 𝐱_ws, 𝐠_ws, 𝐟_ws corresponding to warm-start stepsize

	𝐱_ws, 𝐠_ws, 𝐟_ws, 𝐩_ws = data_generator_function_Lyapunov(γ_kM1_ws, β_kM1_ws)


	# warm-start B_mat_k_star

	B_mat_k_star_ws = ⊙(𝐱_ws[:,k], 𝐱_ws[:,k])

	set_start_value.(B_mat_k_star, B_mat_k_star_ws)

	# warm-start C_tilde_mat_k_star

	C_tilde_mat_k_star_ws = ⊙(𝐩_ws[:, k], 𝐩_ws[:, k])

	set_start_value.(C_tilde_mat_k_star, C_tilde_mat_k_star_ws)

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


	t_ws = c_f*(Ft_ws[end]) + c_x*(tr(G_ws*B_mat_k_star_ws)) + c_g*(tr(G_ws*C_mat(k, ⋆, 𝐠_ws))) + c_p*(tr(G_ws*C_tilde_mat_k_star_ws))

	set_start_value(t, t_ws)

	# Check if all the variables have been warm-started

	if any(isnothing, start_value.(all_variables(Ada_PEP_model))) == true
		@error "all the variables are not warm-started"
	end

	## time to optimize
	# ----------------

	@info "[🙌 	🙏 ] model building done, starting the optimization process"

	if show_output == :off
		set_silent(Ada_PEP_model)
	end

	if PRP_Plus == :positive_case && η_2 == 1
		@info "[🌻 ] Activating PRP_plus positive case"
		@constraint(Ada_PEP_model, tr(G * (C_mat(k, ⋆, 𝐠) - η_2*D_mat(k, k-1, 𝐠))) >= 0)
		# already implemented β_kM1 formula
	elseif PRP_Plus == :negative_case && η_2 == 1
		@info "[🌻 ] Activating PRP_plus negative case"
		@constraint(Ada_PEP_model, tr(G * (C_mat(k, ⋆, 𝐠) - η_2*D_mat(k, k-1, 𝐠))) <= 0)
		fix(β_kM1, 0.0; force = true)
	elseif PRP_Plus == :off
		@info "[🎴 ] Analyzing PRP/FR method"
	else
		@error "[💀 ] PRP_Plus options are :positive_case, :negative_case and :off"
	end


	optimize!(Ada_PEP_model)

	## Time to store the solution

	if (termination_status(Ada_PEP_model) == LOCALLY_SOLVED || termination_status(Ada_PEP_model) == SLOW_PROGRESS || termination_status(Ada_PEP_model) == OPTIMAL)

		if termination_status(Ada_PEP_model) == SLOW_PROGRESS
			@warn "[💀 ] termination status of Ada_PEP_model is SLOW_PROGRESS"
		end

		# store the solutions and return
		# ------------------------------

		@info "[😻 ] optimal solution found done, store the solution"

		# store G_opt

		G_opt = value.(G)

		# store F_opt

		Ft_opt = value.(Ft)

		# store L_cholesky_opt

		L_cholesky_opt =  compute_pivoted_cholesky_L_mat(G_opt)

		# L_cholesky_opt = value.(L_cholesky)

		if norm(G_opt - L_cholesky_opt*L_cholesky_opt', Inf) > 10^-4
			@warn "||G - L_cholesky*L_cholesky^T|| = $(norm(G_opt -  L_cholesky_opt*L_cholesky_opt', Inf))"
		end

		# store γ_k_opt

		γ_kM1_opt = value.(γ_kM1)

		# store β_kM1_opt

	    β_kM1_opt = value.(β_kM1)

		@info "[💹 ] warm-start objective value = $t_ws, and objective value of found solution = $(objective_value(Ada_PEP_model))"

		t_opt = value.(t)

		𝐱_opt, 𝐠_opt, 𝐟_opt, 𝐩_opt = data_generator_function_Lyapunov(γ_kM1_opt, β_kM1_opt)

		@info "IMPORTANT INFO"
		@info "------------------------------------------"
		# @info "f_k - f_star = $(Ft_opt[end])"
		# @info "||x_k - x_star ||^2 = $(tr(G_opt*⊙(𝐱_opt[:,k],𝐱_opt[:,k])))"
		@info "||g_{k-1}||^2 =  $(tr(G_opt*⊙(𝐠_opt[:,k-1],𝐠_opt[:,k-1])))"
		@info "||p_{k-1}||^2 = $(tr(G_opt*⊙(𝐩_opt[:,k-1],𝐩_opt[:,k-1])))"
		# @show (tr(G_opt*C_tilde_mat(k-1, ⋆, β_kM1_opt, 𝐩_opt)) -  (c_0_input_initial*tr(G_opt*C_mat(k-1, ⋆, 𝐠_opt))))
		# @info "||g_k||^2 =  $(tr(G_opt*⊙(𝐠_opt[:,k],𝐠_opt[:,k])))"
		# @info "||p_k||^2 = $(tr(G_opt*⊙(𝐩_opt[:,k],𝐩_opt[:,k])))"
		if η_2 == 1
			@info "NCG-PEP Theory {||p_k||^2/||g_k||^2} = $((L+μ)^2/(4*μ*L))"
		elseif η_2 == 0
			@info "NCG-PEP Theory {||p_k||^2/||g_k||^2} =  $((((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)))"
		end
		@info "NCG-PEP Numerical {||p_k||^2/||g_k||^2}= $(t_opt)"
		# @info " Polyak{||p_k||^2/||g_k||^2}= $(1+(L/μ)^2)"
		# @info " AdaPEP Analytical PRP {||p_k||^2/||g_k||^2}= $(1+((L-μ)^2/(4*L*μ)))"

		@info "------------------------------------------"


	else

		@error "[🙀 ] could not find an optimal solution"

	end

	lower_bound_var = []
	upper_bound_var = []


	solve_time_AdaPEP_model = solve_time(Ada_PEP_model)

	if solver == LocalOptSolver && (termination_status(Ada_PEP_model) == LOCALLY_SOLVED || termination_status(Ada_PEP_model) == SLOW_PROGRESS)
		@info "[💀 ] The dual multipliers for interpolation inequalities are: $(dual.(conInterpol))"
		# @show dual.(conInterpol)
	end

	## time to return the solution

	return G_opt, Ft_opt, L_cholesky_opt, γ_kM1_opt, β_kM1_opt, t_opt, lower_bound_var, upper_bound_var, solve_time_AdaPEP_model

end


# Custom structure for saving the worst-case function
struct worst_case_function
	x_array::OffsetMatrix{Float64, Matrix{Float64}}
	g_array::OffsetMatrix{Float64, Matrix{Float64}}
	f_array::OffsetVector{Float64, Vector{Float64}}
end

function generate_worst_case_function_ratio(G, Ft, L_cholesky, γ_kM1, β_kM1)

	# create the function from the bold points
	𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function_Lyapunov(γ_kM1, β_kM1)
  # create H
	L_cholesky = compute_pivoted_cholesky_L_mat(G)
	H = convert(Array{Float64,2}, LinearAlgebra.transpose(L_cholesky))
	d, _ = size(H)

	x_array = OffsetArray(zeros(d,3), 1:d, -1:1)

	g_array = OffsetArray(zeros(d,3), 1:d, -1:1)

	f_array = OffsetVector(zeros(3), -1:1)

	for i in 0:1
			x_array[:, i] =  H*𝐱[:, i]
			g_array[:, i] = H*𝐠[:, i]
			f_array[i] = Ft[i+1]
			@info "p[:,$(i)] = $(H*𝐩[:, i])"
	end

	wf = worst_case_function(x_array, g_array, f_array)

	return wf

end


# Time to solve the SDP with the stepsize as an input
# ===================================================

function SDP_with_fixed_stepsize_ratio(
	        # Inputs
					# ======
					μ, L, R, γ_kM1, β_kM1;
					# [🐯 ] cost coefficients
					# Options
					# =======
					η_2_input = 1, # controls PRP or FR, if η = 1 => PRP, and if η = 0 => FR
					c_0_input_initial = 1, # decides ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
					show_output = :on, # options are :on and :off
					ϵ_tolerance = 1e-6,
					reduce_index_set_for_λ = :off
          )

	## Some hard-coded parameter

	## Number of points etc

	k = 1

	⋆ = -1 # Keep in mind that this ⋆ is the unicode charectar, which can be input as \star[TAB]

	I_N_star = [⋆; k-1; k]

	dim_G = 4

	dim_Ft = 2

	dim_𝐱 = 4

	dim_𝐩 = 4

	# polyak_contraction_factor = (1 - (μ/L)* (1/(1 + (L/μ)^2)))

	## data generator
	# --------------

	## create the data generator

	@info "[🎍 ] adding the data generator function to create 𝐱, 𝐠, 𝐟"

	𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function_Lyapunov(γ_kM1, β_kM1)

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

	# construct the hypograph variables
	@variable(model_primal_PEP_with_known_stepsizes,  t)

	# Number of interpolation constraints

	# We already have a feasible point from the locally optimal solution
	# @constraint(model_primal_PEP_with_known_stepsizes, Ft[end] >= 0.99*Ft_ws_els[end])

	# Number of interpolation constraints

	idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star)

	## Time to add the constraints one by one

	@info "[🎢 ] adding the constraints"

	## Initial condition

	@constraint(model_primal_PEP_with_known_stepsizes, conInit,
	    (tr(G*C_mat(k, ⋆, 𝐠))) == R^2
	)

	## Other constraints

	@constraint(model_primal_PEP_with_known_stepsizes, tr(G*D_tilde_mat(k-1, k-1, β_kM1, 𝐠, 𝐩)) == tr(G*C_mat(k-1, ⋆, 𝐠)))

	# @constraint(model_primal_PEP_with_known_stepsizes, tr(G*C_mat(k-1, ⋆, 𝐠)) <= tr(G*C_tilde_mat(k-1, ⋆, β_kM1, 𝐩)))

	@constraint(model_primal_PEP_with_known_stepsizes, tr(G*C_tilde_mat(k-1, ⋆, β_kM1, 𝐩)) == c_0_input_initial*tr(G*C_mat(k-1, ⋆, 𝐠)))

	@constraint(model_primal_PEP_with_known_stepsizes, tr(G*D_tilde_mat(k, k-1, β_kM1, 𝐠, 𝐩)) == 0)

	@constraint(model_primal_PEP_with_known_stepsizes, tr(G*A_mat(k-1, k, γ_kM1, 𝐠, 𝐱))==0)

	## β update parameter

	@constraint(model_primal_PEP_with_known_stepsizes, β_kM1 * tr(G*C_mat(k-1, ⋆, 𝐠)) == tr(G * (C_mat(k, ⋆, 𝐠) - η_2_input*D_mat(k, k-1, 𝐠))) )

	# Interpolation constraint

	## Time for the interpolation constraint

	q = μ/L

	conInterpol = map(1:length(idx_set_λ)) do ℓ
		i_j_λ = idx_set_λ[ℓ]
		@constraint(model_primal_PEP_with_known_stepsizes,
		Ft'*a_vec(i_j_λ.i,i_j_λ.j,𝐟)
		+ tr(G * (
		A_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐠, 𝐱)
		+ ( 1/(2*(1-q)) )*(
		(1/(L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠)
		+	(μ*B_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐱))
		- (2*q*E_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐠, 𝐱) )
		)
		)
		)
		<= 0)
	end # end the do map

	## Time for the hypograph constraint

	@constraint(model_primal_PEP_with_known_stepsizes, hypographCon, t <= tr(G*C_tilde_mat(k, ⋆, β_kM1, 𝐩))
	)

	@constraint(model_primal_PEP_with_known_stepsizes, t >= 0)


	# if η_2_input == 1
	# 	@constraint(Ada_PEP_model, t >= 0.8*(L+μ)^2/(4*μ*L))
	# 	@constraint(Ada_PEP_model, t <= 1.2*(L+μ)^2/(4*μ*L))
	# elseif η_2_input == 0
	# 	@constraint(Ada_PEP_model, t >= 0.8*(((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)) )
	# 	@constraint(Ada_PEP_model, t <= 1.2*(((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)) )
	# end


	## Add the objective and the associated hypograph constraint

	# @info "[🎇 ] adding objective"
	# =============================

	@objective(model_primal_PEP_with_known_stepsizes, Max,  t)

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

	t_opt = objective_value(model_primal_PEP_with_known_stepsizes)

	@info "NCG-PEP Numerical {||p_k||^2/||g_k||^2}= $(t_opt)"

	if η_2_input == 1
		@info "NCG-PEP Theory {||p_k||^2/||g_k||^2} = $((L+μ)^2/(4*μ*L))"
	elseif η_2_input == 0
		@info "NCG-PEP Theory {||p_k||^2/||g_k||^2} =  $((((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)))"
	end


	return p_star, G_star, Ft_star, L_cholesky_star

end


function Dual_SDP_with_fixed_stepsize_ratio(
	# Inputs
	# ======
	μ, L, R, γ_kM1, β_kM1;
	# sparsifier coefficients
	# -----------------------
	c_λ = 1, c_τ = 1, c_Z = 1, c_ρ = 1,
	# Options
	# =======
	objective_type = :default, # options are :default, :find_sparse_sol
	η_2_input = 1, # controls PRP or FR, if η = 1 => PRP, and if η = 0 => FR
	c_0_input_initial = 1, # decides ||g_0||^2 <= ||p_0||^2 <= c_0 ||g_0||^2
	show_output = :on, # options are :on and :off
	ϵ_tolerance = 1e-6,
	impose_pattern = :off, # other option is :on
	obj_val_upper_bound = Inf
	)

	## Number of points etc

	k = 1;

	⋆ = -1 # Keep in mind that this ⋆ is the unicode charectar, which can be input as \star[TAB]
	
	I_N_star = [⋆; k-1; k];

	dim_G = 4;

	dim_Ft = 2;

	dim_𝐱 = 4;

	dim_𝐩 = 4;

	# polyak_contraction_factor = (1 - (μ/L)* (1/(1 + (L/μ)^2)))

	## data generator
	# --------------

	## create the data generator

	@info "[🎍 ] adding the data generator function to create 𝐱, 𝐠, 𝐟";

	𝐱, 𝐠, 𝐟, 𝐩 = data_generator_function_Lyapunov(γ_kM1, β_kM1);

	## define the model
	# ----------------

	# model_dual_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer));

	model_dual_PEP_with_known_stepsizes = Model(optimizer_with_attributes(Mosek.Optimizer, "INTPNT_CO_TOL_DFEAS" => ϵ_tolerance, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => ϵ_tolerance))

	# define the variables
	# --------------------

	# define the index set of λ
	idx_set_λ = index_set_constructor_for_dual_vars_full(I_N_star);

	# define λ
	@variable(model_dual_PEP_with_known_stepsizes, λ[idx_set_λ] >= 0);

	# define ν

	@variable(model_dual_PEP_with_known_stepsizes, ν >= 0);

	# define ρ

	@variable(model_dual_PEP_with_known_stepsizes, ρ);

	# define Z ⪰ 0
	@variable(model_dual_PEP_with_known_stepsizes, Z[1:dim_G, 1:dim_G], PSD);

	# define τ
	@variable(model_dual_PEP_with_known_stepsizes, τ[1:4]);

	@variable(model_dual_PEP_with_known_stepsizes, epi_τ[1:4]);

	## define the objective

	if objective_type == :default

		@info "[🐒 ] Minimizing the usual performance measure"

		@objective(model_dual_PEP_with_known_stepsizes, Min,  ν)

	elseif objective_type == :find_sparse_sol

		@info "[🐮 ] Finding a sparse dual solution given the objective value upper bound"

		@objective(model_dual_PEP_with_known_stepsizes, Min, c_λ*sum(λ[i_j] for i_j in idx_set_λ) + c_τ*sum(epi_τ) + c_Z*tr(Z) + c_ρ*ρ)

		if c_τ == 1
			@constraint(model_dual_PEP_with_known_stepsizes, con_epi_tau_1[i=1:4], epi_τ[i] >= τ[i])
			@constraint(model_dual_PEP_with_known_stepsizes, con_epi_tau_2[i=1:4], epi_τ[i] >= -τ[i])
		end

		@constraint(model_dual_PEP_with_known_stepsizes, ν <= obj_val_upper_bound)

	end

	## Add linear constraint

	@constraint(model_dual_PEP_with_known_stepsizes, sum(λ[i_j_λ]*a_vec(i_j_λ.i,i_j_λ.j,𝐟) for i_j_λ in idx_set_λ) .== 0)

	## Define the related constraints

	M_τ_1 = D_tilde_mat(k-1, k-1, β_kM1, 𝐠, 𝐩) - C_mat(k-1, ⋆, 𝐠)

	M_τ_2 = D_tilde_mat(k, k-1, β_kM1, 𝐠, 𝐩)

	M_τ_3 = A_mat(k-1, k, γ_kM1, 𝐠, 𝐱)

	M_τ_4 = C_mat(k, ⋆, 𝐠) - η_2_input*D_mat(k, k-1, 𝐠) - β_kM1*C_mat(k-1, ⋆, 𝐠)

	M_ρ = C_tilde_mat(k-1, ⋆, β_kM1, 𝐩) - c_0_input_initial*C_mat(k-1, ⋆, 𝐠)

	M_ν = C_mat(k, ⋆, 𝐠)

	## Add the LMI constraint

	q = μ/L

	@constraint(model_dual_PEP_with_known_stepsizes,
		-C_tilde_mat(k, ⋆, β_kM1, 𝐩)
		+ (τ[1]*M_τ_1)
		+ (τ[2]*M_τ_2)
		+ (τ[3]*M_τ_3)
		+ (τ[4]*M_τ_4)
		+ (ρ*M_ρ)
		+ ν*M_ν 
		+ sum(λ[i_j_λ]*(
			A_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐠, 𝐱)
			+ (1/(2*(1-q)))*(
			(1/(L))*C_mat(i_j_λ.i,i_j_λ.j,𝐠)
			+	(μ*B_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐱))
			- (2*q*E_mat(i_j_λ.i, i_j_λ.j, γ_kM1, 𝐠, 𝐱) ) 
		)
		) for i_j_λ in idx_set_λ)
		.==
		Z
	)

	## Impose pattern constraints

	@variable(model_dual_PEP_with_known_stepsizes, θ )

	if impose_pattern == :on
	  λ_fix_term=(1/(4*L*(L - μ)*μ))*(L^4 + 
      4*(-4 + 3*c_0_input_initial)*L^3*μ + μ^4 + 
      6*μ^2*sqrt((-1 + c_0_input_initial)*L*μ*(-L + μ)^2) + 
      6*L^2*((5 - 4*c_0_input_initial)*μ^2 + 
      sqrt((-1 + c_0_input_initial)*L*(L - μ)^2*μ)) + 
      4*L*((-4 + 3*c_0_input_initial)*μ^3 + (-5 + 2*c_0_input_initial)*μ*
       sqrt((-1 + c_0_input_initial)*L*(L - μ)^2*μ)));
	   
	   fix(λ[i_j_idx(-1, 0)], 0.0; force = true)
	   fix(λ[i_j_idx(0, -1)], 0.0; force = true)
	   fix(λ[i_j_idx(-1, 1)], 0.0; force = true)
	   fix(λ[i_j_idx(1, -1)], 0.0; force = true)
	   fix(τ[3], 0.0; force = true)
	   # fix(τ[1],  γ_kM1*(L+μ); force = true)
	   # fix(τ[1], 0.0; force = true)
	   # @constraint(model_dual_PEP_with_known_stepsizes, ρ == (L^2- L*μ)*γ_kM1 + β_kM1^2)
	   @constraint(model_dual_PEP_with_known_stepsizes, λ[i_j_idx(k, k-1)] == λ[i_j_idx(k-1, k)])
	   @constraint(model_dual_PEP_with_known_stepsizes, λ[i_j_idx(k, k-1)] + λ[i_j_idx(k-1, k)] == 2*(L-μ))
	   
	   @constraint(model_dual_PEP_with_known_stepsizes, τ[2] == (2*β_kM1 - γ_kM1*(L+μ))+2*θ)

	   @constraint(model_dual_PEP_with_known_stepsizes, τ[1] == γ_kM1*(L+μ) - (2*sqrt(β_kM1)/sqrt(-c_0_input_initial + c_0_input_initial^2)))

	   @constraint(model_dual_PEP_with_known_stepsizes, θ == 1/c_0_input_initial)
	   # @constraint(model_dual_PEP_with_known_stepsizes, ρ == 0.5*(2*β_kM1^2 + γ_kM1*θ*μ + L*γ_kM1*(θ - 2*γ_kM1*μ)))
	   # @constraint(model_dual_PEP_with_known_stepsizes, θ == (2*β_kM1)/(L-μ))
	   # @constraint(model_dual_PEP_with_known_stepsizes, τ[1] == 0)
	   # @constraint(model_dual_PEP_with_known_stepsizes, θ == γ_kM1*(L+μ) / sqrt(4*β_kM1 + c_0_input_initial*γ_kM1^2*(L+μ)^2) )
	   # @constraint(model_dual_PEP_with_known_stepsizes, ρ == -τ[4])
	   # @constraint(model_dual_PEP_with_known_stepsizes, λ[i_j_idx(k, k-1)] + λ[i_j_idx(k-1, k)] == 2*τ[2]*(L-μ))
	   # @constraint(model_dual_PEP_with_known_stepsizes, τ[1] == γ_kM1*(L+μ)*0.5)
       # @constraint(model_dual_PEP_with_known_stepsizes, (L-μ) == τ[2])
	   # @constraint(model_dual_PEP_with_known_stepsizes, Z[2,2] == Z[3,3])
	   # @constraint(model_dual_PEP_with_known_stepsizes, λ[i_j_idx(k, k-1)] == λ_fix_term)
	end

	## Time to optimize

		if show_output == :off
		set_silent(model_dual_PEP_with_known_stepsizes)
	end

	optimize!(model_dual_PEP_with_known_stepsizes)

	@show termination_status(model_dual_PEP_with_known_stepsizes)

	if termination_status(model_dual_PEP_with_known_stepsizes) != MOI.OPTIMAL
		@info "💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀"
		@error "model_dual_PEP_with_known_stepsizes solving did not reach optimality;  termination status = " termination_status(model_dual_PEP_with_known_stepsizes)
	end

	## Store the solution and return

		# store the solutions and return
	# ------------------------------

	# store λ_opt

	λ_opt = value.(λ)

	# store ν_opt

	ν_opt = value.(ν)

	# store τ_opt

	τ_opt = value.(τ)

	# store ρ_opt

	ρ_opt = value.(ρ)

	# store Z_opt

	Z_opt = value.(Z)

	# store θ_opt

	θ_opt = value.(θ)

	# compute cholesky

	L_cholesky_opt_Z =  compute_pivoted_cholesky_L_mat(Z_opt)

	if norm(Z_opt - L_cholesky_opt_Z*L_cholesky_opt_Z', Inf) > 1e-6
		@info "checking the norm bound"
		@warn "||Z - L*L^T|| = $(norm(Z_opt - L_cholesky_opt_Z*L_cholesky_opt_Z', Inf))"
	end

	## effective index sets for the dual variables λ, μ, ν

	idx_set_λ_effective = effective_index_set_finder(λ_opt ; ϵ_tol = 0.00005)

	# store objective

	ℓ_1_norm_λ = sum(λ_opt)

	tr_Z = tr(Z_opt)

	original_performance_measure = ν_opt

	@info "[🌞 ] DUALSDP-POLISHED ratio = $(original_performance_measure)"

	# polyak_contraction_factor = (1 - (μ/L)* (1/(1 + (L/μ)^2)))

		if η_2_input == 1
		@info "NCG-PEP Theory {||p_k||^2/||g_k||^2} = $((L+μ)^2/(4*μ*L))"
	elseif η_2_input == 0
		@info "NCG-PEP Theory {||p_k||^2/||g_k||^2} =  $((((L - μ)^2 + 4*c_0_input_initial*L*μ + 4*(L - μ)*sqrt((-1 + c_0_input_initial)*L*μ))/(4*L*μ)))"
	end

	# return all the stored values

	return original_performance_measure, ℓ_1_norm_λ, tr_Z, λ_opt, ν_opt, τ_opt, ρ_opt, Z_opt, L_cholesky_opt_Z, idx_set_λ_effective, θ_opt


end



function Schur_Complement_Master(M)

	M = copy(M)

	m, n = size(M)

	if m != n
		@error "sqaure matrix required"
		return
	end

	if M[n,n] <= 1e-4
		@error "cannot take Schur complement M[n,n]=0"
		return 0
	end

	for i in 1:n
		for j in 1:n
			if abs(M[i,j]) <= 1e-6
				M[i,j]=0.0
			end
		end
	end

	A = M[1:n-1,1:n-1]

	X = M[1:n-1,n]

	B = M[n, n]

	S = A - (1/B)*(X*transpose(X))

	display(S)

	@show eigvals(S)

	# S = round.(S, digits = 6)

	return S
end

