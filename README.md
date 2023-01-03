# Code for *Nonlinear conjugate gradient methods: worst-case convergence rates and computer-assisted analyses*

<p align="center">
  <a href="#Usage">Usage</a> •
   <a href="#Language and solver installation">Language and solver installation</a> •
  <a href="#Citing">Citing</a> •
  <a href="#Reporting issues">Reporting issues</a> •
  <a href="#Contact">Contact</a> 
</p>



## Usage
The code in this repository can be used to reproduce and verify the results from the work:

> Nonlinear conjugate gradient methods: worst-case convergence rates and computer-assisted analyses

The repository contains three folders  `Code_for_Pseudogradient_Gradient_Ratio`,  `Code_for_NCG_PEP`, and `Symbolic_Verifications`. The usage of the files in these folders is  as follows.

###  `Code_for_Pseudogradient_Gradient_Ratio` 

This folder has the following main contents.

* `Ratio_of_pseudo_gradient_and_gradient.jl` This file contains the Julia code to solve the performance estimation problem to compute the worst-case search directions (problem <img src="https://latex.codecogs.com/svg.image?(\mathcal{D})" title="https://latex.codecogs.com/svg.image?(\mathcal{D})" /> in Section 2.1) for  Fletcher-Reeves (FR) and Polak-Ribière-Polyak (PRP), which are two famous nonlinear conjugate gradient methods. 
* `1. Example_Julia.ipynb` This Jupyter notebook is written in [Julia programming language](https://julialang.org/) and it shows how to use `Ratio_of_pseudo_gradient_and_gradient.jl` to solve a specific instance of the problem <img src="https://latex.codecogs.com/svg.image?(\mathcal{D})" title="https://latex.codecogs.com/svg.image?(\mathcal{D})" /> in Section 2.1.
* `2. Using_the_saved_datasets_Julia.ipynb` Using_the_saved_datasets_Julia.ipynb) This Jupyter notebook is written in Julia and it shows how to use the datasets saved in the subfolder `Saved_Output_Files` to reproduce Figure 5 and Figure 6 in Appendix A. It also shows how to extract relevant data to verify the results using the open-source Python package [PEPit](https://github.com/PerformanceEstimation/PEPit).
* `3. PEPIt_verification_Python.ipynb` This Jupyter notebook is written in Python and it shows how to verify the lower-bounds associated with the datasets saved in the folder `Saved_Output_Files`. 

###  `Code_for_NCG_PEP`

This folder has the following main contents.

* `N_point_NCG_PEP.jl` This file contains the Julia code to solve the performance estimation problems to compute the worst-case bounds (as defined by the problems <img src="https://latex.codecogs.com/svg.image?(\mathcal{B}_\textup{Lyapunov})" title="https://latex.codecogs.com/svg.image?(\mathcal{B}_\textup{Lyapunov})" /> and <img src="https://latex.codecogs.com/svg.image?(\mathcal{B}_\textup{exact})" title="https://latex.codecogs.com/svg.image?(\mathcal{B}_\textup{exact})" /> in Section 3.1) for  Fletcher-Reeves (FR) and Polak-Ribière-Polyak (PRP).
* `1. Example_Julia.ipynb` This Jupyter notebook is written in Julia and it shows how to use `N_point_NCG_PEP.jl` to solve a specific instances of <img src="https://latex.codecogs.com/svg.image?(\mathcal{B}_\textup{Lyapunov})" title="https://latex.codecogs.com/svg.image?(\mathcal{B}_\textup{Lyapunov})" /> and and <img src="https://latex.codecogs.com/svg.image?(\mathcal{B}_\textup{exact})" title="https://latex.codecogs.com/svg.image?(\mathcal{B}_\textup{exact})" /> in Section 3.1.
* `2. Using_the_saved_datasets_Julia.ipynb`  This Jupyter notebook is written in Julia and it shows how to use the datasets saved in the subfolder `Saved_Output_Files` to reproduce Figures 2, 3, and 4 in Sections 3.2 and 3.3. It also shows how to extract relevant data to verify the results using the open-source Python package (PEPit)[https://github.com/PerformanceEstimation/PEPit].
* `3. PEPIt_verification_Python.ipynb`] This Jupyter notebook is written in Python and it shows how to verify the lower-bounds associated with the datasets saved in the folder `Saved_Output_Files`. 

### `Symbolic_Verifications`

This folder contains the following contents.

* [Verify_PRP_Wolfram_Language.ipynb](https://github.com/Shuvomoy/NCG-PEP-code/blob/main/Symbolic_Verifications/Verify_PRP_Wolfram_Language.ipynb) This Jupyter notebook is written in the [Wolfram language](https://nicoguaro.github.io/posts/wolfram_jupyter/)  and is used to verify the algebra for establishing the  worst-case bound for PRP (Section 2.2.1 of the paper).

* [Verify_FR_Wolfram_Language.ipynb](https://github.com/Shuvomoy/NCG-PEP-code/blob/main/Symbolic_Verifications/Verify_FR_Wolfram_Language.ipynb) This Jupyter notebook is written in the Wolfram language and is used to verify the algebra for establishing the  worst-case bound for FR (Section 2.2.2 of the paper).

## Language and solver installation

Installation instructions for different languages and solvers can be found in the links below. 

* [Install Julia](https://julialang.org/downloads/) (required to run the `.jl` files)

* [Install Jupyter](https://jupyter.org/install) to open the Jupyter notebooks ( `.ipynb` files) 

  * Install the kernels for the Jupyter notebooks (required to run the `.ipynb` files)
  * [Install IJulia](https://github.com/JuliaLang/IJulia.jl) (required to run Julia in the `.ipynb` files)
  * [Install Wolfram language](https://nicoguaro.github.io/posts/wolfram_jupyter/) (required to run Wolfram language that is used for the symbolic verifications in the `.ipynb` files)
  * [Install PEPit](https://pypi.org/project/PEPit/) (required to verify the results)

* Install the optimization solvers

  * [Install Mosek](https://www.mosek.com/downloads/) 
  * [Install Gurobi](https://www.gurobi.com/downloads/)
  * [Install KNITRO](https://www.artelys.com/solvers/knitro/)

* To run the `.jl` files install the following `Julia` packages by pressing `]` and then running  the following command in `Julia REPL`:

  ```julia
  add SCS, JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools, DiffOpt, SparseArrays, KNITRO
  ```

For platform specific instructions for the solvers and troubleshooting in Julia, please see https://github.com/jump-dev/Gurobi.jl, https://github.com/jump-dev/KNITRO.jl, https://github.com/jump-dev/MosekTools.jl, and https://github.com/MOSEK/Mosek.jl. 

## Citing

If you find the paper useful, we request you to cite the following paper.

```
arxiv link tbd
```



## Reporting issues
Please report any issues via the [Github issue tracker](https://github.com/Shuvomoy/NExOS.jl/issues). All types of issues are welcome including bug reports, feature requests, implementation for a specific research problem and so on. Also, please feel free to send an email :email: to [sdgupta@mit.edu](mailto:sdgupta@mit.edu) or [adrien.taylor@inria.fr](mailto:adrien.taylor@inria.fr) if you want to say hi :rocket:!	



