# MPS_QAOA

## Installation
    pip install tensornetwork == 0.4.5 (and dependencies)
    pip install GPyOpt == 1.2.6 (and dependencies)
    pip install matplotlib == 3.4.3 (and dependencies)
    pip install pyDoe == 0.3.8 (and dependencies)

## Folders
### Exact Cover 3
- **EC3_Q12**: Folder holding the ten randomly generated 12-qubit Exact Cover 3 (EC3) problem instances from which the angles used in deterministic sampling results were obtained using the INTERP strategy.
- **EC3_Q14**: Folder containing all hundred randomly generated 14-qubit Exact Cover 3 problem instances used in the deterministic sampling results.
- **EC3_Q40**: Folder containing all hundred randomly generated 40-qubit Exact Cover 3 problem instances used in the deterministic sampling results.
- **EC3_Q60**: Folder containing all ten randomly generated 60-qubit Exact Cover 3 problem instances used in the deterministic sampling results.

### MaxCut
- **MxC_Q12**: Folder holding the hundred randomly generated 12-qubit MaxCut problem instances. The fist ten of these instances (Q12R0 to Q12R9) were used to obtain the angles used in deterministic sampling results using the INTERP strategy. All hundred instances (Q12R0 - Q12R99) were used in calculating the optimal angle variations with bond-dimension D for circuit depths p = 1 and p = 2. The first ten instances again (Q12R0 - Q12R9) were further used to calculate the optimal angle variation with D for p = 3 and 4.
- **MxC_Q14**: Folder containing all hundred randomly generated 14-qubit MaxCut problem instances used in the deterministic sampling results.
- **MxC_Q40**: Folder containing all hundred randomly generated 40-qubit MaxCut problem instances used in the deterministic sampling results.
- **MxC_Q60**: Folder containing all ten randomly generated 60-qubit MaxCut problem instances used in the deterministic sampling results.

### Optimal Angles
- **Opt_angles_Q12**: This folder contains the average optimum angles derived from 12-qubit instances of both MaxCut and EC3 problems.

## Dependency Files
1. `Expectation.py` 
    * **Input:** Two Quantum states in MPS form
    * **Output:** A single scalar which is the expectation value
2. `Gates.py`
    * A Collection of all the common gates used for QAOA experessed as numpy tensors.

## Code Files

In each of these files, the required input variables to be modified are specified in a comment in the beginning of the code.

---

a.) `MaxCut_Deterministic_Sampling.py`: Code created to extract the solutions using the deterministic sampling method using MPS techniques on the QAOA. This code produces the data presented in Section IV: QAOA PERFORMANCES WITH RESTRICTED ENTANGLEMENT, Subsection A: Performances for MaxCut, Figure 4. (Note that the data from this code has been averaged over all the instances and plotted using a log scale in Figure 4, separately.)

**Input Needed:**
1. Optimum parameters `Gamma_Opt` and `Beta_Opt`
2. The adjacency matrix `C` describing the problem
3. Solution states to the corresponding problem to calculate approximation ratios
4. `Instance_list`: The list of instances to be studied.

**Output Generated:**
1. `Sol_Pred_String_DetSamp`: A [Plim x Dlim] array of the deterministic samlpe isolated from each MPS state corresponding to every (p,D) pairs. The binary values have been converted to equivalent decimal format for ease of storage. This array is saved in the respective instance folder by default.
2. `AppRatio_DetSample`: A [Plim x Dlim] array with the Approximation Ratio corresponding to the deterministic sample isolated from each MPS state corresponding to every (p,D) pairs. This array is saved in the respective instance folder by default.
---
b.) `EC3_Deterministic_Sampling.py`: Code created to extract the solutions using the deterministic sampling method using MPS techniques on the QAOA. This code produces the data presented in Section IV: QAOA PERFORMANCES WITH RESTRICTED ENTANGLEMENT, subsection B: Performances for EC3, Figure 5. (Note that the data from this code has been averaged over all the instances separately)

**Input Needed:**

1. Optimum parameters `Gamma_Opt` and `Beta_Opt`
2. The Interaction matrix `J` describing the problem
3. The self-energy matrix `h` describing the problem
4. Solution states to the corresponding problem to calculate eigenenergies
5. `Instance_list`: The list of instances to be studied.

**Output Generated:**

1. `Sol_Pred_String_DetSamp`: A [Plim x Dlim] array of the deterministic sample isolated from each MPS state corresponding to every (p,D) pairs. The binary values have been converted to equivalent decimal format for ease of storage. This array is saved in the respective instance folder by default.
2. `EigVal_DetSamp`: A [Plim x Dlim] array with the eigenenergy of the deterministic sample isolated from each MPS state corresponding to every (p,D) pairs. This array is saved in the respective instance folder by default.
---
c.) `MPS_fidelity_calculations.py`: Code created to calculate how the fidelity of QAOA states vary as a function of bond-dimension D and circuit depth p for different MaxCut and EC3 instances. This code generates the fidelity data used in Figure 6 of Subsection C: Entanglement in QAOA in Section IV: QAOA PERFORMANCES WITH RESTRICTED ENTANGLEMENT.

**Input Needed:**
1. Optimum parameters `Gamma_Opt` and `Beta_Opt`
2. `Problem_type` that specifies if we are looking at MaxCut or EC3 problems    
3. The Interaction matrix `J` describing the EC3 problem, the self-energy matrix h describing the EC3 problem, or The adjacency matrix `C` describing the MaxCut problem.
4. `Instance_list`: The list of instances to be studied.

**Output Generated:**
1. `GB_fidelity`: A [Plim x Dlim] array with the fidelity of each MPS state corresponding to every (p,D) pairs. This array is saved in the respective instance folder by default.
---
d.) `MaxCut_P1_GridSearch.py`: This code is for Calculating the P = 1 Landscape of MaxCut problems that are defined using adjacency matrices C. This code is used to generate the data displayed in Figure 7, of Section V: QAOA TRAINING WITH LOW ENTANGLEMENT IN MPS REPRESENTATIONS.

**Input Needed:**
1. `C`: The adjacency matrix describing the MaxCut problem instance.
2. `Instance_list`: The list of instances to be studied.
3. `N`: The number of grid points along each gamma or beta axis.
4. `DList`: The list of different bond-dimensions to be studied

**Output Generated:**
1. `Cost_mps`: An [N x N] matrix storing the cost values corresponding to each (gamma,beta) for each Dmax studied. This data is saved as a .npy file in the respective instance folders.
---
e.) `MPS_parameter_optimization.py`: This code is for calculating the optimum angles for P = p MPS-QAOA applied to Erdos Renyi MaxCut instances for different Bond-dimensions. This code produces the data obtained in Section V, and Appendices F, G of the paper.

**Input Needed:**
1. `C`: The adjacency matrix describing the MaxCut problem instance.
2. `Instance_list`: The list of instances to be studied.
3. `p`: The circuit depth to be studied
4. `DList`: The list of different bond-dimensions to be studied
5. `M`: The number of times Bayesian Optimizations have to be repeated
6. `init_num`: Initial Number of functional evaluations before bayesian optimization
7. `total_fevals`: Total Number of `F_evals` including initializations
8. `max_time`: Maximum time in hours for which each Bayesian Optimization process is allowed to continue.

**Output Generated:**
1. `Global_Cost_Min`: The final minimized cost obtained after repeating Bayesian Optimization having `total_fevals` number of functional evaluations, M number of times.
2. `Global_Para_Opt`: A list of the 2*p QAOA parameters corresponding to a cost value of `Global_Cost_Min`.
---
f.) `MPS_training_success_probabilities.py`: Code created to use the parameters derived from `MPS_parameter_optimization.py` to calculate the solution expectation purely using MPS-QAOA. This code generates the data presented in Section V.A: QAOA Performances with approximated training, Figure 9, and Appendix H, Figure 16.

**Input Needed:**
1. `C`: The adjacency matrix describing the MaxCut problem instance.
2. `Instance_list`: The list of instances to be studied.
3. `p`: The circuit depth to be studied
4. `DList`: The list of different bond-dimensions to be studied
5. **IMPORTANT**: `Para_Opt`: The optimum parameters needed to create the optimized QAOA state.

**Output Generated:**
1. `Sdata_Vqaoa_D_SolP`: A [Dnum x (p+1)] array where the first coulum stands for the bond-dimension. The subsequent p columns stand for how the success probability evolved with each p. This data is on using the `Para_opt` parameters on Exact QAOA states.
2. `Sdata_MPSqaoa_D_SolP`: A [Dnum x (p+1)] array where the first coulum stands for the bond-dimension. The subsequent p columns stand for how the success probability evolved with each p. This data is on using the `Para_opt` parameters on MPS QAOA states with possibly lesser D.
