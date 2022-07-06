#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:22:12 2021

@author: Rishi Sreedhar ( https://orcid.org/0000-0002-7648-4908 )
"""
'''

This code is for calculating the optimum angles for P = p MPS-QAOA applied to 
Erdos Renyi MaxCut instances for different Bond-dimensions. 
This code produces the data obtained in Section V, and Appendices F, G of the paper.

The angles are calculated using the Bayesian Optimisation Technique, using
GPyOpt package.

Input Needed::
    
    1.) C: The adjacency matrix describing the MaxCut problem instance.
    2.) Instance_list: The list of instances to be studied.
    3.) p: The circuit depth to be studied
    4.) DList: The list of different bond-dimensions to be studied
    5.) M: The number of times Bayesian Optimizations have to be repeated
    6.) init_num: Initial Number of functional evaluations before bayesian optimization
    7.) total_fevals: Total Number of F_evals including initializations
    8.) max_time: Maximum time in hours for which each Bayesian Optimization process is allowed to continue.

Output Generated::
    
    1.) Global_Cost_Min: The final minimized cost obtained after repeating Bayesian Optimization having total_fevals number of 
                         functional evaluations, M number of times.
                         
    2.) Global_Para_Opt: A list of the 2*p QAOA parameters corresponding to a cost value of Global_Cost_Min.


    
'''

#%%

import os

import tensornetwork as tn
from Gates import Gates as g
from Expectation import Expectation as exp
import numpy as np
from GPyOpt.methods import BayesianOptimization as BOpt

import matplotlib.pyplot as plt

import time



#%%

##############################
## Parameter Initialization ##
##############################

q = 12 # number of Qubits

folder_location = os.path.dirname(os.path.realpath(__file__)) # Location of the MPS_QAOA Folder
local_location = '/MxC_Q'+str(q)+'/Q' #Location of files within the MPS_QAOA Folder

location = folder_location + local_location

tag = 'R' # R for Random, P for Pontus

p = 1   # QAOA Circuit Depth

M = 5 if (p < 3) else 300 if (p == 3) else 300   # Number of Repeatitions of full Bayesian Optimisation 
max_time = 0.25 if (p == 1) else 1.5 if (p == 2) else 3   # Maximum Time in Hours per repition
max_time = int(max_time*3600)   # Converting into time in seconds

total_fevals = 125 if (p==1) else 500 if (p==2) else 400 # Total Number of F_evals including initializations
init_num = 50 if (p==1) else 200 if (p==2) else 150   # Initial Number of sampling points

max_iter = total_fevals - init_num   # Maximum Number of F_evals after initialisations

Dhighest = int(2**np.floor(q/2))
D_list = [64, 48, 32, 24, 16, 12, 8, 6, 4, 2] #List of Bond-dimensions to be simulated
    

#%%


def get_Cost_block(gamma_beta):
    
    '''
    
    Function to calculate the Cost/Energy corresponding to each state gamma_beta
    given a MaxCut Hamiltonian described using the adjacency matrix C
    
    Input::
        
        gamma_beta: The input quantum state whose energy is to be calculated
        C: The adjacency matrix defining the cost Hamiltonian H_C.
    
    Output::
        
        Cost: Cost value corresponding to each input state gamma_beta and Hamiltonian H_C.
            
    
    '''
    
    Cost = 0
    
    z_i = g.get_Z()
    z_j = g.get_Z()
    
    for i in range(n-1):
        
        for j in range((n-1),i,-1):
            
            g_b = gamma_beta.tensors
            GB = tn.FiniteMPS(g_b, canonicalize = False)
            GB_copy = tn.FiniteMPS(g_b, canonicalize = False)
            
            GB.apply_one_site_gate(z_i,i)
        
            GB.apply_one_site_gate(z_j,j)
            
            ci = exp.exp_MPS_II(GB_copy,GB)
            ci = np.real(ci)
            
            Cost = Cost - 0.5*C[i,j]*(1 - ci)        
    
    
    return Cost

#%%


def QAOA_gamma_block(gamma_beta, gamma):
    
    '''
    
    Function that takes in the parameter gamma and applies
    a single layer of the gamma block within a single QAOA layer.
    
    Input::
        
        gamma_beta: An existing QAOA state upon which one applies the Gamma layer
        gamma: The free parameter that needs to be optimized
        
    Output::
        
        gamma_beta: The QAOA state with an additional Cost layer of the QAOA added to it
    
    '''
    
    gamma_beta = tn.FiniteMPS(gamma_beta.tensors, canonicalize=False)
    gamma_beta.canonicalize(normalize=Normalize)
    
    # Defining the SWAP network
    Q_ord = np.linspace(start = 0, stop = (n-1), num = n, dtype = int) 
    
    SWAP = g.get_SWAP()
    
    for i in range(n): #applying all the nearest neighbout gates
        
        if (i < (n-1)):
            
            for k in range(n-1):
                
                if (Q_ord[k] < Q_ord[k+1]):
                    
                    Cij = g.get_Cij(gamma, C[Q_ord[k]][Q_ord[k+1]])
                    
                    gamma_beta.position(site=k, normalize=Normalize)
                    
                    if (Dmax == Dhighest):
                        
                        gamma_beta.apply_two_site_gate(Cij, site1 = k,
                                                       site2 = (k+1), center_position=k)
                        
                    else:
                        
                        gamma_beta.apply_two_site_gate(Cij, site1 = k, site2 = (k+1),
                                                   max_singular_values=Dmax, center_position=k)
                        
         
        if (i%2 == 0): #Doing the even round of SWAPs from the SWAP network
            
            for s in range(l1):
                
                Q_ord[2*s],Q_ord[2*s+1] = Q_ord[2*s+1],Q_ord[2*s]
                
                gamma_beta.position(site=(2*s), normalize=Normalize)
                
                if (Dmax == Dhighest):
                    
                    gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s),
                                                   site2 = (2*s+1), center_position=(2*s))
                
                else:
                    
                    gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s), site2 = (2*s+1),
                                                   max_singular_values=Dmax, center_position=(2*s))
            
        else:  #Doing the odd round of SWAPs from the SWAP network
            
            for s in range(l2):
                
                Q_ord[2*s+1],Q_ord[2*s+2] = Q_ord[2*s+2],Q_ord[2*s+1]
                
                gamma_beta.position(site=(2*s+1), normalize=Normalize)
                
                if (Dmax == Dhighest):
                    
                    gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s+1),
                                                   site2 = (2*s+2), center_position=(2*s+1))
                    
                else:
                    
                    gamma_beta.apply_two_site_gate(SWAP,site1 = (2*s+1), site2 = (2*s+2),
                                                   max_singular_values=Dmax, center_position=(2*s+1))
         
    gamma_beta = gamma_beta.tensors[::-1]
    gamma_beta = [gamma_beta[x].transpose([2,1,0]) for x in range(n)]
    gamma_beta = tn.FiniteMPS(gamma_beta, canonicalize = False)
    
    return gamma_beta

#%%


def QAOA_beta_block(gamma_beta, beta):
    
    '''
    
    Function that takes in the parameter beta and applies
    a single layer of the mixing block within a single QAOA layer.
    
    Input::
        
        gamma_beta: An existing QAOA state upon which one applies the Gamma layer
        beta: The free parameter that needs to be optimized
        
    Output::
        
        gamma_beta: The QAOA state with an additional mixing layer of the QAOA added to it
    
    '''
        
    gamma_beta = tn.FiniteMPS(gamma_beta.tensors, canonicalize=False)
    
    Rx = g.get_Rx(2*beta)
        
    for i in range(n):
        
        gamma_beta.apply_one_site_gate(Rx, i)
            
    return gamma_beta



#%%


def QAOA_state(Ga, Be):
    
    '''
    
    Function to create a full p-layer QAOA state by using the lists Ga and Be
    
    Input::
        
        Ga: A list of p gamma parameters
        Be: A list of p beta parameters
        
    Output::
        
        gamma_beta: The QAOA state with an additional Cost layer of the QAOA added to it
    
    '''
    
    GB = QAOA_gamma_block(plus, Ga[0])
    GB = QAOA_beta_block(GB, Be[0])
    
    for i in range(1,p):
        
        GB = QAOA_gamma_block(GB, Ga[i])
        GB = QAOA_beta_block(GB, Be[i])
    
    return GB
    

#%%


def QAOA_Cost(GaBe):
    
    '''
    A function that resturns the cost corresponding to each gamma beta set of angles.
    The GPyOpt optimizer needs a function that takes in a single 
    list of all parameters to be optimized. 
    GaBe is a list of 2p parameters that act as the corresponding QAOA parameters.
    
    Input::
        
        GaBe: A list of the 2p parameters that need to be optimized
        
    Output::
        
        Cost: The cost value corresponding to each gamma beta pair.
        This is the function given as an input to the Bayesian optimizer
        and the corresponding cost is the one to be minimized
        
    '''
    
    Ga = GaBe[0][:p]
    Be = GaBe[0][p:]
    
    #### Have added While loops with Fail criteria to ensure repetiotions
    #### to avoid errors due to the minuscule singular values.
    
    Fail = True
    attempt = 0
    while (Fail):
        
        try:
            
            GB = QAOA_state(Ga, Be)
            Cost = get_Cost_block(GB)
            Fail = False
            
        except:
                        
            time.sleep(1.5)
            attempt += 1
            Ga = np.round(Ga, decimals=(11-attempt))
            
            if (attempt > 10):
                print('\n\nStopping Repetition in cost at attempt = ',attempt,'\n\n')
                raise
    
    
    return Cost



#%%


def Sol_Evolution(Angles, Cost_vals):
    
    '''
    
    A function to calculate how the success probability evolves with the cost
    as a function of the different iterations of Bayesian Optimization tool, GPyOpt.
    Let num_iter be the number of iterations of GPyOpt.
    
    Input::
        
        Angles: a num_iter x 2p array of angles corresponding to each optimization iteration
        Cos_vals: The num_iter cost values returned by GPyOpt correspinding to each iteration
        
    Output::
        
        Sol_evol: A list with num_iter elements that depict how the solution expectation.
     
    '''
    
    
    Sol_evol = np.zeros(len(Cost_vals))
    
    Fail = True
    attempt = 0
    while (Fail):
        
        try:
            
            GB = QAOA_state(Angles[0][:p], Angles[0][p:])
            GB.canonicalize(normalize=True)
            Fail = False
            
        except:
            
            
            time.sleep(1.5)
            attempt += 1
            Angles[0][:p] = np.round(Angles[0][:p], decimals=(11-attempt))
            
            if (attempt > 10):
                print('\n\nStopping Repetition in Sol at attempt = ',attempt,'\n\n')
                raise
    
    expt = 0.0
    for s in range(num_Sol):
        expt = expt + abs(exp.exp_MPS_II(Sol_MPS_list[s],GB))**2
    
    Sol_evol[0] = expt
    C_min = Cost_vals[0]
    
    for k in range(1,len(Cost_vals)):
                
        if (Cost_vals[k] < C_min):
            
            C_min = Cost_vals[k]
            
            Fail = True
            attempt = 0
            while (Fail):
                
                try:
                    
                    GB = QAOA_state(Angles[k][:p], Angles[k][p:])
                    GB.canonicalize(normalize=True)
                    Fail = False
                    
                except:
                    
                    time.sleep(1.5)
                    attempt += 1
                    Angles[k][:p] = np.round(Angles[k][:p], decimals=(11-attempt))
                    
                    if (attempt > 10):
                        print('\n\nStopping Repetition in Sol at attempt = ',attempt,'\n\n')
                        raise
                        
            
            expt = 0.0
            for s in range(num_Sol):
                expt = expt + abs(exp.exp_MPS_II(Sol_MPS_list[s],GB))**2
                
            Sol_evol[k] = expt
            
        else:
            
            Sol_evol[k] = Sol_evol[k-1]
            Cost_vals[k] = Cost_vals[k-1]
            
    return Sol_evol
    

#%%


def Convergence_Plot(CostEvol, SolEvol, STR):
    
    """
    
    Function for the cost and solution convergence plots deicting how these two values 
    improve with time.
    
    """
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(23,10),dpi=80)
    
    fig.suptitle('Q'+str(n)+tag+str(r)+' : D = '+str(Dmax)+
                 ' : P = '+str(p)+' : Bayes Opt with '+str(total_fevals)+
                  ' f_evals over '+str(M)+' runs\n',fontweight='bold', fontsize=22)
    
    ax[0].set_xlabel('Iteration', fontweight='bold', fontsize=18)
    ax[0].set_ylabel('Cost', fontweight='bold', fontsize=18)
    ax[0].set_title(STR+' Cost Convergence', fontweight='bold', fontsize=20)
    ax[0].grid(linewidth=2)
    ax[0].tick_params(axis='x', labelsize=14, width = 2)
    ax[0].tick_params(axis='y', labelsize=16, width = 2)
    
    ax[1].set_xlabel('Iteration', fontweight='bold', fontsize=18)
    ax[1].set_ylabel('Solution Expectation', fontweight='bold', fontsize=18)
    ax[1].set_title(STR+' Expectation Evolution', fontweight='bold', fontsize=20)
    ax[1].grid(linewidth=2)
    ax[1].tick_params(axis='x', labelsize=14, width = 2)
    ax[1].tick_params(axis='y', labelsize=16, width = 2)
    
    ax[0].plot(CostEvol, color = 'black',
               linestyle = 'solid', marker = 'o',
               markerfacecolor = 'black', markersize = 14)
    
    ax[1].plot(SolEvol, color = 'black',
               linestyle = 'solid', marker = 'o',
               markerfacecolor = 'black', markersize = 14)
    
    plt.show()
    
#%%

def polar_plots(Cost_Min_list, Gamma_Opt_list, Beta_Opt_list, Para_Opt, Iter):
    
    '''
    
    Code to make polar plots depicting the different optimum parameters obtained 
    for each iteration of the Bayesian optimizer.
    
    Cost_Min_list: The minimum cost calculated for each iteration, so as to decide on
                   the radius assigned to each angle point depending on how low the cost is.
                   
    Gamma_Opt_list: List of Optimum gamma angles found for each iteration
    
    Beta_Opt_list: List of Optimum beta angles found for each iteration
    
    Para_Opt: Best (gamma, beta) set calculated among all iterations yet
    
    Iter: The iter^th iteration (Just for a title)
            
    '''
        
    fig, ax = plt.subplots(nrows=1, ncols=2,
                            subplot_kw=dict(projection='polar'), figsize=(19,10),dpi=80)
    
    fig.suptitle('Q'+str(n)+tag+str(r)+' : D = '+str(Dmax)+
                  ' : P = '+str(p)+' : Bayes Opt with '+str(total_fevals)+
                  ' f_evals over '+str(Iter)+' runs.\n', fontweight='bold', fontsize=22)
    
    Point_Size = (abs(Cost_Min_list[:Iter]) - min(abs(Cost_Min_list[:Iter])))*10
    Point_Size = (Point_Size + 1)*60
    
    ax[0].set_xticks(np.arange(0,2.0*np.pi,np.pi/4.0))
    ax[0].set_yticks(np.arange(1,p+3,1.0))
    ax[0].set_ylim(0,p+2)
    ax[0].set_rlabel_position(270)
    ax[0].set_title('Optimum Gamma Distribution', fontweight='bold', fontsize=16)
    ax[0].tick_params(axis='x', labelsize=14, width = 2)
    ax[0].tick_params(axis='y', labelsize=16, width = 2)
    
    
    for P in range(p):
        
        ax[0].scatter(Gamma_Opt_list[:Iter,P],
                      np.array([P+1 for x in range(Iter)]),s=Point_Size)
        
        ax[0].scatter(Para_Opt[P],P+1,c='black',s=100,marker='o')
    
    
    ax[1].set_xticks(np.arange(0,2.0*np.pi,np.pi/4.0))
    ax[1].set_yticks(np.arange(1,p+3,1.0))
    ax[1].set_ylim(0,p+2)
    ax[1].set_rlabel_position(270)
    ax[1].set_title('Optimum Beta Distribution', fontweight='bold', fontsize=16)
    ax[1].tick_params(axis='x', labelsize=14, width = 2)
    ax[1].tick_params(axis='y', labelsize=16, width = 2)
    
    for P in range(p):
        
        ax[1].scatter(Beta_Opt_list[:Iter,P],
                      np.array([P+1 for x in range(Iter)]),s=Point_Size)
        
        ax[1].scatter(Para_Opt[p+P],P+1,c='black',s=100,marker='o')
    
    plt.show()
    


#%%

instance_list = [0]
for r in instance_list: # Loop looping over the different random instances
    
    C = np.load(location+str(q)+tag+str(r)+'/C_Q'+str(q)+tag+str(r)+'.npy')
    
    n = len(C)    
    l1 = int(np.floor((n)/2))
    l2 = int(np.floor((n-1)/2))
    
    Normalize = False
    
    plus = tn.FiniteMPS([np.array([[[1/np.sqrt(2)],
                                    [1/np.sqrt(2)]]],
                                  dtype = np.complex128) for x in range(n)])
        
    
    ################# Solution Extraction #################
    
    
    Zero = np.array([[[1.],
                      [0.]]])
    One = np.array([[[0.],
                     [1.]]])
    
    Sol_MPS_list = []
    for j in range(1,100):
        
        try:
            
            S = np.load(location+str(q)+tag+str(r)+'/Sol_str_EL0_'+str(j)+'.npy')
            S = str(S)
            
            S_mps = []
            for k in range(n):
                
                if (S[k] == '0'):
                    
                    S_mps.append(Zero.copy())
                    
                else:
                    
                    S_mps.append(One.copy())
                    
            
            S_mps = tn.FiniteMPS(S_mps)
            Sol_MPS_list.append(S_mps)
            
        except:
            continue
    
    num_Sol = len(Sol_MPS_list)
    
    
    ################# End Solution Extraction #################    
    
    
    Cdata_D_Cmin_Ga_Be = np.zeros([len(D_list),2*p+2]) # A multi-dimensional array that 
                                                       # stores the minimum cost and parameters for each D
                                                       
    Global_Cdata_D_Cmin_Ga_Be = np.zeros([len(D_list),2*p+2])
    
    for d_ind, Dmax in enumerate(D_list): # Loop looping over the different bond-dimensions
        
        bounds = []
        Gamma_set = []
        Beta_set = []
        pi = np.pi
        
        Ga_ub = pi
        Ga_lb = 0
        
        Be_ub = 0.5*pi
        Be_lb = 0
        
        
        for P in range(p):
            
            Gamma_set += [{'name': 'gamma'+str(P+1),
                           'type': 'continuous', 'domain': (Ga_lb,Ga_ub)}]
            
            Beta_set += [{'name': 'beta'+str(P+1),
                          'type': 'continuous', 'domain': (Be_lb,Be_ub)}]
            
        bounds = Gamma_set + Beta_set
        
        M_start = 0
        
        Gamma_Opt_list = np.zeros([M,p])
        Beta_Opt_list = np.zeros([M,p])
        Cost_Min_list = np.zeros(M)
        
        Cost_MinMin = 0
        Para_Opt = np.zeros(2*p)
            
        Global_Para_Opt = np.zeros(2*p)
        Global_Cost_Min = 0
        
        CostEval_List = [None]*M
        SolExp_List = [None]*M
        
        for i in range(M_start,M):
            # it = 1
            t0 = time.time()
            
            # print('Q'+str(n)+tag+str(r)+' : D = '+str(Dmax)+' : Tot = '+str(total_fevals)+
            #       ' : init = '+str(init_num)+' : P = '+str(p)+' : iteration = ',i,'\n')
            
            ###################################
            ## Running Bayesian Optimization ##
            ###################################
            
            Fail = True
            attempt = 0
            while (Fail):
                
                try:
                    
                    myBopt = BOpt(f=QAOA_Cost, domain=bounds, model_type = 'GP',
                                  initial_design_numdata = init_num, initial_design_type = 'latin',
                                  acquisition_type='EI', exact_feval = True)
                    
                    myBopt.run_optimization(max_iter=max_iter, eps=1e-3, max_time=max_time)
                    Fail = False
                    
                except:
                    
                    attempt += 1
                    time.sleep(1.5)
                    
                    if (attempt > 1):
                        print('\n\nStopping Bayes_Opt after ',attempt,' attempts!!\n\n')
                        raise
                        
            
            Cost_Min_list[i] = myBopt.fx_opt
            Gamma_Opt_list[i,:] = myBopt.x_opt[:p]
            Beta_Opt_list[i,:] = myBopt.x_opt[p:]
            
            
            Angles,CostEval_List[i] = myBopt.get_evaluations()
            
            Fail = True
            Sol_evol_status = ''
            attempt = 0
            while (Fail):
                
                try:
                    
                    SolExp_List[i] = Sol_Evolution(Angles, CostEval_List[i])
                    Sol_evol_status = 'Y'
                    Fail = False
                    
                except:
                    
                    time.sleep(1.5)
                    attempt += 1
                    
                    if (attempt > 10):
                        print('\n\nStopping Sol Evol after ',attempt,' attempts!!\n\n')
                        Sol_evol_status = 'N'
                        Fail = False
            
            if (Cost_Min_list[i] < Cost_MinMin):
                
                Cost_MinMin = Cost_Min_list[i]
                Para_Opt = myBopt.x_opt
                
                
            ##################
            ## Global Check ##
            ##################
            
            if (Cost_MinMin < Global_Cost_Min):
                
                Global_Cost_Min = Cost_MinMin
                Global_Para_Opt = Para_Opt
                
                print('\n New Global Minimum found for Q'+str(n)+tag+str(r)+
                      ' : D = '+str(Dmax)+' : P = '+str(p)+' !! \n\n')
                
                
                #####################
                ## Plotting Global ##
                #####################
                
                fig, ax = plt.subplots(nrows=1, ncols=2,subplot_kw = 
                                       dict(projection='polar'), figsize=(19,10),dpi=80)
                
                fig.suptitle('Q'+str(n)+tag+str(r)+' : D = '+str(Dmax)+
                              ' : P = '+str(p), fontweight='bold', fontsize=22)
                
                ax[0].set_xticks(np.arange(0,2.0*np.pi,np.pi/4.0))
                ax[0].set_yticks(np.arange(1,p+3,1.0))
                ax[0].set_ylim(0,p+2)
                ax[0].set_rlabel_position(270)
                ax[0].set_title('Global Optimum : Gamma',
                                fontweight='bold', fontsize=16)
                ax[0].tick_params(axis='x', labelsize=14, width = 2)
                ax[0].tick_params(axis='y', labelsize=16, width = 2)
                
                
                for P in range(p):
                    
                    ax[0].scatter(Global_Para_Opt[P],P+1,c='red',s=100,marker='o')
                
                
                ax[1].set_xticks(np.arange(0,2.0*np.pi,np.pi/4.0))
                ax[1].set_yticks(np.arange(1,p+3,1.0))
                ax[1].set_ylim(0,p+2)
                ax[1].set_rlabel_position(270)
                ax[1].set_title('Global Optimum : Beta',
                                fontweight='bold', fontsize=16)
                ax[1].tick_params(axis='x', labelsize=14, width = 2)
                ax[1].tick_params(axis='y', labelsize=16, width = 2)
                
                for P in range(p):
                    
                    ax[1].scatter(Global_Para_Opt[p+P],P+1,c='red',s=100,marker='o')
                
                plt.show()
                
                # Insert Command here to save plot
                
                #######################################################
            
            polar_plots(Cost_Min_list, Gamma_Opt_list, Beta_Opt_list, Para_Opt, i+1)
            
            del myBopt
            
        print('The 2*'+str(p)+' Optimized parameters [gamma, beta] are: '+str(Global_Para_Opt)+
              ' and minimum cost obtained is: '+str(Global_Cost_Min))
        
        np.save(location+str(q)+tag+str(r)+'/BestParaOpt_D'
                +str(Dmax)+'_P'+str(p)+'.npy',Global_Para_Opt)
        
        np.save(location+str(q)+tag+str(r)+'/BestCostMin_D'
                +str(Dmax)+'_P'+str(p)+'.npy',Global_Cost_Min)            
