#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:54:34 2021

@author: Rishi Sreedhar ( https://orcid.org/0000-0002-7648-4908 )

Code created to use the parameters derived from MPS_parameter_optimization.py to calculate the 
solution expectation purely using MPS-QAOA. This code generates the data presented in 
Section V.A: QAOA Performances with approximated training, Figure 9, and Appendix H, Figure 16.


"""

'''

This code is for calculating the optimum angles for P = p MPS-QAOA 
applied to Erdos Renyi MaxCut instances for different Bond-dimensions.

The angles are calculated using the Bayesian Optimisation Technique, using
GPyOpt package.

##########################################################################################
## IMPORTANT:: Enter the optimum parameters here and replace the random function belore ##
##########################################################################################

    
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

tag = 'R' # R for Random

p = 1   # QAOA Circuit Depth

D_highest = int(2**np.floor(q/2))
D_list = [64, 48, 32, 24, 16, 12, 8, 6, 4, 2] #List of Bond-dimensions to be simulated

instance_list = [0]

       
#%%


def QAOA_gamma_block(gamma_beta, gamma, Dmax, C):
    
    '''
    
    Function that takes in the parameter gamma and applies
    a single layer of the gamma block within a single QAOA layer.
    
    Input::
        
        gamma_beta  : An existing QAOA state upon which one applies the Gamma layer
        gamma       : The free parameter that needs to be optimized
        Dmax        : The maximum bond-dimension limit imposed on the MPSs
        C           : The adjacency matrix describing the MaxCut problem instance.
        
    Output::
        
        gamma_beta  : The QAOA state with an additional Cost layer of the QAOA added to it
    
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
                    
                    if (Dmax == D_highest):
                        
                        gamma_beta.apply_two_site_gate(Cij, site1 = k,
                                                       site2 = (k+1), center_position=k)
                        
                    else:
                        
                        gamma_beta.apply_two_site_gate(Cij, site1 = k, site2 = (k+1),
                                                   max_singular_values=Dmax, center_position=k)
                        
         
        if (i%2 == 0): #Doing the even round of SWAPs from the SWAP network
            
            for s in range(l1):
                
                Q_ord[2*s],Q_ord[2*s+1] = Q_ord[2*s+1],Q_ord[2*s]
                
                gamma_beta.position(site=(2*s), normalize=Normalize)
                
                if (Dmax == D_highest):
                    
                    gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s),
                                                   site2 = (2*s+1), center_position=(2*s))
                
                else:
                    
                    gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s), site2 = (2*s+1),
                                                   max_singular_values=Dmax, center_position=(2*s))
            
        else:  #Doing the odd round of SWAPs from the SWAP network
            
            for s in range(l2):
                
                Q_ord[2*s+1],Q_ord[2*s+2] = Q_ord[2*s+2],Q_ord[2*s+1]
                
                gamma_beta.position(site=(2*s+1), normalize=Normalize)
                
                if (Dmax == D_highest):
                    
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


def QAOA_state(Ga, Be, p_ind, Dmax, C):
    
    '''
    
    Function to create a full p-layer QAOA state by using the lists Ga and Be
    
    Input::
        
        Ga         : A list of p gamma parameters
        Be         : A list of p beta parameters
        p_ind      : The circuit depth to be prepared from the full p-layer QAOA
        Dmax       : The maximum bond-dimension limit imposed on MPSs
        C          : The adjacency matric describing the MaxCut problem instance
        
    Output::
        
        GB         : The QAOA state with an additional Cost layer of the QAOA added to it
    
    '''
    
    GB = QAOA_gamma_block(plus, Ga[0], Dmax, C)
    GB = QAOA_beta_block(GB, Be[0])
    
    for i in range(1,p_ind):
        
        GB = QAOA_gamma_block(GB, Ga[i])
        GB = QAOA_beta_block(GB, Be[i])
    
    return GB
    

#%%


def Sol_Expectation(Para_Opt, p_ind, Dmax):
    
    '''
    
    Function to calculate the probability of obtaining the solution from a given QAOA state
    defined using the 2*p optimum parameters Para_Opt.
    
    Input::
        
        Para_opt   : A list of 2*p optimum parameters (gamma,beta)
        p_ind      : The layer upto which the state has to be prepared within the full p-layer QAOA
        Dmax       : The maximum bond-dimension limit imposed on the QAOA state.
                     If Dmax == D_highest, then MPS state is the exact QAOA state.
        
    Output::
        
        Sol_expt   : The net success probability of obtaining solutions for the given QAOA state
                     with parameters = Para_Opt, circuit depth = p_ind, and bond-dimension limit = Dmax
    
    '''
    
    Fail = True
    attempt = 0
    while (Fail):
        
        try:
            
            GB = QAOA_state(Para_Opt[:p_ind], Para_Opt[p:p+p_ind], p_ind, Dmax)
            GB.canonicalize(normalize=True)
            Fail = False
            
        except:
            
            attempt += 1
            Para_Opt[:p] = np.round(Para_Opt[:p], decimals=(11-attempt))
            
            if (attempt > 10):
                print('\n\nStopping Repetition in Sol at attempt = ',attempt,'\n\n')
                raise
    
    Sol_expt = 0.0
    for s in range(num_Sol):
        Sol_expt = Sol_expt + abs(exp.exp_MPS_II(Sol_MPS_list[s],GB))**2
    
    
    return Sol_expt
    

#%%

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
    
    
    
    Cdata_D_Cmin_Ga_Be = np.zeros([len(D_list),2*p+2]) 
    Sdata_Vqaoa_D_SolP = np.zeros([len(D_list),p+1])
    Sdata_MPSqaoa_D_SolP = np.zeros([len(D_list),p+1])
    
    
    for d_ind, Dmax in enumerate(D_list): # Loop looping over the different bond-dimensions
        
        ###################################################################################
        ## IMPORTANT:: Enter the optimum parameters here and replace the random function ##
        ###################################################################################
        
        if (os.path.isfile(location+str(q)+tag+str(r)+'/BestParaOpt_D'+str(Dmax)+'_P'+str(p)+'.npy')):
            
            Para_Opt = np.load(location+str(q)+tag+str(r)+'/BestParaOpt_D'+str(Dmax)+'_P'+str(p)+'.npy')
        
        else:
            
            Para_Opt = np.random.rand(2*p) # Choosing random values in case optimum parameters are not provided!
        
        Sdata_Vqaoa_D_SolP[d_ind,0] = Dmax
        Sdata_MPSqaoa_D_SolP[d_ind,0] = Dmax
        
        for p_ind in range(1,p+1):
            
            Sdata_Vqaoa_D_SolP[d_ind,p_ind] = Sol_Expectation(Para_Opt, p_ind, D_highest)
            Sdata_MPSqaoa_D_SolP[d_ind,p_ind] = Sol_Expectation(Para_Opt, p_ind, Dmax)
            
        try:
            
            os.remove(location+str(q)+tag+str(r)+'/Q'+str(q)
                      +tag+str(r)+'_P'+str(p)+'_Sdata_D_SolP.npy')
        except:
            
            pass
        
        
        np.save(location+str(q)+tag+str(r)+'/Q'+str(q)
                +tag+str(r)+'_P'+str(p)+'_Sdata_Vqaoa_D_SolP.npy',Sdata_Vqaoa_D_SolP)
        
        np.save(location+str(q)+tag+str(r)+'/Q'+str(q)
                +tag+str(r)+'_P'+str(p)+'_Sdata_MPSqaoa_D_SolP.npy',Sdata_MPSqaoa_D_SolP)
        
    print('For the given optimum parameters Para_Opt = '+str(Para_Opt)+
          ', \n The success probability obtained on using Para_Opt in QAOA state is: '+str(Sdata_Vqaoa_D_SolP)+
          ', and \n The success probability obtained on using Para_Opt in MPS state is: '+str(Sdata_MPSqaoa_D_SolP))

        