#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:14:31 2021

@author: Rishi Sreedhar ( https://orcid.org/0000-0002-7648-4908 )
"""
"""

This code is for Calculating the P = 1 Landscape of MaxCut problems 
that are defined using adjacency matrices C.
This code is used to generate the data displayed in Figure 7, of Section V: 
QAOA TRAINING WITH LOW ENTANGLEMENT IN MPS REPRESENTATIONS.

Input Needed::
    
    1.) C: The adjacency matrix describing the MaxCut problem instance.
    2.) Instance_list: The list of instances to be studied.
    3.) N: The number of grid points along each gamma or beta axis.
    4.) DList: The list of different bond-dimensions to be studied
    

Output Generated::
    
    1.) Cost_mps: An [N x N] matrix storing the cost values corresponding to each (gamma,beta) for each Dmax studied.
                  This data is saved as a .npy file in the respective instance folders.


"""

import os
import tensornetwork as tn
from Gates import Gates as g
from Expectation import Expectation as exp
import numpy as np

tn.set_default_backend("numpy")

#%%


##############################
## Parameter Initialization ##
##############################

q = 12 # number of Qubits

l1 = int(np.floor((q)/2))   # Paramteres required for the SWAP network
l2 = int(np.floor((q-1)/2)) # Paramteres required for the SWAP network

N = 100 # Number of sampling points in Gamma and Beta
pi = np.pi

Gamma = [0.0 + x*pi/(N-1) for x in range(N)] # List of 100 Gamma angles between [0 , pi]
Beta = [0.0 + x*0.50*pi/(N-1) for x in range(N)]  # List of 100 Beta angles between [0 , pi/2]

G,B = np.meshgrid(Beta,Gamma) # Creating a mesh for the 3D plots

DList = [64, 48, 32, 24, 16, 12, 8, 6, 4, 2] # List of Bond-dimensions to be simulated

folder_location = os.path.dirname(os.path.realpath(__file__)) # Location of the MPS_QAOA Folder
local_location = '/MxC_Q'+str(q)+'/Q' #Location of files within the MPS_QAOA Folder
location = folder_location + local_location

tag = 'R' # R for Random

D_highest = int(2**np.floor(q/2)) # Highest possible bond-dimension for a q-qubit MPS

Normalize = False # Normalization constraint for the MPS-QAOA states. IMPORTANT: Keep False

Instance_list = [0] # The indices of the instances one wishes to run the calculations for. 
                    # Eg: If q = 12, enter [0,42] if the instances of interest are 12R0 and 12R42


#%%


def get_Cost_mpsIX(gamma_beta):
    
    Cost = 0
    
    z_i = g.get_Z()
    z_j = g.get_Z()
    
    for i in range(n-1):
        
        for j in range((n-1),i,-1):
            
            if (C[i,j] != 0):
                
                g_b = gamma_beta.tensors
                GB = tn.FiniteMPS(g_b, canonicalize = False)
                GB_copy = tn.FiniteMPS(g_b, canonicalize = False)
                
                GB.apply_one_site_gate(z_i,i)
            
                GB.apply_one_site_gate(z_j,j)
                
                ci = exp.exp_MPS(GB_copy,GB)
                ci = np.real(ci)
                
                Cost = Cost - 0.5*C[i,j]*(1 - ci)        
    
    
    return Cost


#%%


def QAOA_gamma_block(gamma_beta, gamma, C, Dmax):
    
    '''
    
    Function that takes in the parameter gamma and applies
    a single layer of the gamma block within a single QAOA layer.
    
    Input::
        
        gamma_beta  : An existing QAOA state upon which one applies the Gamma layer
        gamma       : The free parameter that needs to be optimized
        C           : The adjacency matrix describing the MaxCut problem instance.
        Dmax        : The maximum bond-dimension limit imposed on the MPSs
        
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


for r in Instance_list:
    
    C = np.load(location+str(q)+tag+str(r)+'/C_Q'+str(q)+tag+str(r)+'.npy')
    n = len(C)
    
    for Dmax in DList:
        
        Cost_mps = np.zeros([N,N])
        
        for i in range(N):
            
            print('\nQ'+str(n)+'R'+str(r)+', D = '+str(Dmax)+', i = '+str(i)+'\n')
            
            plus = tn.FiniteMPS([np.array([[[1/np.sqrt(2)],
                                               [1/np.sqrt(2)]]],
                                             dtype = np.complex128) for x in range(n)])
            
            Fail = True
            attempt = 1
            
            while (Fail):
                
                try:
                        
                    gamma_st = QAOA_gamma_block(plus, Gamma[i], C, Dmax)
                    
                    Fail = False
                    
                except:
                    
                    print('\nSVD Error!! for D = '+str(Dmax)+
                          ', and attempt = '+str(attempt)+'!\n')
                    
                    attempt += 1
                    Gamma[i] = np.round(Gamma[i], decimals=(11-attempt))
                    
                    if (attempt > 10):
                        
                        print('Stopping Repetition at attempt = ',attempt,'\n')
                        raise
            
            for j in range(N):
                
                gamma_beta = QAOA_beta_block(gamma_st, Beta[j])
                
                Cost_mps[i,j] = get_Cost_mpsIX(gamma_beta)
                    
                del gamma_beta
            
            del gamma_st
            
        np.save(location+str(n)+'R'+str(r)+'/N'+str(N)+'D'+str(Dmax)+'.npy',Cost_mps)
