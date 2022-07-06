#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:06:37 2022

Code created to calculate how the fidelity of QAOA states vary as a function of 
bond-dimension D and circuit depth p for different MaxCut and EC3 instances.
This code generates the fidelity data used in Figure 6 of Subsection C: 
Entanglement in QAOA in Section IV: QAOA PERFORMANCES WITH RESTRICTED ENTANGLEMENT.


Circuit depths studied 1 <= p <= Plim = 100

Bond-dimension Limit = Dlim = int(2**np.floor(q/2)) if (q < 16) else 100

System sizes studied n = {12, 14}

Problem type: 
    1.) MaxCut applied to randomly generated Erdos Renyi graphs with edge probability = 0.5 
    2.) Randomly generated Exact Cover 3 Problem instances.

Input Needed::
    
    1.) Optimum parameters Gamma_Opt and Beta_Opt
    2.) Problem_type that specifies if we are looking at MaxCut or EC3 problems    
    3.) The Interaction matrix J describing the EC3 problem
        The self-energy matrix h describing the EC3 problem
        or
        The adjacency matrix C describing the MaxCut problem
    4.) Instance_list: The list of instances to be studied.

Output Generated::
    
    1.) GB_fidelity: A [Plim x Dlim] array with the fidelity of each MPS state corresponding to 
                     every (p,D) pairs. This array is saved in the respective instance folder by default.
        
@author: Rishi Sreedhar ( https://orcid.org/0000-0002-7648-4908 )
"""

import os
import tensornetwork as tn
from Gates import Gates as g
import numpy as np
from Expectation import Expectation as exp

tn.set_default_backend("numpy")

#%%


##############################
## Parameter Initialization ##
##############################

q = 14 # number of Qubits

l1 = int(np.floor((q)/2))   # Paramteres required for the SWAP network
l2 = int(np.floor((q-1)/2)) # Paramteres required for the SWAP network

# Problem_type = 'MaxCut'
Problem_type = 'EC3'


folder_location = os.path.dirname(os.path.realpath(__file__)) # Location of the MPS_QAOA Folder 

if (Problem_type == 'MaxCut'): #Location of files within the MPS_QAOA Folder
    
    local_location = '/MxC_Q'+str(q)+'/Q'
    #### Optimum parameters derived from 12-qubit MaxCut instances!
    Gamma_Opt = np.load(folder_location+'/Opt_angels_Q12/MxC_GammaOpt.npy')
    Beta_Opt = np.load(folder_location+'/Opt_angels_Q12/MxC_BetaOpt.npy')
    
elif (Problem_type == 'EC3'):
    
    local_location = '/EC3_Q'+str(q)+'/Q'
    #### Optimum parameters derived from 12-qubit MaxCut instances!
    Gamma_Opt = np.load(folder_location+'/Opt_angels_Q12/EC3_GammaOpt.npy')
    Beta_Opt = np.load(folder_location+'/Opt_angels_Q12/EC3_BetaOpt.npy')

else:
     
    print("Invalid Problem_type!! Please enter either 'MaxCut' or 'EC3' exactly! \n ")
    raise

    
location = folder_location + local_location

tag = 'R' # R for Random

D_highest = int(2**np.floor(q/2)) # highest possible bond-dimension for a q-qubit system

Plim = 100 # Circuit depth limit
Dlim = int(2**np.floor(q/2)) if (q < 16) else 100 # setting a bond-dimension limit for large instances
    
Normalize = True # Normalization constraint for the MPS-QAOA states
       
Instance_list = [0] # The indices of the instances one wishes to run the calculations for. 
                    # Eg: If q = 12, enter [0,42] if the instances of interest are 12R0 and 12R42


#%%


def QAOA_MaxCut_gamma_block(gamma_beta, gamma, C, Dmax):
    
    '''
    
    Function that takes in the parameter gamma and applies
    a single layer of the gamma block within a single QAOA layer for MaxCut problems.
    
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


def QAOA_EC3_gamma_block(gamma_beta, gamma, J, h, Dmax):
    
    '''
    
    Function that takes in the parameter gamma and applies
    a single layer of the gamma block within a single QAOA layer for EC3 problems.
    
    Input::
        
        gamma_beta  : An existing QAOA state upon which one applies the Gamma layer
        gamma       : The free parameter that needs to be optimized
        J           : The Interaction Energies matrix describing the EC3 problem instance.
        h           : The self-energy matrix describing the EC3 problem instance.
        Dmax        : The maximum bond-dimension limit imposed on the MPSs
        
    Output::
        
        gamma_beta  : The QAOA state with an additional Cost layer of the QAOA added to it
    
    '''
    
    gamma_beta = tn.FiniteMPS(gamma_beta.tensors, canonicalize=False)
    gamma_beta.canonicalize(normalize=Normalize)
    
    Q_ord = np.linspace(start = 0, stop = (n-1), num = n, dtype = int)
    
    SWAP = g.get_SWAP()
    
    for i in range(n):
    
        Rz = g.get_Rz(2*gamma*h[i])
        gamma_beta.apply_one_site_gate(Rz, i)
    
    
    for i in range(n):
        
        if (i < (n-1)):
            
            for k in range(n-1):
                
                if (Q_ord[k] < Q_ord[k+1]):
                    
                    Jij = g.get_Jij(gamma, J[Q_ord[k]][Q_ord[k+1]])
                    
                    gamma_beta.position(site=k, normalize=Normalize)
                    gamma_beta.apply_two_site_gate(Jij, site1 = k, site2 = (k+1),
                                               max_singular_values=Dmax, center_position=k)
                    
        
        if (i%2 == 0):
            
            for s in range(l1):
                
                Q_ord[2*s],Q_ord[2*s+1] = Q_ord[2*s+1],Q_ord[2*s]
                
                gamma_beta.position(site=(2*s), normalize=Normalize)
                gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s), site2 = (2*s+1),
                                               max_singular_values=Dmax, center_position=(2*s))
                
            
        else:
            
            for s in range(l2):
                
                Q_ord[2*s+1],Q_ord[2*s+2] = Q_ord[2*s+2],Q_ord[2*s+1]
                
                gamma_beta.position(site=(2*s+1), normalize=Normalize)
                gamma_beta.apply_two_site_gate(SWAP, site1 = (2*s+1), site2 = (2*s+2),
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


for s in Instance_list:
    
    ####### Data Extraction #######
    
    if (Problem_type == 'MaxCut'): #Location of files within the MPS_QAOA Folder
        
        C = np.load(location+str(q)+tag+str(s)+'/C_Q'+str(q)+tag+str(s)+'.npy')
        n = len(C)
        
    elif (Problem_type == 'EC3'):
        
        J = np.load(location+str(q)+tag+str(s)+'/J_Q'+str(q)+tag+str(s)+'.npy')
        h = np.load(location+str(q)+tag+str(s)+'/h_Q'+str(q)+tag+str(s)+'.npy')
        n = len(h)
        
    else:
         
        print("Invalid Problem_type!! Please enter either 'MaxCut' or 'EC3' exactly! \n ")
        raise
        
    
    D_start = 1
    
    GB_fidelity = np.zeros([Plim,Dlim])
    
    GB_Dlim_list = [None]*Plim # List of the full bond-dimension states for each circuit depth p
    
    GB_D_pCpy = tn.FiniteMPS([np.array([[[1/np.sqrt(2)],
                                       [1/np.sqrt(2)]]],
                                     dtype = np.complex128) for x in range(n)])
    
    for p in range(1,Plim+1):
    
        attempt = 1
        Fail = True
        
        while (Fail):
            
            try:
                
                ###################### For full Bond-dimension D = Dlim ######################
                
                GB_D_p = tn.FiniteMPS(GB_D_pCpy.tensors, canonicalize=True)
                
                if (Problem_type == 'MaxCut'): #Location of files within the MPS_QAOA Folder
                    
                    GB_D_p = QAOA_MaxCut_gamma_block(GB_D_p, Gamma_Opt[p-1], C, Dlim)
                    
                elif (Problem_type == 'EC3'):
                    
                    GB_D_p = QAOA_EC3_gamma_block(GB_D_p, Gamma_Opt[p-1], J, h, Dlim)
                    
                else:
                     
                    print("Invalid Problem_type!! Please enter either 'MaxCut' or 'EC3' exactly! \n ")
                    raise
                
                
                GB_D_p = QAOA_beta_block(GB_D_p, Beta_Opt[p-1])
                
                GB_D_p = tn.FiniteMPS(GB_D_p.tensors, canonicalize = True)
                GB_D_p.canonicalize(normalize=True)
                
                GB_D_pCpy = tn.FiniteMPS(GB_D_p.tensors, canonicalize = True)
                
                GB_Dlim_list[p-1] = GB_D_p
                
                print('\n'+Problem_type+': Q'+str(q)+'R'+str(s)+': D'+str(Dlim)+' P'+str(p)+' \n')
                
                Fail = False
                    
            except:
                
                print('\nSVD Error!! for D = '+str(Dlim)+', p = '+str(p)+
                      ', and attempt = '+str(attempt)+'!\n')
                
                if (attempt > 10):
                    
                    print('Stopping Repetition at attempt = ',attempt,'\n')
                    Fail = False
                    
                attempt += 1
                
                Gamma_Opt[p-1] = np.round(Gamma_Opt[p-1], decimals=(11-attempt))
                Beta_Opt[p-1] = np.round(Beta_Opt[p-1], decimals=(11-attempt))
                
                raise
    
    
    ######################### State Preparation #########################
    
    
    for Dmax in range(D_start,Dlim+1):
        
        GB_pCpy = tn.FiniteMPS([np.array([[[1/np.sqrt(2)],
                                           [1/np.sqrt(2)]]],
                                         dtype = np.complex128) for x in range(n)])

        for p in range(1,Plim+1):
            
            attempt = 1
            Fail = True
            
            while (Fail):
                
                try:
                    
                    # ###################### For full Bond-dimension D = Dlim ######################
                    
                    GB_D_p = GB_Dlim_list[p-1]
                    
                    ###################### For Bond-dimension D < Dlim ######################
                    
                    
                    GB_p = tn.FiniteMPS(GB_pCpy.tensors, canonicalize=True)
                    
                    if (Problem_type == 'MaxCut'): #Location of files within the MPS_QAOA Folder
                        
                        GB_D_p = QAOA_MaxCut_gamma_block(GB_D_p, Gamma_Opt[p-1], C, Dlim)
                        
                    elif (Problem_type == 'EC3'):
                        
                        GB_D_p = QAOA_EC3_gamma_block(GB_D_p, Gamma_Opt[p-1], J, h, Dlim)
                        
                    else:
                         
                        print("Invalid Problem_type!! Please enter either 'MaxCut' or 'EC3' exactly! \n ")
                        raise
                        
                    GB_p = QAOA_beta_block(GB_p, Beta_Opt[p-1])
                    
                    GB_p = tn.FiniteMPS(GB_p.tensors, canonicalize = True)
                    GB_p.canonicalize(normalize=True)
                    
                    GB_pCpy = tn.FiniteMPS(GB_p.tensors, canonicalize = True)
                    
                    Fail = False
                    
                    e_val = exp.exp_MPS_II(GB_p, GB_D_p)
                    GB_fidelity[p-1,Dmax-1] = abs(e_val)**2
                    
                    print('\n '+Problem_type+': Q'+str(q)+'R'+str(s)+': D'+str(Dmax)+' P'+str(p)
                          +': Fidelity = '+str(GB_fidelity[p-1,Dmax-1])+' \n')
                    
                except:
                    
                    print('\nSVD Error!! for D = '+str(Dmax)+', p = '+str(p)+
                          ', and attempt = '+str(attempt)+'!\n')
                    
                    if (attempt > 10):
                        
                        print('Stopping Repetition at attempt = ',attempt,'\n')
                        Fail = False
                        
                    attempt += 1
                    
                    Gamma_Opt[p-1] = np.round(Gamma_Opt[p-1], decimals=(11-attempt))
                    Beta_Opt[p-1] = np.round(Beta_Opt[p-1], decimals=(11-attempt))
                    
                    raise
        
        ########## Save Fidelity Data ##########
        
        np.save(location+str(q)+tag+str(s)+'/Fidelity_'+Problem_type+'_p'
                +str(Plim)+'_'+str(q)+tag+str(s)+'.npy',GB_fidelity)
                
        ########## End Save Fidelity Data ##########
        
