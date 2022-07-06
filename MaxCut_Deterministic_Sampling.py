#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 18:56:18 2021

@author: Rishi Sreedhar ( https://orcid.org/0000-0002-7648-4908 )
"""

'''

Code created to extract the solutions using the deterministic sampling method using MPS techniques on the QAOA.
This code produces the data presented in Section IV: QAOA PERFORMANCES WITH RESTRICTED ENTANGLEMENT, 
Subsection A: Performances for MaxCut, Figure 4. (Note that the data from this code has been 
averaged over all the instances and plotted using a log scale in Figure 4, separately.)

Circuit depths studied 1 <= p <= Plim = 100

Bond-dimension Limit = Dlim = int(2**np.floor(q/2)) if (q < 16) else 100

System sizes studied n = {14, 40, 60}

Problem type: MaxCut applied to Erdos Renyi graphs with an edge probability of 0.5

Input Needed::
    
    1.) Optimum parameters Gamma_Opt and Beta_Opt
    2.) The adjacency matrix C describing the problem
    3.) Solution states to the corresponding problem to calculate approximation ratios
    4.) Instance_list: The list of instances to be studied.

Output Generated::
    
    1.) Sol_Pred_String_DetSamp : A [Plim x Dlim] array of the deterministic samlpe isolated from
                                  each MPS state corresponding to every (p,D) pairs. 
                                  The binary values have been converted to equivalent decimal format for ease of storage. 
                                  This array is saved in the respective instance folder by default.
        
    2.) AppRatio_DetSample      : A [Plim x Dlim] array with the Approximation Ratio corresponding to the
                                  deterministic sample isolated from each MPS state corresponding to every (p,D) pairs. 
                                  This array is saved in the respective instance folder by default.

'''


import os
import time
import tensornetwork as tn
from Gates import Gates as g
import numpy as np

tn.set_default_backend("numpy")

#%%


##############################
## Parameter Initialization ##
##############################

q = 12 # number of Qubits

l1 = int(np.floor((q)/2))   # Paramteres required for the SWAP network
l2 = int(np.floor((q-1)/2)) # Paramteres required for the SWAP network


folder_location = os.path.dirname(os.path.realpath(__file__)) # Location of the MPS_QAOA Folder
local_location = '/MxC_Q'+str(q)+'/Q' #Location of files within the MPS_QAOA Folder
location = folder_location + local_location

tag = 'R' # R for Random

D_highest = int(2**np.floor(q/2)) # highest possible bond-dimension for a q-qubit system

Plim = 100 # Circuit depth limit
Dlim = int(2**np.floor(q/2)) if (q < 16) else 100 # setting a bond-dimension limit for large instances
    
Normalize = True # Normalization constraint for the MPS-QAOA states

#### Optimum parameters derived from 12-qubit MaxCut instances!
Gamma_Opt = np.load(folder_location+'/Opt_angels_Q12/MxC_GammaOpt.npy')
Beta_Opt = np.load(folder_location+'/Opt_angels_Q12/MxC_BetaOpt.npy')
       
Instance_list = [0] # The indices of the instances one wishes to run the calculations for. 
                    # Eg: If q = 12, enter [0,42] if the instances of interest are 12R0 and 12R42


#%%


def QAOA_gamma_block(gamma_beta, gamma, C, Dmax):
    
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

def get_kth_DetSamp(gamma_beta,sol):
    
    '''
    Function to obtain the state of the k^th qubit in the deterministic sample.
    
    Input::
        
        gamma_beta : The approximated or exact QAOA state expressed as an MPS
        sol        : The solution predicted till now for the previous k-1 qubits
        
    Output::
        
        s_DetSamp  : The state of the kth qubit in the sampled bit-string
        k0_DetSamp : The probability of obtaining a 0 on kth qubit after projections
        k1_DetSamp : The probability of obtaining a 1 on kth qubit after projections
    
    '''
    
    l = len(sol)
    
    g_b = gamma_beta.tensors
    
    g_bcon = [tn.Node(g_b[x].conj()) for x in range(n)]
    g_b = [tn.Node(g_b[x]) for x in range(n)]
    
    
    for i in range(l):
        
        z = np.array([1.0,0.0])
        z_conj = tn.Node(z.conj())
        z = tn.Node(z)
        
        o = np.array([0.0,1.0])
        o_conj = tn.Node(o.conj())
        o = tn.Node(o)
        
        if (sol[i] == '0'):
            
            g_b[i][1]^z[0]
            g_b[i] = tn.contract_between(g_b[i],z,
                                         output_edge_order = [g_b[i][0],g_b[i][2]])
            
            g_bcon[i][1]^z_conj[0]
            g_bcon[i] = tn.contract_between(g_bcon[i],z_conj,
                                            output_edge_order = [g_bcon[i][0],g_bcon[i][2]])
            
        else:
            
            g_b[i][1]^o[0]
            g_b[i] = tn.contract_between(g_b[i],o,
                                         output_edge_order = [g_b[i][0],g_b[i][2]])
            
            g_bcon[i][1]^o_conj[0]
            g_bcon[i] = tn.contract_between(g_bcon[i],o_conj,
                                            output_edge_order = [g_bcon[i][0],g_bcon[i][2]])
        
        
        if (i != (l-1)):
            
            g_b[i][1]^g_b[i+1][0]
            g_bcon[i][1]^g_bcon[i+1][0]
        
    
    if (l != 0):
        
        GL_proj = tn.contractors.greedy(g_b[:l],
                                       output_edge_order = [g_b[0][0],g_b[l-1][1]])
        
        GLcon_proj = tn.contractors.greedy(g_bcon[:l],
                                       output_edge_order = [g_bcon[0][0],g_bcon[l-1][1]])
        
    else:
        
        GL_proj = tn.Node(np.array([1.0]).reshape((1,1)))
        GLcon_proj = tn.Node(np.array([1.0]).reshape((1,1)))
            
        
    for i in range(l+1,n):
        
        if (i != (n-1)):
            
            g_b[i][2]^g_b[i+1][0]
            g_bcon[i][2]^g_bcon[i+1][0]
        
        g_bcon[i][1]^g_b[i][1]
        
    
    if (l != (n-1)):
        
        GR_proj = tn.contractors.greedy((g_b[(l+1):] + g_bcon[(l+1):]), output_edge_order 
                                        = [g_bcon[l+1][0],g_b[l+1][0],g_bcon[-1][2],g_b[-1][2]])
        
    else:
        
        GR_proj = tn.Node(np.array([1.0]).reshape((1,1,1,1)))
    
    GLcon_proj[1]^g_bcon[l][0]
    GL_proj[1]^g_b[l][0]
    
    GLcon_proj[0]^GR_proj[2]
    GL_proj[0]^GR_proj[3]
    
    g_bcon[l][2]^GR_proj[0]
    g_b[l][2]^GR_proj[1]
    
    K = tn.contractors.greedy([GLcon_proj, GL_proj, g_bcon[l], g_b[l], GR_proj],
                              output_edge_order = [g_bcon[l][1],g_b[l][1]])
    
    k0_DetSamp = abs(K.tensor[0,0])
    k1_DetSamp = abs(K.tensor[1,1])    
    
    s_DetSamp = ''
    
    if ( abs(k0_DetSamp) > abs(k1_DetSamp)):
        
        s_DetSamp = '0'
    
    else:
        
        s_DetSamp = '1'
    
    return s_DetSamp,k0_DetSamp,k1_DetSamp
    

#%%  

def bin2dec(binary_string):
    
    '''
    
    Function that returns the equivalent decimal number corresponding to a binary string
    
    Input::
        
        binary_string : The n-bit binary string that needs to be converted
        
    Output::
        
        decimal       : The decimal number corresponding to the input binary string
    
    '''
    
    B = list(binary_string)
    B = [int(val) for idx, val in enumerate(B)]
    
    decimal = sum(val*(2**idx) for idx, val in enumerate(reversed(B)))
    
    return decimal

    
#%%


def get_EigenEnergy(basis_str, C):
    
    '''
    
    Function that returns the Eigen Energy corresponding to a binary string (basis state in Z basis)
    
    Input::
        
        binary_string : The n-bit binary string representing a basis state in Z basis
        C             : The adjacency matrix encoding the MaxCut problem and the cost Hamiltonian H_C
        
    Output::
        
        EigenEnergy   : The Eigen Energy of the input state corresponding to the Cost Hamiltonian H_C
    
    '''    
    
    B = [1 if (basis_str[x] == '1') else -1 for x in range(len(basis_str))]
    
    EigenEnergy = 0
    
    for i in range(n):
        
        for j in range(i+1,n):
            
            EigenEnergy -= 0.5*C[i,j]*(1 - B[i]*B[j])
    
    return EigenEnergy


#%%


for s in Instance_list:
    
    ####### Data Extraction #######
    
    C = np.load(location+str(q)+tag+str(s)+'/C_Q'+str(q)+tag+str(s)+'.npy')
    n = len(C)
    
    DetSample_Energy = 0

    ####### Ground State and Eigen Energy Extraction #######
    
    if (os.path.isfile(location+str(n)+tag+str(s)+'/Sol_str_EL0_1.npy')):
        
        GroundState_Sol_list = []
        for j in range(1,1000):
            
            try:
                
                S = np.load(location+str(n)+tag+str(s)+'/Sol_str_EL0_'+str(j)+'.npy')
                
                S = str(S)
                GroundState_Sol_list.append(S)
                
            except:
                continue
                
        
        GroundState_Energy = get_EigenEnergy(GroundState_Sol_list[0], C)
    
    elif (os.path.isfile(location+str(n)+tag+str(s)+'/Sol_Q'+str(n)+tag+str(s)+'.txt')):
        
        file = open(location+str(n)+tag+str(s)+'/Sol_Q'+str(n)+tag+str(s)+'.txt')
        GroundState_Energy = -1*int(file.readline())
            
    else:
        
        GroundState_Energy = n*(n-1)*0.5
        
    ####### End All Energy State Extraction #######
    
    AppRatio_DetSample = np.zeros([Plim,Dlim])
    Sol_Pred_String_DetSample = np.zeros([Plim,Dlim]) 
    
    ######################### State Preparation #########################
    
    for Dmax in range(1,Dlim+1):
        
        GB_pCpy = tn.FiniteMPS([np.array([[[1/np.sqrt(2)],
                                           [1/np.sqrt(2)]]],
                                         dtype = np.complex128) for x in range(n)])
        
        for p in range(1,Plim+1):
            
            attempt = 1
            Fail = True
            
            while (Fail):
                
                try:
                    
                    Sol_detsamp_pred = ''
                    
                    GB_p = tn.FiniteMPS(GB_pCpy.tensors, canonicalize=True)
                    
                    GB_p = QAOA_gamma_block(GB_p, Gamma_Opt[p-1], C, Dmax)
                    GB_p = QAOA_beta_block(GB_p, Beta_Opt[p-1])
                    
                    GB_p = tn.FiniteMPS(GB_p.tensors, canonicalize = True)
                    
                    GB_pCpy = tn.FiniteMPS(GB_p.tensors, canonicalize = True)
                    
                    GB = tn.FiniteMPS(GB_p.tensors, center_position = 0, canonicalize = True)
                    
                    ######################### End State Preparation #########################
                    
                    ########## Starting Deterministic Sampling method ##########
                    
                    for i in range(n):
                        
                        s_DetSamp, k0_DetSamp,k1_DetSamp = get_kth_DetSamp(GB,Sol_detsamp_pred)
                        Sol_detsamp_pred += s_DetSamp
                        
                    print('DetSamp Solution = ',Sol_detsamp_pred)
                    
                    ########## Ending Deterministic Sampling method ##########
                    
                    ########## Calculate Approximation Ratio ##########
                    
                    DetSample_decimal = bin2dec(Sol_detsamp_pred)
                    
                    DetSample_Energy = get_EigenEnergy(Sol_detsamp_pred, C)
                    
                    AppRatio_DetSample[p-1,Dmax-1] = DetSample_Energy/GroundState_Energy
                    
                    ########## End Calculate Approximation Ratio ##########
                    
                    
                    ########## Checking Solution for Deterministic Sampling method ##########
                    
                    if (AppRatio_DetSample[p-1,Dmax-1] == 1):
                        
                        print('DetSamp SUCCESS for '+str(q)+tag+str(s)+
                              ' for D = '+str(Dmax)+' and p = '+str(p)+'!!!!\n')
                        
                    else:
                        
                        print('DetSamp Not working for '+str(q)+tag+str(s)+
                              ' for D = '+str(Dmax)+' and p = '+str(p)+'!')
                        
                        print('Approximation Ratio = ',AppRatio_DetSample[p-1,Dmax-1],'\n')
                    
                    ########## End Check for DetSamp method ##########                    
                    
                    ########## Storing Predicted Strings ##########
                    
                    Sol_Pred_String_DetSample[p-1,Dmax-1] = DetSample_decimal
                    
                    ########## End Storing Predicted Strings ##########
                    
                    Fail = False
                    
                except:
                    
                    print('\nSVD Error!! for D = '+str(Dmax)+', p = '+str(p)+
                          ', and attempt = '+str(attempt)+'!\n')
                    
                    time.sleep(2.5)
                    
                    if (attempt > 10):
                        
                        print('Stopping Repetition at attempt = ',attempt,'\n')
                        Fail = False
                        
                        AppRatio_DetSample[p-1,Dmax-1] = -0.1
                        Sol_Pred_String_DetSample[p-1,Dmax-1] = -1
                    
                    attempt += 1
                    
                    Gamma_Opt[p-1] = np.round(Gamma_Opt[p-1], decimals=(11-attempt))
                    Beta_Opt[p-1] = np.round(Beta_Opt[p-1], decimals=(11-attempt))
                    
           
        ########## Save Deterministic Sampling Results ###############
        
        np.save(location+str(q)+tag+str(s)+'/AppRatio_MC_DetSamp_p'+str(Plim)
                +'_'+str(q)+tag+str(s)+'.npy',AppRatio_DetSample)
        
        np.save(location+str(q)+tag+str(s)+'/SolString_MC_DetSamp_p'+str(Plim)
                +'_'+str(q)+tag+str(s)+'.npy',Sol_Pred_String_DetSample)
        
        ########## End Saving Deterministic Sampling Results ##########
        