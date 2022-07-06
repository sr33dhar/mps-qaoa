#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:22:49 2020

@author: ph30n1x
"""

import tensornetwork as tn
import numpy as np

'''

Set of codes to calculate the inner prodict of two tensors.

'''

class Expectation():
    
    def exp_Node(state1,state2):
        
        '''
        
        Function to obtain the final contraction of all edges from two tensors state1 and state2
        
        Input::
            
            state1: A tensor of rank r1
            state2: A second tensor of rank r2
            
        Output::
            
            Gives an error if r1 != r2
            
            If r1 = r2, contracts the corresponding edges to finally give a scalar as output
        
        '''
        
        n = len(state1.get_all_dangling())
        
        state1 = tn.conj(state1)
        
        if (n != len(state2.get_all_dangling())):
            
            print("\n\nError!! The number of dangling edges do not match!\n")
            print("length of state 1 is ",n,' whereas the length of state 2 is ',len(state2.get_all_dangling()))
            return None
            
        else:
            
            for i in range(n):
                
                state1[i]^state2[i]
                
            expt = (state1@state2).tensor
            expt = expt.item()
           
        return expt
    
    
    def exp_MPS(state1,state2):
        
        '''
        
        Function to obtain the inner product of two MPSs state1 and state2
        
        Here, all the connections are made by hand to form the full network.
        The network is finally contracted at once using the greedy function inbuilt in 
        the tensornetwork package.
        
        Input::
            
            state1: An MPS with n1 real indices
            state2: An MPS with n2 real indices
            
        Output::
            
            Gives an error if n1 != n2
            
            If n1 = n2, contracts the corresponding edges to finally
            give the inner product scalar as output
        
        '''
        
        S1 = state1.tensors
        S2 = state2.tensors
        n = len(S1)
        
        if (n != len(S2)):
            
            print("\n\nError!! The number of dangling edges do not match!\n")
            print("length of state 1 is ",n,' whereas the length of state 2 is ',len(S2))
            return None
            
        else:
            
            S1 = [tn.Node(np.conj(S1[x])) for x in range(n)]
            S2 = [tn.Node(S2[x]) for x in range(n)]
            
            for i in range(n):
                
                S1[i][1]^S2[i][1]
                
                if (i == (n-1)):
                    
                    S1[i][2]^S1[0][0]
                    S2[i][2]^S2[0][0]
                    
                else:
                    
                    S1[i][2]^S1[i+1][0]
                    S2[i][2]^S2[i+1][0]
                    
            
            expt = tn.contractors.greedy((S1 + S2))
            
            expt = expt.tensor
            expt = expt.item()
           
            return expt        
    
    
    def exp_MPS_II(state1,state2):
        
        '''
        
        Function to obtain the inner product of two MPSs state1 and state2
        
        A more optimized tensor contraction implementation as recommended by the 
        package creators is implemented here.
        
        Input::
            
            state1: An MPS with n1 real indices
            state2: An MPS with n2 real indices
            
        Output::
            
            Gives an error if n1 != n2
            
            If n1 = n2, contracts the corresponding edges to finally
            give the inner product scalar as output
        
        '''
        
        nodes1 = [tn.Node(np.conj(tensor)) for i, tensor in enumerate(state1.tensors)]
        nodes2 = [tn.Node(tensor) for i, tensor in enumerate(state2.tensors)]
          
        n = len(nodes1)
            
        if (n != len(nodes2)):
            
            print("\n\nError!! The number of dangling edges do not match!\n")
            print("length of state 1 is ",n,' whereas the length of state 2 is ',len(nodes2))
            return None
            
        else:
            
            nodes1[0][0] ^ nodes2[0][0]
            nodes1[-1][2] ^ nodes2[-1][2]
            
            [nodes1[k][2] ^ nodes1[k+1][0] for k in range(n-1)]
            [nodes2[k][2] ^ nodes2[k+1][0] for k in range(n-1)]
            [nodes1[k][1] ^ nodes2[k][1] for k in range(n)]
            
            expt = nodes1[0] @ nodes2[0]
            
            for i in range(1, len(nodes1)):
                expt = expt @ nodes1[i] @ nodes2[i]
            
            expt = expt.tensor
            expt = expt.item()
            
            return expt
            
            
             
