#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:44:07 2020

@author: ph30n1x
"""

'''

Code defining all the required quantum gates as tensors

'''


import numpy as np
import math
import cmath

class Gates():
    
    def get_X():
        
        x = np.array([[0.0,1.0],
                      [1.0,0.0]], dtype = np.complex128)
        return x
    
    def get_Rx(theta=math.pi):
        
        if (theta == math.pi):
            Rx = np.array([[0.0,1.0],
                           [1.0,0.0]], dtype = np.complex128)
            
        else:
            
            Rx = np.array([[math.cos(theta*0.5),-1.0j*math.sin(theta*0.5)],
                           [-1.0j*math.sin(theta*0.5),math.cos(theta*0.5)]],
                          dtype = np.complex128)
            
        return Rx
            
    def get_Y():
        
        y = np.array([[0.0,-1.0j],
                      [1.0j,0.0]], dtype = np.complex128)
        return y
                
    def get_Ry(theta=math.pi):
        
        if (theta == math.pi):
            Ry = np.array([[0.0,-1.0j],
                           [1.0j,0.0]], dtype = np.complex128)
            
        else:
            
            Ry = np.array([[math.cos(theta*0.5),-1.0*math.sin(theta*0.5)],
                           [1.0*math.sin(theta*0.5),math.cos(theta*0.5)]],
                          dtype = np.complex128)
            
        return Ry
    
    def get_Z():
        
        z = np.array([[1.0,0.0],
                      [0.0,-1.0]], dtype = np.complex128)
        
        return z
            
    def get_Rz(theta=math.pi):
        
        if (theta == math.pi):
            Rz = np.array([[1.0,0.0],
                           [0.0,-1.0]], dtype = np.complex128)
            
        else:
            
            Rz = np.array([[cmath.exp(-1.0j*theta*0.5),0.0],
                           [0.0,cmath.exp(1.0j*theta*0.5)]],
                          dtype = np.complex128)
            
        return Rz
    
    def get_I():
        
        i = np.eye(2, dtype = np.complex128)
        return i
    
    def get_H():
        
        h = np.array([[1.0,1.0],
                      [1.0,-1.0]], dtype = np.complex128)/np.sqrt(2)
        
        return h
    
    def get_CNOT():
        
        cnot = np.array([[1.0,0.0,0.0,0.0],
                         [0.0,1.0,0.0,0.0],
                         [0.0,0.0,0.0,1.0],
                         [0.0,0.0,1.0,0.0]], dtype = np.complex128)
        
        cnot = cnot.reshape((2,2,2,2))
        
        return cnot
    
    def get_CZ():
        
        cz = np.array([[1.0,0.0,0.0,0.0],
                       [0.0,1.0,0.0,0.0],
                       [0.0,0.0,1.0,0.0],
                       [0.0,0.0,0.0,-1.0]], dtype = np.complex128)
        
        cz = cz.reshape((2,2,2,2))
        
        return cz
    
    def get_CY():
        
        cy = np.array([[1.0,0.0,0.0,0.0],
                       [0.0,1.0,0.0,0.0],
                       [0.0,0.0,0.0,-1.0j],
                       [0.0,0.0,1.0j,0.0]], dtype = np.complex128)
        
        cy = cy.reshape((2,2,2,2))
        
        return cy
    
    def get_SWAP():
        
        swap = np.array([[1., 0., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 1.]], dtype = np.complex128)
        
        swap = swap.reshape((2,2,2,2))
        
        return swap
    
    def get_Jij(gamma,jij):
        
        Jij = np.array([[np.exp(-1j*gamma*jij), 0., 0., 0.],
                        [0., np.exp(+1j*gamma*jij), 0., 0.],
                        [0., 0., np.exp(+1j*gamma*jij), 0.],
                        [0., 0., 0., np.exp(-1j*gamma*jij)]], dtype = np.complex128)
        
        Jij = Jij.reshape((2,2,2,2))
        
        return Jij

    def get_Cij(gamma,cij):
        
        Cij = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, np.exp(1j*gamma*cij), 0.0, 0.0],
                        [0.0, 0.0, np.exp(1j*gamma*cij), 0.0],
                        [0.0, 0.0, 0.0, 1.0]], dtype = np.complex128)
        
        Cij = Cij.reshape((2,2,2,2))
        
        return Cij


    def get_fSim(theta,phi):
        
        fSim = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, np.cos(theta), -1j*np.sin(theta), 0.0],
                        [0.0, -1j*np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 0.0, np.exp(-1j*phi)]], dtype = np.complex128)
        
        fSim = fSim.reshape((2,2,2,2))
        
        return fSim


