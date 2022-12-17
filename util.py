import os
import pandas as pd
import numpy as np
import pylab as plt
from scipy import signal
from collections import deque
from scipy.interpolate import interp1d
import shaiutil
from scipy.special import erfc
import copy

def affenize(Q):
    '''
        Function:
            Q -> [Q 1]
        Input:
            Q - numpy array (m x n)
        Output:
            Q_ - numpy array (m x n+1)
    '''
    m, n = np.shape(Q)
    Q_ = np.ones((m,n+1))
    Q_[:,:-1] = Q
    return Q_

def set_mapper(S, Z, o, M=None):
    '''
        Function:
            Assumes that S is monotonically increasing
            Y_i+o = A_o * X_i
        Input:
            S - numpy array (m x 1)
            Z - numpy array (m x n)
            o - value (1 x 1)
        Output:
            Y - numpy array (l x n)
            X - numpy array (l x n)
    '''
    if np.shape(M) == ():
        M = range(np.shape(Z)[0])

    n = np.shape(Z)[1]
    I_x = deque()

    for i in M:
        s_i = S[i] + o
        i_plus = np.where(s_i < S)[0]
        if len(i_plus) > 0:
            I_x.append(i)
    l, = np.shape(I_x)
    X = np.zeros((l, n))
    Y = np.zeros((l, n))
    f_y = interp1d(S, Z, axis=0)
    for j, i_x in enumerate(I_x):
        X[j,:] = Z[i_x, :]
        Y[j,:] = f_y(S[i_x]+o)
    return Y, X, np.array(I_x)

def set_mapper_2(S, Z, o):
    '''
        Function:
            Assumes that S is monotonically increasing
            Y_i+o = A_o * X_i
        Input:
            S - numpy array (m x 1)
            Z - numpy array (m x n)
            o - value (1 x 1)
        Output:
            Y - numpy array (l x n)
            X - numpy array (l x n)
    '''
    m, n = np.shape(Z)
    I_x = deque()
    for i in range(m):
        s_i = S[i] + o
        i_plus = np.where(S >= s_i)[0]
        if len(i_plus) > 0:
            I_x.append(i)
    l, = np.shape(I_x)
    X = np.zeros((l, n))
    Y = np.zeros((l, n))
    f_y = interp1d(S, Z, axis=0)
    for j, i_x in enumerate(I_x):
        X[j,:] = Z[i_x, :]
        Y[j,:] = f_y(S[i_x]+o)
    return Y, X, np.array(I_x)

'''
def bootstrapPhaseResiduals(Yh, Y, phi, samples):
    ORDER = 3
    m, n = np.shape(Y)
    np.random.seed(2)
    fsList = []

    #for i in range(samples):
    for i in [0, 1]:
        I = np.random.choice(range(m), m)
        fs = shaiutil.FourierSeries()
        fsList.append(fs.fit(ORDER, phi[np.newaxis,:], (Yh[I]-Y[I]).T))

    #fs = shaiutil.FourierSeries()
    #fs.coef = np.mean([fs.coef for fs in fsList], axis=0)
    #fs.m = np.mean([fs.m for fs in fsList], axis=0)
    #fs.om = np.mean([fs.om for fs in fsList], axis=0)
'''

def bootstrap_residuals(Yh, Y, s):
    '''
        Function:
            Finds r = E[Y-Yh] quantity and then bootstraps
        Input:
            Yh - numpy array (m x n)
            Y - numpy array (m x n)
            s - int
        Output:
            R - numpy array (s x n)
    '''
    m, n = np.shape(Y)
    R = np.zeros((s, n))
    for i in range(s):
        I = np.random.choice(range(m), m)
        R[i,:] = np.mean(np.abs(Y[I] - Yh[I]), axis=0)
    return R

def bootstrap_rrv(Yh, Y, s):
    '''
        Function:
            Finds r = Var[Y-Yh]/Var[Y] quantity and then bootstraps
        Input:
            Yh - numpy array (m x n)
            Y - numpy array (m x n)
            s - int
        Output:
            R - numpy array (s x n)
    '''
    m, n = np.shape(Y)
    RRV = np.zeros((s, n))
    for i in range(s):
        I = np.random.choice(range(m),m)
        RRV[i,:] = np.var(Y[I] - Yh[I], axis=0)/np.var(Y[I], axis=0)
    return RRV

def detrend(X):
    '''
        Function:
            Detrends the columns of a matrix
        Input:
            X - numpy array (m x n)
        Output:
            X - numpy array (m x n)
    '''
    for k in range(np.shape(X)[1]):
        h = X[:,k]
        b, a = signal.butter(2, .01)
        h = signal.filtfilt(b, a, h.T).T
        X[:,k] -= h
    return X

def phase_discounting(phi_0, phi, STD_DEVIATION = 0.5):
    '''
        Function:
            Takes a current phasor, phi_0, and finds its weighted phase
            difference with respect an array, phi.
        Input:
            phi_0 - complex scalar
            phi - complex numpy array (m)
        Output:
            weighted_diff - numpy array (m)
    '''
    phase_diff = phi_0 / phi
    weighted_diff = erfc(np.abs(np.arctan2(phase_diff.imag, phase_diff.real)/(STD_DEVIATION*np.sqrt(2))))
    return weighted_diff

def integrateTorque(T,X,dt,side='Ipsi'):
    '''Inegration of data (T) b/t transition points (X). For gait, send in only ipsilateral states to be integrated.
    Contralateral states are 180 deg out of phase with the transition points, and will give erroniuous impulses.
    For contralateral states, add 1/2 cycle to each transition point
    INPUTS:
        T = (N x 1) array of N time points & S states to be integrated
        X = (M x 1) array of M transition points, at which the integral is reset to zero
        dt = scalar time step (s)
        side = 'Ipsi' OR 'Contra' for the ipsilateral (used to compute transition points) OR
            contalateral (180 deg out of phase) states. These generally correspond to right / left legs
    OUTPUTS:
        I = (N x S) array of the integrated data'''
    # Get dimensions
    N=T.shape
    M=X.shape
    # Option to alter transition points for contralateral limb
    if side=='Contra':
        D=int(round(np.mean(np.diff(X)))/2); # half a cycle, in frames
        if X[0]>D:
            X=X-D
        else:
            X=X+D;
        if M[0]-X[-1]>D:
            X.append(X[-1]+D) # add one more step if space permits
    # Preallocate arrays
    I=np.zeros((N[0],1))
    I_euler=np.zeros((N[0],1))
    # Integrate
    n=1;
    for m in range(0,M[0]): # GO through the Xition points
        while n<X[0]:
            I_euler[n]=I[n-1]+T[n]*dt # Forward Euler
            I[n]=np.trapz(T[0:n],dx=dt,axis=0) # Trapezoidal
            n+=1
        while n<X[m] and n>X[0]: # Integrate up to next transition
            I_euler[n]=I[n-1,:]+T[n]*dt
            I[n]=np.trapz(T[int(X[m-1]):n],dx=dt,axis=0)
            n+=1
        I[int(X[m])]=0.
        n+=1    
    return I

if __name__ == "__main__":
    print
