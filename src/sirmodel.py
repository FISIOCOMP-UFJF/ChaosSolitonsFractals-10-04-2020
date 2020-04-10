#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import factorial as fat
from scipy.optimize import fmin,differential_evolution
import sys
from expmmq import *

# global data
# ref: https://www.worldometers.info/coronavirus/coronavirus-cases/#total-cases
dia = np.array([0,1,2,3,4,5,6,7])
casos = np.array([156653, 169593, 182490, 198238, 218822, 244933, 275597, 305036])
c0,c1 = exp_mmq(dia, casos)

data_italy = {}
data_italy['first_day'] = 32
data_italy['total_population'] = 6.048e7
data_italy['deaths'] = np.loadtxt('../data/death_Italy.txt')
data_italy['infected'] = np.loadtxt('../data/active_Italy.txt')

# global data for korea
data_korea = {}
data_korea['first_day'] = 14
data_korea['total_population'] = 51.47e6
data_korea['deaths'] = np.loadtxt('../data/death_Korea.txt')
data_korea['infected'] = np.loadtxt('../data/active_Korea.txt')

# global data for brazil
data_brazil = {}
data_brazil['first_day'] = 35
data_brazil['total_population'] = 209300000
data_brazil['deaths'] = np.loadtxt('../data/death_Brazil.txt')
data_brazil['infected'] = np.loadtxt('../data/active_Brazil.txt')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# proposed model    
def SIR_PP(P,t,amax,r,ti,tf,m,e,r1,r2):
    """
    Modelo final da NT01
    """
    S,I,R,M = P[0], P[1], P[2], P[3]

    # parametros
    ft = e*c0*np.exp(c1*t)
    
    if t<ti:
        alfa=amax
    elif ti <= t and t<=tf:
        alfa=amax*(1-r)/(ti-tf)*(t-ti) + amax
    else:
        alfa=amax*r

    beta1 = m*r1
    beta2 = (1.0-m)*r2

    dSdt = -alfa*S*I
    dIdt = alfa*S*I + ft- beta1*I - beta2*I
    dRdt = beta2*I
    dMdt = beta1*I
    
    return [dSdt, dIdt, dRdt, dMdt]
    

# model evaluation
def model(x):
    """
    x: model parameters
    x[0]:  a_max
    x[1]:  r
    x[2]:  ti
    x[3]:  tf
    x[4]:  e
    x[5]:  theta
    x[6]:  tau1
    x[7]:  tau2
    x[8]:  tau3
    x[9]:  m
    """

    # time array
    ts = range(len(infected) - first_day)

    # initial conditions
    theta = x[5]
    S0 = data['total_population'] - infected[first_day]/theta
    I0 = infected[first_day]/theta
    R0 = 0
    D0 = deaths[first_day]
    P0 = [S0,I0,R0,D0]

    # SIR_PP function args: amax,r,ti,tf,m,e,r1,r2
    sirargs = ( x[0],x[1],x[2],x[3],x[9],x[4], 1.0/(x[6]+x[7]), 1.0/(x[6] + x[8]) )

    # solve the model
    Ps = odeint(SIR_PP, P0, ts, args=sirargs)

    # getting results
    S = Ps[:,0] # suscetible
    I = Ps[:,1] # infected
    R = Ps[:,2] # recovered
    D = Ps[:,3] # deaths
    
    t1 = x[6]
    t2 = x[7]
    t3 = x[8]
    
    # compute reported cases
    Ir = np.zeros(len(I)); # reported infected
    Rr = np.zeros(len(R)); # reported recovered
    
    ii = find_nearest(ts, (1-theta)*t1)
    if (ts[ii]-(1-theta)*t1) < 0.0: ii = ii+1;
    for i in range(0, ii):Ir[i] = infected[first_day]
    for i in range(ii, len(Ir)):
        j = find_nearest(ts, ts[i]-(1-theta)*t1);
        Ir[i] = theta*I[j]
        Rr[i] = theta*R[j]

    return S,Ir,Rr,D,ts


if __name__ == "__main__":

    # choose country case: Brazil, Korea or Italy
    data = None
    case = sys.argv[1]

    if(case == 'Italy'):
        data = data_italy
        amax,r,ti,delta_t, e,theta, tau_1,tau_2, tau_3, m = np.loadtxt('../data/param_italy.txt');
        opt_leito = False
    elif(case == 'Korea'):
        data = data_korea
        amax,r,ti,delta_t, e,theta, tau_1,tau_2, tau_3, m = np.loadtxt('../data/param_korea.txt');
        opt_leito = False
    elif(case == 'Brazil'):
        data = data_brazil
        amax,r,ti,delta_t, e,theta, tau_1,tau_2, tau_3, m = np.loadtxt('../data/param_brazil.txt');
        opt_leito = False
    else:
        print(f"Coutry {case} not found or not available")
        sys.exit(0)
    
    tf = ti + delta_t
    
    
    # data from literature
    first_day = data['first_day']
    deaths = data['deaths']
    infected = data['infected']

    # model results
    suc,inf,rec,dea,ts = model((amax,r,ti,tf, e,theta, tau_1,tau_2, tau_3, m))

    # plot infected and deaths along time
    plt.figure(0)
    plt.plot(ts, inf, label="Infected")
    plt.plot(ts, infected[first_day:], 'o', color="darkred", label='data: Infected')
    plt.title(case)
    plt.legend()

    plt.figure(1)
    plt.plot(ts, dea, label="Deaths")
    plt.plot(ts, deaths[first_day:], 'o', color="darkred", label='data: Deaths')
    plt.title(case)
    plt.legend()
    
    plt.show()








