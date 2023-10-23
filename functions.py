import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pylab
import numpy as np
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import itertools
import subprocess
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import simps

#READ ALL DATASETS
def read_data():
    Data={}
    thr=60#data was recorded from t=0, but the experiment does not start until t=60
    DataCl=pd.read_excel('data/dataCl.xlsx')
    DataCl=DataCl[DataCl.time>=thr].reset_index(drop=True)
    DataS=pd.read_excel('data/dataS.xlsx')
    DataS=DataS[DataS.time>=thr].reset_index(drop=True)
    DataE=pd.read_excel('data/dataE.xlsx')
    DataE=DataE[DataE.time>=thr].reset_index(drop=True)
    DataE2=pd.read_excel('data/dataE2.xlsx')
    DataE2=DataE2[DataE2.time>=thr].reset_index(drop=True)
    DataR100=pd.read_excel('data/dataR100.xlsx')
    DataR100=DataR100[DataR100.time>=thr].reset_index(drop=True)
    DataR400=pd.read_excel('data/dataR400.xlsx')
    DataR400=DataR400[DataR400.time>=thr].reset_index(drop=True)
    Data['Cl']=DataCl
    Data['S']=DataS
    Data['E']=DataE
    Data['E2']=DataE2
    Data['R100']=DataR100
    Data['R400']=DataR400
    return Data

#ODE RELATING REAL AND MEASURED SIGNAL
def itc(t,y,tITC,f):
    H=float(f(t))
    return 1.0/tITC*(H-y)

#FROM REAL TO MEASURED SIGNAL
def measuredH(T,H):
    tITC=8
    f=interp1d(T,H)
    sol=solve_ivp(itc,(0,T[-1]),y0=[0],method='RK45',max_step=1, t_eval=T,args=(tITC,f))
    return sol.y[0,:]

#FROM MESURED TO REAL SIGNAL
def realH(D,tITC=8):
    D=pd.Series(D)
    Ddiff=D.diff()
    return D+tITC*Ddiff

#ODES FOR MODEL
def oxa(t, y, Params):      
    #Variables
    E=y[0]#dimer
    S=y[1]#antibiotic
    C=y[2]#active complex
    C2=y[3]#inactive complex
    #Parameters
    E0=Params.E0
    k0=Params.k0
    k1=Params.k1
    k2=Params.k2
    k3=Params.k3
    k4=Params.k4
    #Equations
    dEdt=E0/k0*(1-np.heaviside(t-k0,1))-k1*E*S+k2*C
    dSdt=-k1*E*S
    dCdt=k1*E*S-k2*C-k3*C+k4*C2
    dC2dt=k3*C-k4*C2
    return [dEdt, dSdt, dCdt, dC2dt]

#SIMULATED HEAT RELEASE
def heat(C,c,k2):
    return -c*k2*C

#PARAMETERS CLASS FOR OXA ROUTINE
class Parameters:
    def __init__(self,p0,p1,p2,p3,p4,E0):
        self.k0=p0
        self.k1=p1
        self.k2=p2
        self.k3=p3
        self.k4=p4
        self.E0=E0

#NUMERICAL INTEGRATION
def auc(D):
    return simps(D)

#OBTAIN c FROM DATA
def getc(D,S0):
    return -D.sum()/S0

#OBTAIN k2 FROM DATA
def getk2(D,c,E0):
    return (-D).max()/c/E0

#OBTAIN k3 AND k4 FROM DATA
def getk3k4(D,Cl,E0,S0,xend):
    if Cl==0:
        return 0,0
    xinit=33
    #xend=750
    xp=np.array(range(xinit,xend))
    c=getc(D,S0)
    k2=getk2(D,c,E0)
    D1=-D[xinit:xend]
    D1/=c
    D1/=k2
    D1/=E0
    f = lambda x, *p: p[1]/(p[0]+p[1])+p[0]/(p[0]+p[1])*np.exp(-(p[0]+p[1])*x)
    popt, pcov = curve_fit(f,xp-xinit,D1,[0.01,0.001])
    return popt[0],popt[1]

#COMPUTE R SQUARE
def getR2(y,pred):
    Y=np.array(y)
    eT=Y-Y.mean()
    ssT=np.sum(eT**2)
    eR=Y-pred#y_hat
    ssE=np.sum(eR**2)
    R2=1-ssE/ssT
    return R2

#MODIFIED MODEL (HYDROLYSIS FROM THE INACTIVE INTERMEDIATE)
def oxak5(t, y, Params):      
    #Variables
    E=y[0]#dimer
    S=y[1]#antibiotic
    C=y[2]#active complex
    C2=y[3]#inactive complex
    #Parameters
    Cl=Params.Cl
    E0=Params.E0
    k0=Params.k0
    k1=Params.k1
    k2=Params.k2
    k3=Params.k3
    k4=Params.k4
    k5=Params.k5
    #Equations
    dEdt=E0/k0*(1-np.heaviside(t-k0,1))-k1*E*S+k2*C+k5*C2
    dSdt=-k1*E*S
    dCdt=k1*E*S-k2*C-k3*C+k4*C2
    dC2dt=k3*C-k4*C2-k5*C2
    return [dEdt, dSdt, dCdt, dC2dt]

def heatk5(c,k2,C,k5,C2):
    return -c*k2*C-c*k5*C2

class Parametersk5:
    def __init__(self,p0,p1,p2,p3,p4,p5,Cl,E0):
        self.k0=p0
        self.k1=p1
        self.k2=p2
        self.k3=p3
        self.k4=p4
        self.k5=p5
        self.Cl=Cl
        self.E0=E0
