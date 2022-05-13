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

#READ ALL DATASETS
def read_data():
    Data={}
    thr=60#data was recorded from t=0, but the experiment does not start until t=60
    DataCl=pd.read_excel('data/dataCl.xls')
    DataCl=DataCl[DataCl.time>=thr].reset_index(drop=True)
    DataS=pd.read_excel('data/dataS.xls')
    DataS=DataS[DataS.time>=thr].reset_index(drop=True)
    DataE=pd.read_excel('data/dataE.xls')
    DataE=DataE[DataE.time>=thr].reset_index(drop=True)
    DataE2=pd.read_excel('data/dataE2.xls')
    DataE2=DataE2[DataE2.time>=thr].reset_index(drop=True)
    DataR100=pd.read_excel('data/dataR100.xls')
    DataR100=DataR100[DataR100.time>=thr].reset_index(drop=True)
    DataR400=pd.read_excel('data/dataR400.xls')
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
    sol=solve_ivp(itc,(0,T[-1]),y0=[0],method='Radau',t_eval=T,args=(tITC,f))
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
    k3=Params.k3*np.heaviside(t-Params.delay,0)
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
    def __init__(self,p0,p1,p2,p3,p4,delay,E0):
        self.k0=p0
        self.k1=p1
        self.k2=p2
        self.k3=p3
        self.k4=p4
        self.E0=E0
        self.delay=delay

#OBTAIN c FROM DATA
def getc(D,S0):
    return -D.sum()/S0

#OBTAIN k2 FROM DATA
def getk2(D,c,E0):
    return (-D).max()/c/E0

#OBTAIN k3 FROM DATA
def getk3(D,Cl):
    if Cl==0:
        return 0
    xinit=35
    xend=55
    xp=range(xinit,xend)
    D1=-D[xinit:xend]
    f = lambda x, *p: p[1] * x + p[0]
    popt, pcov = curve_fit(f,xp,np.log(D1),[1,1])
    return -popt[1]

#OBTAIN k4 FROM DATA
def getk4(D,k3,Cl):
    if Cl==0:
        return 1e8
    Hs=-D[400:600].mean()
    if Cl>200:
        Hs=-D[600:1200].mean()
    Hmax=(-D).max()
    k4=k3*Hs/(Hmax-Hs)
    return k4

#COMPUTE R SQUARE
def getR2(y,pred):
    Y=np.array(y)
    eT=Y-Y.mean()
    ssT=np.sum(eT**2)
    eR=Y-pred#y_hat
    ssE=np.sum(eR**2)
    R2=1-ssE/ssT
    return R2