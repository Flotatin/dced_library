"Pressure Law"
import numpy as np
from pynverse import inversefunc

def Raman_Dc12__2003_Occelli(peakmax,lamb0=1333,sigmalambda=None):
    deltalambda=peakmax-lamb0
    P=deltalambda/2.83 # delta= lamb0 + 2.83*P + 3.65e-3*P**2
    if sigmalambda == None:
        return P
    else:
        sigmaP= sigmalambda/2.90
        return P , sigmaP

def Raman_Dc13_1997_Schieferl(peakmax,lamb0=1287.79,sigmalambda=None):
    deltalambda=peakmax-lamb0
    P=deltalambda/2.83
    if sigmalambda == None:
        return P
    else:
        sigmaP= sigmalambda/2.83
        return P , sigmaP


def Sm_2015_Rashenko(peakMax,lamb0=685.41,sigmalambda=None): #https://doi.org/10.1063/1.4918304
    deltalambda=peakMax-lamb0
    P=4.20*deltalambda*(1+0.020*deltalambda)/(1+0.036*deltalambda)

    if sigmalambda == None:
        return P
    else:
        coe1=4.20
        coe2=0.02
        coe3=0.036
        sigmaP= sigmalambda*(((((coe1*(1 + (coe2*deltalambda)))+(coe1*coe2*deltalambda))*(1 + (coe3*deltalambda)))-coe3*deltalambda*coe1*(1 + (coe2*deltalambda)))/((1 + (coe3*deltalambda))**2))
        return P , sigmaP

def Sm_1997_Datchi(peakMax,lamb0=685.41,sigmalambda=None): #https://doi.org/10.1063/1.365025
    deltalambda=peakMax-lamb0
    P=4.032*(deltalambda)*(1+(deltalambda)*9.29e-3)/(1+(deltalambda)*2.32e-2)
    return P

def T_Ruby_Sm_1997_Datchi(peakMaxR,peakMaxS,lamb0S=685.41,lamb0R=694.3,sigmalambda=None): #https://doi.org/10.1063/1.365025
    deltaR=peakMaxR-lamb0R
    deltaS=peakMaxS-lamb0S
    T=300 + 137*((deltaR)-1.443*deltaS)

    if sigmalambda == None:
        return T
    else:
        sigmaT= 137*np.sqrt( sigmalambda[0]**2 + (1.443*sigmalambda[1])**2)
        return T , sigmaT

def T_Ruby_by_P(peakMaxR,lamb0R=694.3,sigmalambda=None,P=0.01):
    deltaS=inversefunc(Sm_1997_Datchi,P,[680,800])-685.41
    deltaR=peakMaxR-lamb0R
    T=300 + 137*((deltaR)-1.443*deltaS)

    if sigmalambda == None:
        return T
    else:
        sigmaT= 137*np.sqrt( sigmalambda[0]**2 + (1.443*sigmalambda[1])**2)
        return T , sigmaT

def Ruby_2020_Shen(peakMax,lamb0=694.3,sigmalambda=None): #https://doi.org/10.1080/08957959.2020.1791107
    deltalambda=peakMax-lamb0
    P=1870*(deltalambda)/lamb0*(1+5.63*((deltalambda)/lamb0))
    if sigmalambda == None:
        return P
    else:
        sigmaP=sigmalambda*(1870/lamb0*(1+2*5.63*((deltalambda)/lamb0)))
        return P , sigmaP

def Ruby_1986_Mao(peakMax,lamb0=694.24,sigmalambda=None,hydro=True): #https://doi.org/10.1029/JB091iB05p04673
    deltalambda=peakMax-lamb0
    if hydro==True:
        B=5
    else:
        B=7.665
    P= 1904/B*((1+deltalambda/lamb0)**B-1)
    if sigmalambda == None:
        return P
    else:
        sigmaP=sigmalambda*1904/lamb0(1+deltalambda/lamb0)**(B-1)
        return P , sigmaP

def SrFCl(peakMax,lamb0=690.1,sigmalambda=None): # lambda0 fausse et article a retrouver
    x= (peakMax - lamb0)/lamb0
    coe1=620.6
    coe2=-4.92
    P= coe1*x*(1+coe2*x)
    if sigmalambda ==None:
        return P
    else:
        sigmaP= sigmalambda*coe1*(1 + coe2*x)/lamb0
        return P , sigmaP

def Rhodamine_6G_2024_Dembele(peakMax,lamb0 = 551.3916,sigmalambda=None): # !!!!! ETALONNER POUR TROUVER LA LOI PUIS CODER !!!!!!
    #Calib1_251024_16h39_d11_04t11_26     lamb0 = 551.3916
    a = 0
    b = 0.04166
    #Calib Wintec
    #a = 0.0010497850908414974 
    #b = 0.04267468326014777
    deltalambda= peakMax -lamb0
    P=   a*((deltalambda))**2+b*((deltalambda)) 
    if sigmalambda ==None:
        return P
    else:
        sigmaP= a*sigmalambda*deltalambda*2 + b*sigmalambda
        return P , sigmaP

