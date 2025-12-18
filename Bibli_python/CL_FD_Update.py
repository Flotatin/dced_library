import copy
import os
import re
import sys
import time
from math import log, sqrt
import cv2
import dill
import lecroyscope
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils as pk
from PIL import Image
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import Button, Slider
from lmfit.models import (
    GaussianModel,
    MoffatModel,
    Pearson4Model,
    PseudoVoigtModel,
    SplitLorentzianModel,
)
from pynverse import inversefunc
from scipy.signal import savgol_filter
from scipy.special import beta, gamma
from tqdm import tqdm

""" ------------------------------------- FONCTION GESTION DE FICHIER -------------------------------------"""
def SAVE_CEDd(file_CEDd,bit_try=False):
    if file_CEDd:
        if bit_try==True:
            try:
                dill.dump( file_CEDd, open( file_CEDd.CEDd_path, "wb" ) )
            except Exception as e:
                print("ERROR : ",e," in SAVE_CEDd")
        else:
            dill.dump( file_CEDd, open( file_CEDd.CEDd_path, "wb" ) )
            
def LOAD_CEDd(CEDd_path,bit_try=False):
    if CEDd_path:
        if bit_try==True:
            try:
                CEDd = dill.load( open( CEDd_path, "rb" ) )
                CEDd.CEDd_path=CEDd_path
                return CEDd
            except Exception as e:
                print("ERROR : ",e," in LOAD_CEDd")
        else:
            CEDd = dill.load( open( CEDd_path, "rb" ) )
            CEDd.CEDd_path=CEDd_path
            return CEDd

def Load_last(Folder,extend=None,file=True):
    if file ==True:
        if extend != None :
            file_names = [f for f in os.listdir(Folder) if os.path.isfile(os.path.join(Folder, f)) and extend in f]
        else:
            file_names = [f for f in os.listdir(Folder) if os.path.isfile(os.path.join(Folder, f))]
    else:
        file_names = [f for f in os.listdir(Folder)]
    if file_names:
        file_names.sort(key=lambda f: os.path.getmtime(os.path.join(Folder, f)))
        latest_file_name = file_names[-1]
        latest_file_path = os.path.join(Folder, latest_file_name)
    
        return latest_file_path, latest_file_name
    else:
        return None,None
""" ------------------------------------- FENETRE UTILE -------------------------------------"""

class ProgressDialog(QDialog):
    def   __init__(self,figs,value,Gauges,timeout=25000, parent = None):
        super().__init__(parent)
        self.setWindowTitle("See the loop")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.user_choice=True
        self.time_delay=True

        #layout principale
        layout = QVBoxLayout(self)
        #self.fig ((self.ax1,self.ax2),(self.self.ax1b,self.ax2b)) = plt.subplots(2,2,figsize=(10,5))
        fig_layout=QHBoxLayout()
        for fig in figs:
            containers = QVBoxLayout()
            canvas=FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas,self)
            containers.addWidget(canvas)
            containers.addWidget(toolbar)
            fig_layout.addLayout(containers)
        layout.addLayout(fig_layout)

        self.param_Gauges=[]
        self.Param0=[]
        self.inter=0.5
        param_layout=QVBoxLayout()
        for i,G in enumerate(Gauges):
            self.param_Gauges.append([])
            self.Param0.append([])
            param_Gi_layout=QHBoxLayout()
            for p in G.pics:
                ctr_label = QLabel(f"{p.name} x:")
                spinbox_ctr= QDoubleSpinBox()
                spinbox_ctr.valueChanged.connect(self.setFocus)
                spinbox_ctr.setRange(0,4000)
                spinbox_ctr.setSingleStep(0.2)
                spinbox_ctr.setValue(p.ctr[0])

                sigma_label = QLabel(f"{p.name} Sig:")
                spinbox_sigma= QDoubleSpinBox()
                spinbox_sigma.valueChanged.connect(self.setFocus)
                spinbox_sigma.setRange(0.01,80)
                spinbox_sigma.setSingleStep(0.05)
                spinbox_sigma.setValue(p.sigma[0])
                self.Param0[i].append([p.ctr[0],p.sigma[0]])
                self.param_Gauges[i].append([spinbox_ctr,spinbox_sigma])
                param_Gi_layout.addWidget(ctr_label)
                param_Gi_layout.addWidget(spinbox_ctr)
                param_Gi_layout.addWidget(sigma_label)
                param_Gi_layout.addWidget(spinbox_sigma)
            param_layout.addLayout(param_Gi_layout)
        
        inter_name = QLabel("Variation:")
        self.inter_entry = QDoubleSpinBox()
        self.inter_entry.setRange(0.1,5)
        self.inter_entry.setSingleStep(0.1)
        self.inter_entry.setValue(Gauges[0].pics[0].inter)
        param_layout.addWidget(inter_name)
        param_layout.addWidget(self.inter_entry)


        layout.addLayout(param_layout)

        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0,100)
        self.progressBar.setValue(value)
        layout.addWidget(self.progressBar)


        #boutons
        btn_layout = QHBoxLayout()
        self.btn_continue =QPushButton("Continue [v]")
        self.btn_cancel=QPushButton("Cancel [c]")
        self.box_time_delay=QCheckBox("CODE")
        btn_layout.addWidget(self.btn_continue)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)


        #connexion
        self.btn_continue.clicked.connect(self.on_continue)
        self.btn_cancel.clicked.connect(self.on_cancel)


        self.timer =QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.on_continue)
        self.timer.start(timeout)


        
    def on_continue(self):
        self.user_choice=True
        for i,p_G in enumerate(self.Param0):
            for j,p_p in enumerate(p_G):
                self.Param0[i][j]=[float(self.param_Gauges[i][j][0].value()),float(self.param_Gauges[i][j][1].value())]
        self.inter=float(self.inter_entry.value())
        print(self.Param0)
        self.accept()
    
    def on_cancel(self):
        self.user_choice=False
        self.reject()
    
    def keyPressevent(self,event):
        print(event)
        if event.key== Qt.Key_V:
            self.on_continue()
        elif event.key== Qt.Key_C:
            self.on_cancel()

        elif event.key()== Qt.Key_Space:
            event.ignore()
    
    def get_user_choice(self):
        return self.user_choice,self.Param0,self.inter


""" ------------------------------------- LOI DE PRESSION -------------------------------------"""

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

"""------------------------------------- LOI FIT -------------------------------------"""      
def PseudoVoigt(x,center,ampH,sigma,fraction):
    amp=ampH/(((1-fraction))/(sigma*sqrt(np.pi/log(2)))+(fraction)/(np.pi*sigma))
    sigma_g=(sigma/(sqrt(2*log(2))))
    return (1-fraction)*amp/(sigma_g*sqrt(2*np.pi))*np.exp(-(x-center)**2/(2*sigma_g**2))+fraction*amp/np.pi*(sigma/((x-center)**2+sigma**2))

def Moffat(x,center,ampH,sigma,beta):
    amp=ampH*beta
    return amp*(((x-center)/sigma)**2+1)**(-beta)

def SplitLorenzian(x, center, ampH, sigma, sigma_r):
    amp = ampH * 2 / (np.pi * (sigma + sigma_r))
    sig = np.where(x < center, sigma, sigma_r)
    y = amp * (sig**2) / ((x - center)**2 + sig**2)
    return y

def PearsonIV(x, center,ampH, sigma, m, skew):
    center =center + sigma*skew/(2*m)
    return ampH / ((1 + (skew/(2*m))**2)**-m * np.exp(-skew * np.arctan(-skew/(2*m)))) * (1 + ((x - center) / sigma)**2)**-m * np.exp(-skew * np.arctan((x - center) / sigma))

 #  amp= ampH / (normalization*(1 + (skew/(2*m))**2)**-m * np.exp(-skew * np.arctan(-skew/(2*m))))   #amp * normalization * (1 + ((x - center) / sigma)**2)**-m * np.exp(-skew * np.arctan((x - center) / sigma))

def Gaussian(x,center,ampH,sigma):
    return ampH*np.exp(-(x-center)**2/(2*sigma**2))

def Gen_sum_F(list_F):
    def sum_F(x,*params):
        params_split=[]
        index=0
        for f in list_F:
            num_params= f.__code__.co_argcount-1 #Nb parma de la foncion
            params_split.append(params[index:index+num_params])
            index+=num_params
        result = np.array([0 for _ in x])#np.zeros((1,len(x)))
        for f, f_params in zip(list_F,params_split):
            result=result+f(x,*f_params)
        return result
    return sum_F


"""------------------------------------- CLASSE JAUGE SPECTRO -------------------------------------"""      
""" CLASSE modele de Pics """
class Pics:
    def __init__(self,name='',ctr=0,ampH=1,coef_spe=[0.5],sigma=0.25,inter=3,model_fit="PseudoVoigt",Delta_ctr=0.5,amp=None):
        self.name=name.replace("-", "_")
        self.ctr=[ctr,[ctr-Delta_ctr,ctr+Delta_ctr]]
        self.model_fit=model_fit
        self.name_coef_spe=None
        self.f_model=None
        self.f_amp=None
        inter_min=max(1-inter,0)
        self.sigma=[sigma,[sigma*inter_min,sigma*(1+inter)]]
        self.lim_sigma=[sigma*0.05,sigma*8]
        self.inter=inter
        self.ampH=[ampH,[ampH*inter_min,ampH*((1+inter))]]
        self.best_fit=None
        self.help="Pics: définition de modéle de pics de luminesance"
        coef_spe=np.array(coef_spe)
        self.coef_spe= [[c,[c*inter_min,c*(1+inter)]]for c in coef_spe]# [coef_spe,[coef_spe*inter_min,[max(1,c*(1+inter)) for c in coef_spe]]]

        if model_fit == "PseudoVoigt":
            self.coef_spe=[[max([min([1,coef_spe[0]]),0]),[coef_spe[0]*inter_min,max(0.1,min([1,coef_spe[0]*(1+inter)]))]]]
            self.f_amp=self.Amp_PsD
            self.name_coef_spe=["fraction"]
            self.model = PseudoVoigtModel(prefix=self.name)
            self.f_model=PseudoVoigt
        elif model_fit =="Moffat":
            self.f_amp=self.Amp_Moffat
            self.name_coef_spe=["beta"]
            self.model = MoffatModel(prefix=self.name)
            self.f_model=Moffat
        elif model_fit == "SplitLorentzian":
            self.f_amp=self.Amp_SplitL
            self.name_coef_spe=["sigma_r"]
            self.model = SplitLorentzianModel(prefix=self.name)
            self.f_model=SplitLorenzian

        elif model_fit == "Gaussian":
            self.f_amp=self.Amp_Gaussian
            self.name_coef_spe=[]
            self.model =GaussianModel(prefix=self.name)
            self.f_model=Gaussian

        elif model_fit =="PearsonIV":
            self.f_amp=self.Amp_PearsonIV
            self.name_coef_spe=["expon","skew"]
            self.model = Pearson4Model(prefix=self.name)
            self.f_model=PearsonIV
            if len(coef_spe)<2:
                coef_spe=[coef_spe[0],coef_spe[0]]
            lc=[coef_spe[1]*(1-self.inter)-0.1,coef_spe[1]*(1+self.inter)+0.1]
            self.coef_spe=[ [max([0.505,coef_spe[0]]),[max([0.501,self.coef_spe[0][0]*inter_min]),max(0.51,coef_spe[0]*(1+self.inter))]] , [coef_spe[1],[min(lc),max(lc)]]]
            ctr=ctr + self.sigma[0]*self.coef_spe[1][0]/(2*(self.coef_spe[0][0]))
            self.ctr=[ctr,[ctr-Delta_ctr,ctr+Delta_ctr]]

        if amp is None:
            amp=self.f_amp(self.ampH[0],[float(c[0]) for c in self.coef_spe],sigma) #ampH/(((1-coef_spe))/(sigma*sqrt(np.pi/log(2)))+(coef_spe)/(np.pi*sigma))
        
        self.amp=[amp,[amp*inter_min,amp*((1+inter))]]


        for i , name_spe in enumerate(self.name_coef_spe):
            self.model.set_param_hint(self.name+name_spe, value=self.coef_spe[i][0],min=self.coef_spe[i][1][0],max=self.coef_spe[i][1][1])
        self.model.set_param_hint(self.name+'amplitude', value=self.amp[0],min=self.amp[1][0],max=self.amp[1][1])
        self.model.set_param_hint(self.name+'sigma', value=self.sigma[0],min=self.sigma[1][0],max=self.sigma[1][1])
        self.model.set_param_hint(self.name+'center', value=self.ctr[0],min=self.ctr[1][0],max=self.ctr[1][1])

    def Amp_PsD(self,ampH,coef_spe,sigma):
        return ampH/(((1-coef_spe[0]))/(sigma*sqrt(np.pi/log(2)))+(coef_spe[0])/(np.pi*sigma))
    
    def Amp_Gaussian(self,ampH,coef_spe,sigma):
        return ampH*sigma*sqrt(np.pi*2)
    
    def Amp_Moffat(self,ampH,coef_spe,sigma):
        return ampH*coef_spe[0]
    
    def Amp_SplitL(self,ampH,coef_spe,sigma):
        return ampH/2*np.pi*(sigma+coef_spe[0])
    
    def Amp_PearsonIV(self,ampH, coef_spe,sigma):
        m,skew = coef_spe[0],coef_spe[1]
        normalization = np.abs(gamma(m + 1j * skew / 2)/gamma(m))**2 / (sigma * beta(m-0.5,0.5))
        return  ampH / (normalization*(1 + (skew/(2*m))**2)**-m * np.exp(-skew * np.arctan(-skew/(2*m))))

    def Update(self,ctr=None,ampH=None,coef_spe=None,sigma=None,inter=None,model_fit=None,Delta_ctr=None,amp=None,move=False):
        if Delta_ctr is None:
            Delta_ctr =0.4
        if inter !=None:
            self.inter=inter
        inter_min=max(1-self.inter,0)

        if coef_spe is not None :
            self.coef_spe=[[c,[c*inter_min,c*(1+self.inter)]]for c in coef_spe]

        if ctr != None :
            self.ctr=[ctr,[ctr-Delta_ctr,ctr+Delta_ctr]]
        if sigma == None:
            sigma=float(self.sigma[0])
        sig_min,sig_max=getattr(self,'lim_sigma',(0,2.5))
        self.sigma=[sigma,[max(sigma*inter_min,sig_min),min(sigma*(1+self.inter),sig_max)]]

        if (model_fit is not None) and  (model_fit is not self.model_fit):
            self.model_fit=model_fit
            if model_fit == "PseudoVoigt":
                self.f_amp=self.Amp_PsD
                self.name_coef_spe=["fraction"]
                self.model = PseudoVoigtModel(prefix=self.name)
                self.coef_spe=[[max(min(1,self.coef_spe[0][0]),0),[self.coef_spe[0][0]*inter_min,max(0.1,min(1,self.coef_spe[0][0]*(1+self.inter)))]]]
                self.f_model=PseudoVoigt
            elif model_fit =="Moffat":
                self.f_amp=self.Amp_Moffat
                self.name_coef_spe=["beta"]
                self.model = MoffatModel(prefix=self.name)
                self.f_model=Moffat
            elif model_fit == "SplitLorentzian":
                self.f_amp=self.Amp_Moffat
                self.name_coef_spe=["sigma_r"]
                self.model = SplitLorentzianModel(prefix=self.name)
                self.f_model=SplitLorenzian
            elif model_fit == "Gaussian":
                self.f_amp=self.Amp_Gaussian
                self.name_coef_spe=[]
                self.model = GaussianModel(prefix=self.name)
                self.f_model=Gaussian
            elif model_fit =="PearsonIV":
                self.f_amp=self.Amp_PearsonIV
                self.name_coef_spe=["expon","skew"]
                self.model = Pearson4Model(prefix=self.name)
                self.f_model=PearsonIV
                if len(self.coef_spe)!=2:
                    self.coef_spe=[[self.coef_spe[0][0],[0.5,1]],[self.coef_spe[0][0],[0.5,1]]]
                lc=[self.coef_spe[1][0]*(1-self.inter)-0.1,self.coef_spe[1][0]*(1+self.inter)+0.1]
                self.coef_spe=[[max(0.505,self.coef_spe[0][0]),[max(0.501,self.coef_spe[0][0]*inter_min),max(0.51,self.coef_spe[0][0]*(1+self.inter))]],[self.coef_spe[1][0],[min(lc),max(lc)]]]
                ctr=self.ctr[0] + self.sigma[0]*self.coef_spe[1][0]/(2*(self.coef_spe[0][0]))
                self.ctr=[ctr,[ctr-Delta_ctr,ctr+Delta_ctr]]
        else:
            if self.model_fit== "PseudoVoigt":
                self.coef_spe=[[max(min(1,self.coef_spe[0][0]),0),[self.coef_spe[0][0]*inter_min,max(0.1,min(1,self.coef_spe[0][0]*(1+self.inter)))]]]

            elif self.model_fit =="PearsonIV":
                self.f_model=PearsonIV
                if len(self.coef_spe)<2:
                    self.coef_spe=[coef_spe[0][0]],[coef_spe[0][0]]
                lc=[self.coef_spe[1][0]*(1-self.inter)-0.1,self.coef_spe[1][0]*(1+self.inter)+0.1]
                self.coef_spe=[[max(0.501,self.coef_spe[0][0]),[max(0.501,self.coef_spe[0][0]*inter_min),max(0.51,self.coef_spe[0][0]*(1+self.inter))]],[self.coef_spe[1][0],[min(lc),max(lc)]]]
                ctr=self.ctr[0] + self.sigma[0]*self.coef_spe[1][0]/(2*(self.coef_spe[0][0]))
                self.ctr=[ctr,[ctr-Delta_ctr,ctr+Delta_ctr]]

        if ampH is not None and amp is None:
            self.ampH=[ampH,[ampH*inter_min,ampH*((1+self.inter))]]
            amp=self.f_amp(self.ampH[0],[c[0] for c in self.coef_spe],self.sigma[0]) #ampH/(((1-coef_spe))/(sigma*sqrt(np.pi/log(2)))+(coef_spe)/(np.pi*sigma))
            self.amp=[amp,[amp*inter_min,amp*((1+self.inter))]]
        elif amp is not None:
            self.amp=[amp,[amp*inter_min,amp*((1+self.inter))]]
        else:
            amp=self.f_amp(self.ampH[0],[c[0] for c in self.coef_spe],self.sigma[0]) #ampH/(((1-coef_spe))/(sigma*sqrt(np.pi/log(2)))+(coef_spe)/(np.pi*sigma))
            self.amp=[amp,[amp*inter_min,amp*((1+self.inter))]]

        for i , name_spe in enumerate(self.name_coef_spe):
            self.model.set_param_hint(self.name+name_spe, value=self.coef_spe[i][0],min=self.coef_spe[i][1][0],max=self.coef_spe[i][1][1])
        self.model.set_param_hint(self.name+'amplitude', value=self.amp[0],min=self.amp[1][0],max=self.amp[1][1])
        self.model.set_param_hint(self.name+'sigma', value=self.sigma[0],min=self.sigma[1][0],max=self.sigma[1][1])
        self.model.set_param_hint(self.name+'center', value=self.ctr[0],min=self.ctr[1][0],max=self.ctr[1][1])

    def Out_model(self,out=None,l_params=None):
        if out is not None:
            ctr=round(out.params[self.name+'center'].value,3)
            sigma=round(out.params[self.name+'sigma'].value,3)
            ampH=round(out.params[self.name+'height'].value,3)
            coef_spe=np.array([round(out.params[self.name+name_spe].value,3) for name_spe in self.name_coef_spe])
            if self.model_fit == 'PearsonIV':
                #ctr1=ctr - sigma*coef_spe[1]/(4*(coef_spe[0]-1))
                ctr=round(out.params[self.name+'position'].value,3)    
            return [ctr,ampH,sigma,coef_spe]
        elif l_params is not None:
            ctr=round(l_params[0],3)
            sigma=round(l_params[2],3)
            ampH=round(l_params[1],3)
            coef_spe=np.array([round(p,3) for p in l_params[3]])
            if self.model_fit == 'PearsonIV':
                ctr=ctr - sigma*coef_spe[1]/(4*(coef_spe[0]-1))
            return [ctr,ampH,sigma,coef_spe] 
        else:
            params=self.model.make_params()
            ctr=round(params[self.name+'center'].value,3)
            sigma=round(params[self.name+'sigma'].value,3)
            ampH=round(params[self.name+'height'].value,3)
            coef_spe=np.array([round(params[self.name+name_spe].value,3) for name_spe in self.name_coef_spe])
            if self.model_fit == 'PearsonIV':
                #ctr1=ctr - sigma*coef_spe[1]/(4*(coef_spe[0]-1))
                ctr=round(params[self.name+'position'].value,3)
            return [ctr,ampH,sigma,coef_spe] , params
        
    def Help(self, request=None):
        if request is None:
            print("Choisissez un modèle pour obtenir des informations détaillées :")
            print("Disponible : 'PseudoVoigt', 'Moffat', 'SplitLorentzian', 'Gaussian', 'PearsonIV'.")
        elif request == 'param':
            if self.model_fit == "PseudoVoigt":
                print("\nModèle : PseudoVoigt")
                print("Description : Modèle hybride entre un Lorentzien et un Gaussien.")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic, écart type (défini par 'sigma').")
                print("  - fraction : Fraction entre le Lorentzien et le Gaussien (0 < fraction < 1).")
                print("  Exemple : PseudoVoigt(x, ctr, ampH, sigma, fraction)")

            elif self.model_fit == "Moffat":
                print("\nModèle : Moffat")
                print("Description : Modèle de pic Moffat, utilisé pour des pics avec une décroissance plus lente qu'un Lorentzien.")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic, écart type (défini par 'sigma').")
                print("  - beta : Paramètre de forme qui détermine l'étalement du pic (beta > 0).")
                print("  Exemple : Moffat(x, ctr, ampH, sigma, beta)")

            elif self.model_fit == "SplitLorentzian":
                print("\nModèle : SplitLorentzian")
                print("Description : Modèle Lorentzien fractionné avec deux valeurs de largeur (sigma et sigma_r).")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic gauche (défini par 'sigma').")
                print("  - sigma_r : Largeur du pic droite (défini par 'sigma_r').")
                print("  Exemple : SplitLorentzian(x, ctr, ampH, sigma, sigma_r)")

            elif self.model_fit == "Gaussian":
                print("\nModèle : Gaussian")
                print("Description : Modèle Gaussien classique.")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic, écart type (défini par 'sigma').")
                print("  Exemple : Gaussian(x, ctr, ampH, sigma)")

            elif self.model_fit == "PearsonIV":
                print("\nModèle : Pearson IV")
                print("Description : Modèle Pearson IV, adapté pour des pics plus asymétriques avec une forme flexible.")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic, écart type (défini par 'sigma').")
                print("  - m : Paramètre de forme de Pearson IV, contrôle la symétrie du pic (m > 0).")
                print("  - skew : Asymétrie du pic, peut être positive ou négative.")
                print("  Exemple : PearsonIV(x, ctr, ampH, sigma, m, skew)")

        elif request == 'test':
            print("OK")

""" CLASSE JAUGE """
# p1 le pique principale de mesure 
# nb_pic : le nombre de pic
# lamb0 la lambda0 de l'ambitante
# name = nom à trouver
# color_print = [ couleur principale , couleur des pique]

class Gauge:
    def __init__(self,name="",lamb0=None,nb_pic=None,X=None,Y=None,deltaP0i=[],spe=None,f_P=None,name_spe=''):
        self.lamb0=lamb0
        self.nb_pic=nb_pic
        self.deltaP0i=deltaP0i
        self.pics=[]
        self.name=name
        self.X=X
        self.Y=Y
        self.dY=None
        self.spe=spe
        self.f_P=f_P
        if f_P is not None:
            self.inv_f_P= inversefunc((lambda x :f_P(x)),domain=[10,1000])
        else:
            self.inv_f_P=None
        self.P=0
        
        self.T=0
        self.name_spe=name_spe
    
        self.bit_model=False
        self.model=None
        self.color_print=[None,None]
        self.bit_fit=False
        self.lamb_fit=None
        self.indexX=None
        self.fit="Fit Non effectué"
        self.study=pd.DataFrame()
        self.study_add=pd.DataFrame()
        
        self.state="Y" #Y, IN_NOISE , N 1,0,None
        self.Jauge_choice=["Ruby","Sm","SrFCl","Rhodamine6G"]

        self.help="Gauge: G de spectroscopie (Ruby,Samarium,SrFCL) prochainement jauge X A CODER \n #Calib1_251024_16h39_d11_04t11_26 Coefficients ajustés X=x_lamb0 et P=aX**2+bX+c: a = 0, b = 0.04165619359945429,,lambda0 = 551.3915961705188 "

        if "Ruby" in self.name:
            self.Ruby()
        elif ("Sm" in self.name) or  ("Samarium" in self.name):
            self.Sm()
        elif "SrFCl" in self.name:
            self.SrFCl()
        elif "Rhodamine" in self.name:
            self.Rhodamine6G()
        elif "Diamond_c12" in self.name:
            self.Diamond(isotope="c12")
        elif "Diamond_c13" in self.name:
            self.Diamond(isotope="c13")

    def Init_perso(self):
        self.pics=[]
        for i in range(self.nb_pic):
            if "Sig" in self.name_spe:
                match = re.search("Sig(\d+)", self.name_spe)
                sigma= float(match.group(1))
            else:
                sigma=0.25
            if "Mod:" in self.name_spe:
                match = re.search(r"Mod:(\w+)", self.name_spe)
                model_fit= str(match.group(1))
            else:
                model_fit="PseudoVoigt"
            new_pics=Pics(name=self.name + '_p'+str(i+1),ctr=self.lamb0+self.deltaP0i[i][0],sigma=sigma,model_fit=model_fit)
            self.pics.append(new_pics)
            if i == 0:
                self.model = new_pics.model
            else:
                self.model = self.model + new_pics.model
        
        self.bit_model=True

    #JAUGE PREREMPLIS
    def Ruby(self):
        self.lamb0=694.4
        self.nb_pic=2
        self.name="Ruby"
        self.name_spe="Ru"
        self.deltaP0i=[[0,1],[-1.5,0.75]]
        self.color_print=['darkred',['darkred','brown']]
        if self.f_P == None:
            self.f_P=Ruby_2020_Shen
            self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[692,750])

        self.Init_perso()

    def Calcul_Ruby(self,input_spe):
        if 'RuSmT' in self.name_spe :
            if input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr != None:
                sigmals=input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr
            else :
                sigmals=0
            
            if self.fit.params[self.name+'_p1center'].stderr != None:
                sigmal=self.fit.params[self.name+'_p1center'].stderr
            else :
                sigmal=0
            
            T , sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit,input_spe[self.spe].lamb_fit,lamb0S=input_spe[self.spe].lamb0,lamb0R=self.lamb0,sigmalambda=[sigmal,sigmals])
            self.study_add=pd.DataFrame(np.array([[self.deltaP0i[1][0],T,sigmaT]]),columns=['Deltap12','T','sigmaT'])
        elif "RuT" in self.name_spe:
            T , sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit,lamb0R=self.lamb0,sigmalambda=[sigmal,input_spe[self.spe].sigmaP],P=input_spe[self.spe].P)
            self.study_add=pd.DataFrame(np.array([[self.deltaP0i[1][0],T,sigmaT]]),columns=['Deltap12','T','sigmaT'])
        else:
            self.study_add=pd.DataFrame(np.array([self.deltaP0i[1][0]]),columns=['Deltap12'])
                
    def Sm(self,all=False):
        self.lamb0=685.39
        self.nb_pic=1
        self.name="Sm"
        if all ==False:
            self.deltaP0i=[[0,1]]
        else:
            print("to code")

        if self.f_P == None:
            self.f_P= Sm_2015_Rashenko
            self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[683,730])
        self.color_print=['darkblue',['darkblue']]
        self.Init_perso()
    
    def Diamond(self,isotope='c12'):
        if isotope == 'c12':
            self.lamb0=1333
            self.f_P= Raman_Dc12__2003_Occelli
            self.color_print=['silver',['silver']]
        elif isotope =='c13':
            self.lamb0=1287
            self.f_P= Raman_Dc13_1997_Schieferl
            self.color_print=['dimgrey',['dimgrey']]
        self.nb_pic=1
        self.name_spe="Sig20"
        self.name="D_"+isotope
        self.deltaP0i=[[0,1]]
        self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[self.lamb0-10,self.lamb0+200])
        
        self.Init_perso()
    
    def SrFCl(self,all=False): 
        self.lamb0=685.39
        self.nb_pic=1
        self.name="SrFCl"
        if all ==False:
            self.deltaP0i=[[0,1]]
        else:
            print("to code")
        if self.f_P is None:
            self.f_P=SrFCl
            self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[683,730])

        self.color_print=['darkorange',['darkorange']]
        self.Init_perso()

    def Rhodamine6G(self):
        self.lamb0=550
        self.nb_pic=2
        self.name="Rhodamine6G"
        self.deltaP0i=[[0,1],[52,0.5,4]]
        self.color_print=['limegreen',['limegreen',"darkgreen"]]
        self.name_spe="Lw50Hg150Sig20Mod:Gaussian"

        if self.f_P == None:
            self.f_P=Rhodamine_6G_2024_Dembele
            self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[530,600])

        self.Init_perso()

    #CALCULE
    def Calcul(self,input_spe=None,mini=False,lambda_error=0):
        if self.bit_fit is False :
            return print("NO FIT of ",self.name,"do it !")
        else:
            if "DRX" in self.name_spe :
                if self.state == "IN_NOISE":
                    self.a,self.b,self.c,self.rca,self.V,self.P=self.Element_ref.A,self.Element_ref.B,self.Element_ref.C,self.Element_ref.RCA,self.Element_ref.AV0,0
                else:
                    self.CALCUL(mini=mini)
                self.study =pd.concat([pd.DataFrame(np.array([[self.a,self.b,self.c,self.rca,self.V,self.P]]) , columns=['a_'+self.name,'b_'+self.name,'c_'+self.name,'c/a_'+self.name,'V_'+self.name,'P_'+self.name]),self.study_add],axis=1)
                return print("CODER DRX")
            
            self.lamb_fit=round(self.pics[0].ctr[0],3)

            sigmal=lambda_error
            if self.state == "Y":
                
                if hasattr(self.fit,"params"):
                    if self.fit.params[self.name+'_p1center'].stderr != None:
                        if self.fit.params[self.name+'_p1center'].stderr > lambda_error:
                            sigmal=self.fit.params[self.name+'_p1center'].stderr
        
                self.fwhm=round(self.pics[0].sigma[0],5)
                #self.fwhm=self.fit.best_values[self.name+'_p1sigma']

                for i in range(1,len(self.deltaP0i)):
                    #self.deltaP0i[i][0]= round(self.fit.best_values[self.name+'_p'+str(i+1)+'center'] - self.lamb_fit,3)
                    self.deltaP0i[i][0]= round(self.pics[i].ctr[0]- self.lamb_fit,5)
                self.P , self.sigmaP = self.f_P(self.lamb_fit,self.lamb0,sigmalambda=sigmal)

            else:
                self.fwhm,self.P,self.sigmaP =0.1 ,0 ,0

            if 'Ru' in self.name_spe:
                self.study_add=pd.DataFrame(np.array([abs(self.deltaP0i[1][0])]),columns=['Deltap12'])
            
                if 'SmT' in self.name_spe :
                    sigmals= lambda_error
                    if self.state == "Y":
                        if hasattr(input_spe[self.spe].fit,"params"):
                            if input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr != None:
                                if input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr > lambda_error:
                                    sigmals=input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr                        

                        T , sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit,input_spe[self.spe].lamb_fit,lamb0S=input_spe[self.spe].lamb0,lamb0R=self.lamb0,sigmalambda=[sigmal,sigmals])
                    else:
                        T,sigmaT=273,0
                    self.study_add=pd.DataFrame(np.array([[self.deltaP0i[1][0],T,sigmaT]]),columns=['Deltap12','T','sigma_T'])


            self.study =pd.concat([pd.DataFrame(np.array([[self.P,self.sigmaP,self.lamb_fit,self.fwhm,self.state]]) , columns=['P_'+self.name,'sigma_P_'+self.name,'lambda_'+self.name,'fwhm_'+self.name,"State_"+self.name]),self.study_add],axis=1)
    
    def Clear(self,c=None):
        self.study.loc[:, :] = c
        self.bit_model=False
        self.model=None
        self.bit_fit=False
        self.lamb_fit=None
        self.indexX=None
        self.fit="Fit Non effectué"
    
    #MODIFICATION DE PIQUE
    def Update_Fit(self,crt,ampH,coef_spe=None,sigma=None,inter=None,model_fit=None,Delta_ctr=None):
        for i in range(len(self.pics)):
            if len(self.deltaP0i[i]) >2:
                if sigma is None:
                    sigma= self.pics[0].sigma[0]
                sigma=sigma*self.deltaP0i[i][2]
            self.pics[i].Update(ctr=crt+self.deltaP0i[i][0],ampH=ampH*self.deltaP0i[i][1],coef_spe=coef_spe,sigma=sigma,inter=inter,model_fit=model_fit,Delta_ctr=Delta_ctr)
            if i == 0:
                    self.model = self.pics[i].model
            else:
                self.model = self.model + self.pics[i].model

    def Update_model(self):
        for i in range(len(self.pics)):
            if i == 0:
                    self.model = self.pics[i].model
            else:
                self.model = self.model + self.pics[i].model

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

"""------------------------------------- CLASSE ELEMENT SPECTRE -------------------------------------"""      
class Spectre:
    def __init__(self,wnb,spec,Gauges=[],type_filtre="svg",param_f=[9,2],deg_baseline=0,E=None): #lambda0_s=None,lambda0_r=None,lambda0_SrFCl = None,Temperture=False,Model="psdV",pic
        self.wnb=np.array(wnb)
        self.spec=spec
        self.spec_brut=spec
        self.param_f=param_f
        self.deg_baseline=deg_baseline
        self.type_filtre=type_filtre
        self.y_filtre,self.blfit=None,None
        self.x_corr=wnb
        self.Data_treatement(print_data=False)
        self.E=E
        self.X=None
        self.Y=None
        self.dY=None
        self.bit_model=False
        self.model=None
        self.fit="Fit Non effectué"
        self.bit_fit=False
        self.lamb_fit=None
        self.indexX=None
        #FIT PIC
        self.Gauges=Gauges
        self.lambda_error=round((self.wnb[-1]-self.wnb[0])*0.5/len(self.wnb),4)
        #SYNTHESE
        self.study=pd.DataFrame()
        self.help="Spectre: etude de spectre"
    
    def Corr(self,list_lamb0):
        for i in range(len(self.Gauges)):
            if list_lamb0[i] !=None :
                self.Gauges.lamb0=list_lamb0[i]
            self.Gauges[i].Calcul(input_spe=self.Gauges,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)

    def Print(self,ax=None,ax2=None,return_fig=False):
        if ax == None:
            print_fig=True
            fig, (ax,ax2) =plt.subplots(ncols=1,nrows=2,figsize=(8,4),gridspec_kw={'height_ratios': [0.85, 0.15]})      
        else:
            print_fig=False
        ax.plot(self.x_corr,self.blfit,'-.', c='g', markersize=1) #label="Baseline"
        ax.plot(self.wnb,self.spec,'-',color='lightgray',markersize=4) #,label='Data Brut'
        ax.plot(self.x_corr,self.y_corr+self.blfit,'.',color='black',markersize=3) #,label='Data Fit + Baseline'
        
        for G in self.Gauges:
            if G.bit_fit ==True:
                titre_fiti= G.name+":$\lambda_0=$"+str(G.lamb0)
                if G.indexX is None :
                    bf= self.blfit
                else:
                    bf=self.blfit[G.indexX]
                if G.color_print[0] != None:
                    ax.plot(G.X,G.Y,'--',label=titre_fiti,markersize=1,c=G.color_print[0])
                    if ax2 is not None:
                        ax2.plot(G.X,G.dY/max(np.abs(G.dY)),'-',c=G.color_print[0])
                else:
                    ax.plot(G.X,G.Y,'--',label=titre_fiti,markersize=1)
                    if ax2 is not None:
                        ax2.plot(G.X,G.dY/max(np.abs(G.dY)),'-')
                for i,pic in enumerate(G.pics):
                    if pic.best_fit is not None:
                        y_p=pic.best_fit[G.indexX] +bf
                    if G.color_print[1] != None:
                        titre_pic= rf" $p_{i+1}^{(G.name[0])}= {round(pic.ctr[0],3)}$" #fit.best_values[pic[1].name+'center']
                        ax.fill_between(G.X, y_p, bf, where=y_p>min(y_p), alpha=0.3, label=titre_pic,color=G.color_print[1][i])
                    else:
                        ax.fill_between(G.X, y_p, bf, where=y_p>min(y_p), alpha=0.1)
       
        ax.minorticks_on()
        ax.tick_params(which='major', length=10, width=1.5, direction='in')
        ax.tick_params(which='minor', length=5, width=1.5, direction='in')
        ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)
        ax.set_title(f'$Spectre,\Delta\lambda=$'+ str(self.lambda_error))
        ax.set_ylabel('Amplitude (U.A.)')
        ax.set_xlim([min(self.x_corr),max(self.x_corr)])
        ax.ticklabel_format(axis="y",style="sci",scilimits=(0, 0))
        if ax2 is not None:
            ax2.axhline(0,color="k",ls='-.')
            ax2.minorticks_on()
            ax2.tick_params(which='major', length=10, width=1.5, direction='in')
            ax2.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax2.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)

            ax2.set_xlabel(f'$\lambda$ (nm)')
            ax2.set_ylabel(f'$(Data-Fit)/max (U.A.)$')
            ax2.set_xlim([min(self.x_corr),max(self.x_corr)])
            ax2.ticklabel_format(axis="y",style="sci",scilimits=(0, 0))
        else:
            ax.set_xlabel(f'$\lambda$ (nm)')
        #ax.yaxis.get_offset_text().set_fontsize(10)
        ax.legend(loc="best")
        if return_fig is True:
            return fig
        else:
            if print_fig is True:
                plt.show() 
            else:
                return ax
    
    def FIT_One_Jauge(self,num_jauge=0,peakMax0=None,wnb_range=3,coef_spe=None,sigma=None,inter=None,model_fit=None,manuel=False,model_jauge=None,Delta_ctr=None):
        G=self.Gauges[num_jauge]
        y_sub=self.y_corr
        """
        for other_G in self.Gauges:
            if other_G.name != G.name and G.state =="Y":
                param=other_G.model.make_params()
                y_sub=y_sub - other_G.model.eval(param,x=self.wnb)
        """
        if (peakMax0 is not None) and (model_jauge is None):
            G=self.Gauges[num_jauge]
            peakMax=peakMax0


        elif model_jauge is not None:
            G = model_jauge[num_jauge]
            wnb_range_model=(self.wnb[G.indexX][-1]-self.wnb[G.indexX][0])/2
            if  ("Lw" and "Hg") not in G.name_spe:
                if wnb_range_model < self.lambda_error*10:
                    wnb_range=self.lambda_error*10
                elif wnb_range_model <= wnb_range:
                    wnb_range=wnb_range_model+self.lambda_error
            peakMax=G.lamb_fit
        else:
            G=self.Gauges[num_jauge]
            peakMax=G.lamb0
        
        dpic=[dp[0] for dp in G.deltaP0i]
        
        if ("Lw" and "Hg") in G.name_spe :
            match = re.search("Lw(\d+)", G.name_spe)
            Dwnb_low= float(match.group(1))
            match = re.search("Hg(\d+)", G.name_spe)#float(G.name_spe[1:3]) avec l aps Low
            Dwnb_hight= float(match.group(1)) #float(G.name_spe[4:6]) avce H pas Hight 
            G.indexX=np.where((self.wnb > (peakMax-Dwnb_low)) & (self.wnb < (peakMax+Dwnb_hight)))[0]
            wnb_range=Dwnb_hight+Dwnb_low
        else:
            G.indexX=np.where((self.wnb > peakMax-(wnb_range+abs(min(min(dpic),0)))) & (self.wnb < peakMax+(wnb_range+max(max(dpic),0))))[0]

        x_sub = np.array(self.wnb[G.indexX])
        y_sub=np.array(y_sub[G.indexX])

        if Delta_ctr is None:
            Delta_ctr=wnb_range/10
        
        if manuel == False:
            indexX=np.where((x_sub > peakMax-Delta_ctr*2) & (x_sub < peakMax+Delta_ctr*2))[0]
            x_max=x_sub[indexX]
            y_max=y_sub[indexX]
            peakMax = x_max[np.argmax(y_max)]
            ampMax = np.max(y_max)
            
        else:
            i0=np.argmin(abs(peakMax0-x_sub))
            peakMax =x_sub[i0]
            ampMax=y_sub[i0]
        
        G.Update_Fit(crt=peakMax,ampH=ampMax,coef_spe=coef_spe,sigma=sigma,inter=inter,model_fit=model_fit,Delta_ctr=Delta_ctr)


        G.fit=G.model.fit(y_sub, x=x_sub)
        G.model = G.fit.model
        G.Y= G.fit.best_fit + self.blfit[G.indexX]
        G.dY= G.fit.best_fit - y_sub
        G.X=x_sub
        G.lamb_fit =G.fit.best_values[G.name+'_p1center']
        G.bit_fit=True
        #print(G.fit.fit_report())
        if "DRX" in G.name_spe :      
            G.pic=[G.fit.best_values[G.name + '_p'+str(i+1)+'center'] for i in range(G.nb_pic) ]
        
        for p in G.pics:
            new_param=p.Out_model(out=G.fit)
            p.Update(ctr=float(new_param[0]),ampH=float(new_param[1]),coef_spe=new_param[3],sigma=float(new_param[2]))     
            param=p.model.make_params()
            p.best_fit=p.model.eval(param,x=self.wnb)

        self.Gauges[num_jauge]=G
    
    def FIT(self,wnb_range=2,coef_spe=None,sigma=None,inter=None,model_fit=None,model_jauge=None):
        for i,G in enumerate(self.Gauges):
            if G.state == "Y":
                try:
                    self.FIT_One_Jauge(num_jauge=i,peakMax0=G.lamb_fit,wnb_range=wnb_range,coef_spe=coef_spe,sigma=sigma,inter=inter,model_fit=model_fit,model_jauge=model_jauge)
                except Exception as e:
                    G.state="IN_NOISE"
                    print("error:",e,"in fit of :",G.name)
            G.bit_fit=True
        for G in self.Gauges:
            G.Calcul(input_spe=self.Gauges,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)
        self.bit_fit=True
    
    def FIT_Curv(self,inter=1):
        list_F=[]
        initial_guess=[]
        bounds_min,bounds_max=[],[]
        x_min, x_max=float(self.Gauges[0].lamb0),float(self.Gauges[0].lamb0)
        for j,G in enumerate(self.Gauges):
            for i,p in enumerate(G.pics):
                x_min,x_max=min(x_min,p.ctr[0]-p.sigma[0]*5),max(x_max,p.ctr[0]+p.sigma[0]*5)
                list_F.append(p.f_model)
                initial_guess+= [p.ctr[0],p.ampH[0],p.sigma[0]]
                for c in p.coef_spe:
                    initial_guess+=[c[0]]
                bounds_min+=[p.ctr[1][0],p.ampH[1][0],p.sigma[1][0]]
                bounds_max+=[p.ctr[1][1],p.ampH[1][1],p.sigma[1][1]]
                for c in p.coef_spe:
                    bounds_min+=[c[1][0]]
                    bounds_max+=[c[1][1]]
            G.Update_model()
        bounds=[bounds_min,bounds_max]

        for i,G in enumerate(self.Gauges):
            if i ==0:
                self.model = G.model
            else:
                self.model+=G.model
       
        self.Data_treatement()

        self.indexX=np.where((self.wnb >= x_min) & (self.wnb <= x_max))[0]
        x_corr=self.wnb[self.indexX]
        y_sub = self.y_corr[self.indexX]
        blfit = self.blfit[self.indexX]
        sum_function = Gen_sum_F(list_F)
        params , params_covar = curve_fit(sum_function,x_corr,y_sub,p0=initial_guess,bounds=bounds)
        fit=sum_function(x_corr,*params)
        self.Y= fit +  blfit
        self.X=x_corr
        self.dY= y_sub-fit 
        self.lamb_fit =params[0]
        ij_3,ij_4,ij_5=0,0,0
        params_list=list(params)

        for i, J in enumerate(self.Gauges):
            for j , p in enumerate(J.pics):
                n_c=len(p.coef_spe)
                start_idx = 3 * ij_3 + 4 * ij_4 + 5 * ij_5
                end_idx = start_idx + 3
                if n_c == 0:
                    params_pic = params_list[start_idx:end_idx] #list(params[start_idx:end_idx]) 
                    ij_3 += 1
                elif n_c == 1:
                    params_pic = params_list[start_idx:end_idx] + [np.array([params_list[end_idx]])]#list(params[start_idx:end_idx]) + list(np.array(params[end_idx]))
                    ij_4 += 1  
                elif n_c == 2:
                    params_pic = params_list[start_idx:end_idx] + [np.array(params_list[end_idx:end_idx+2])] #list(params[start_idx:end_idx]) + list(np.array(params[end_idx:end_idx+2]))
                    ij_5 += 1
                p.Update(ctr=float(params_pic[0]),ampH=float(params_pic[1]),coef_spe=params_pic[3],sigma=float(params_pic[2]),inter=float(inter))
                param=p.model.make_params()
                p.best_fit=p.model.eval(param,x=self.wnb)
            if j==0:
                J.lamb_fit=params_pic[0]
            param=J.model.make_params()
            J.Y=p.model.eval(param,x=self.wnb)
            J.X=self.wnb
            J.dY=J.Y-self.y_corr
            J.bit_fit=True

        for G in self.Gauges:
            G.Calcul(input_spe=self.Gauges,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)
        self.bit_fit=True

    def Clear_study(self,num_jauge):
        self.Gauges[num_jauge].study.loc[:, :] = None

    def Calcul_study(self,mini=False):
        #self.lambda_error=round((self.wnb[0]-self.wnb[-1])*0.5/len(self.wnb),4)
        for i in range(len(self.Gauges)):
            if ("DRX" in self.Gauges[i].name_spe) and (self.bit_fit is True): #
                self.Gauges[i].bit_fit =True
            self.Gauges[i].Calcul(input_spe=self.Gauges,mini=mini,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)  
    
    def Data_treatement(self,deg_baseline=None,type_filtre=None,param_f=None,print_data=False,ax=None,ax2=None):
        if deg_baseline is not None:
            self.deg_baseline=deg_baseline
        if param_f is not None:
            self.param_f=param_f
        self.blfit = pk.baseline(self.spec, deg=self.deg_baseline) #retrait du dark
        if self.deg_baseline ==0:
            deltaBG=min(self.spec)-self.blfit[0]
            if  deltaBG <0 :
                self.blfit =np.array(self.blfit) + deltaBG*1.05
        
        if type_filtre is not None:
            self.type_filtre=type_filtre

        if "svg" == self.type_filtre: # Appliquer un filtre de Savitzky-Golay pour lisser le spectre
            self.y_filtre = savgol_filter(self.spec,window_length=self.param_f[0],polyorder=self.param_f[1])
        elif "fft" == self.type_filtre:
            # Transformée de Fourier du signal
            spectre_fft = np.fft.fft(self.spec)
            # Fréquences associées
            frequences_fft_brut = np.fft.fftfreq(len(self.spec), d=self.wnb[1]-self.wnb[0])
            # Filtrage en supprimant les basses fréquences
            cutoff_low = self.param_f[0]  # Fréquence de coupure inférieure
            cutoff_high = self.param_f[1] # Fréquence de coupure supérieure
            # Supprimer les fréquences indésirables dans cette plage
            spectre_fft_brut=copy.deepcopy(spectre_fft)
            spectre_fft[(np.abs(frequences_fft_brut) > cutoff_low) & (np.abs(frequences_fft_brut) < cutoff_high)] = 0
            # Transformée de Fourier inverse pour ré
            self.y_filtre = np.real(np.fft.ifft(spectre_fft))
        else:
            self.y_filtre=self.spec
        self.y_corr = self.y_filtre - self.blfit

        if print_data is True:
            if ax == None:
                figure=False
                fig, (ax,ax2) =plt.subplots(ncols=1,nrows=2,figsize=(8,4),gridspec_kw={'height_ratios': [0.7, 0.3]})      
            else:
                figure=True

            if "fft" != self.type_filtre:
                # Transformée de Fourier du signal
                spectre_fft_brut = np.fft.fft(self.spec)
                spectre_fft_fit = np.fft.fft(self.y_corr)
                # Fréquences associées
                frequences_fft_brut = np.fft.fftfreq(len(self.spec), d=self.wnb[1]-self.wnb[0])
                frequences_fft_fit = np.fft.fftfreq(len(self.y_corr), d=self.x_corr[1]-self.x_corr[0])
                ax2.plot(np.abs(frequences_fft_fit), np.abs(spectre_fft_fit),'-.g', label='Data_fit')
            else:
                ax2.fill_between([self.param_f[0],self.param_f[1]],min(np.abs(spectre_fft)),max(np.abs(spectre_fft)) , color="red", alpha=0.2,label="freq filtré")
            
            ax.plot(self.x_corr,self.blfit,'-.', c='g', markersize=1,label="Bkg")
            ax.plot(self.wnb,self.spec,'-',color='gray',markersize=4,label='Brut')
            ax.plot(self.x_corr,self.y_corr+self.blfit,'-.+',color='black',markersize=3,label='Corr + bkg')
            ax.minorticks_on()
            ax.tick_params(which='major', length=10, width=1.5, direction='in')
            ax.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)
            ax.set_title(f'$Spectre \Delta\lambda=$'+ str(self.lambda_error))
            ax.set_xlabel(f'$\lambda$ (nm)')
            ax.set_ylabel('U.A.')
            ax.set_xlim([min(self.x_corr),max(self.x_corr)])
            ax.ticklabel_format(axis="y",style="sci",scilimits=(0, 0))
            ax2.plot(np.abs(frequences_fft_brut), np.abs(spectre_fft_brut),'-.k', label='Data_brut')
            
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.tick_params(which='major', length=10, width=1.5, direction='in')
            ax2.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax2.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)
            ax2.set_xlabel(f'$f$ (Hz)')
            ax2.set_ylabel('Amplitude (u.a.)')
            ax2.set_title('FFT')
            ax2.legend(loc="best")
            #ax2.set_xlim([min(np.abs(frequences_fft)),max(np.abs(frequences_fft))])
            if figure is False:
                plt.show() 
            else:
                return ax,ax2
    
    """------------------------------------- CLASSE BANC CED DYNAMIQUE -------------------------------------"""

class CEDd:
    def __init__(self,data,Gauges_init,N=None,data_Oscillo=None,folder_Movie=None,time_index=[2,4],fps=None,fit=False,skiprow_spec=43,reload=False,type_filtre="svg",param_f=[9,2],Kinetic=False,window=True):
        if reload == False:
            self.Kinetic=Kinetic
            """------------------------------------- SPECTRE -------------------------------------"""
            self.data_Spectres= pd.read_csv(data, sep='\s+',header=None, skipfooter=0,skiprows=skiprow_spec, engine='python')
            print(len(self.data_Spectres.columns) ==2)
            if len(self.data_Spectres.columns) ==2:
                wave=self.data_Spectres.iloc[:,0]
                Iua=self.data_Spectres.iloc[:,1]
                wave_unique=np.unique(wave)
                num_spec = len(wave)//len(wave_unique)
                print(num_spec)
                if num_spec >=1:
                    Iua=Iua.values.reshape(num_spec,len(wave_unique)).T
                    self.data_Spectres=pd.DataFrame(np.column_stack([wave_unique,Iua]),columns=[0]+ [i+1 for i in range(num_spec)])
            self.Spectra=[]
            self.Gauges_init=Gauges_init
            self.N=N
            self.Summary=pd.DataFrame()
            self.list_lamb0 =[J.lamb0 for J in self.Gauges_init]
            self.list_nspec=[]
            """------------------------------------- OSCILLOSCOPE -------------------------------------"""
            self.time_index=time_index
            self.data_Oscillo=data_Oscillo
            if self.data_Oscillo != None:
                self.data_Oscillo = pd.read_csv(self.data_Oscillo, sep='\s+', skipfooter=0, engine='python')
            self.Time_spectrum=None
            """------------------------------------- MOVIE -------------------------------------"""
            self.folder_Movie=folder_Movie
            self.Movie=None
            self.list_Movie=None
            self.time_movie=[]
            self.fps=fps
            self.CEDd_path="not_save"
            self.Gauges_select=[None for _ in range(len(self.Gauges_init))]
            self.initData(fit,time_index,type_filtre,param_f,window)
          
        else:
            self.Kinetic=Kinetic
            " A TOUT REFAIR POUR CHANGER LES ANCIEN DONNER"
            """SPECTRE"""
            self.data_Spectres= data.data_Spectres
            self.Spectra=data.Spectra

            if type(data.Gauges_init) is not str :
                self.Gauges_init=data.Gauges_init
            else:
                self.Gauges_init=Gauges_init
            self.N=data.N
            self.Summary=data.Summary
            self.list_lamb0 =[J.lamb0 for J in self.Gauges_init]
            self.list_nspec=data.list_nspec
            """OSCILO"""
            self.time_index=data.time_index
            self.data_Oscillo=data.data_Oscillo
            self.Time_spectrum=data.Time_spectrum
            self.t0_movie=0
            """ MOVIE"""
            self.folder_Movie=folder_Movie
            self.Movie=data.Movie
            self.list_Movie=data.list_Movie
            self.time_movie=data.time_movie
            self.fps=data.fps
            self.initData(fit,time_index,param_f)

        self.help="CEDd: Etude d'experience de CEDd spectro/Movie prochainement dif X A CODER"
    
    def initData(self,fit=False,time_index="Channel2",type_filtre="svg",param_f=[9,2],window=True):
        if self.N ==None:
            self.N=len(np.array(self.data_Spectres.drop(columns=[0]))[1])
        if fit == True:
            new_jauges=copy.deepcopy(self.Gauges_init)
            last_check=time.time()
            
            for i in tqdm(range(0,self.N), desc="Progress"):
                t_i=time.time()            
                new_spec=Spectre(self.data_Spectres[0],self.data_Spectres.drop(columns=[0])[i+1],new_jauges,param_f=param_f,type_filtre=type_filtre)
                try:
                    new_spec.FIT()
                    stu=new_spec.study
                    new_jauges=copy.deepcopy(new_spec.Gauges)
                except Exception as e :
                    print("ERROR:",e, "\n Spec n°:",str(i))
                    stu=self.Spectra[-2].study
                    new_jauges=copy.deepcopy(self.Spectra[-2].Gauges)
                    new_spec.Gauges=new_jauges
                    new_spec.study=stu
                self.Summary=pd.concat([self.Summary,pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),stu],ignore_index=False,axis=1)],ignore_index=True)
                            
                if self.Kinetic == False or i%10==0 or i<2:
                    self.Spectra.append(new_spec)
                    self.list_nspec.append(int(i))
                

                if window is True:
                    t=time.time()
                    if i ==1 or  t-last_check >30 or t-t_i>1: #(i/self.N)%0.1==0:
                    
                        figs=[spec.Print(return_fig=True) for spec in self.Spectra[i-1:i+1]]
                        dlg = ProgressDialog(figs,value=int((i/self.N)*100),Gauges=self.Spectra[i].Gauges)

                        result=dlg.exec_()

                        result,param,inter=dlg.get_user_choice()
                        last_check=time.time()
                        
                        if result:
                            for i,p_G in enumerate(param):
                                for j,p_p in enumerate(p_G):
                                    print(p_p)
                                    new_jauges[i].pics[j].Update(ctr=p_p[0],sigma=p_p[1],inter=inter)
                            continue
                        else:
                            self.list_nspec=[]
                            self.Spectra=[]
                            self.Summary=pd.DataFrame()
                            break
            print("Spectre loading & fit DONE")
        else:
            for i in range(1, self.N+1):
                self.Spectra.append(Spectre(self.data_Spectres[0],self.data_Spectres.drop(columns=[0])[i],self.Gauges_init,param_f=param_f,type_filtre=type_filtre))
                self.list_nspec.append(int(i))
            print("Spectre loading DONE")

        if time_index is not None and self.data_Oscillo is not None:
            self.Extract_time()
    
            
        if self.folder_Movie != None :
            if os.path.isdir(self.folder_Movie):
                self.list_Movie=[os.path.join(self.folder_Movie, name) for name in os.listdir(self.folder_Movie) ]
                self.Movie=[]
                for i in range(len(self.list_Movie)):
                    if os.path.isfile(self.list_Movie[i]):
                        try:
                            self.Movie.append(Image.open(self.list_Movie[i]))
                            self.time_movie.append(i/self.fps+self.t0_movie)
                        except IOError:
                            pass
            if os.path.isfile(self.folder_Movie):
                try:
                    cap = cv2.VideoCapture(self.folder_Movie)
                    self.fps = cap.get(cv2.CAP_PROP_FPS)
                    num_frames = int( cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.time_movie=[i/self.fps+self.t0_movie for i in range(num_frames)]
                except IOError:
                            pass
            print("Movie DONE")

    def FIT(self,end=None,start=0,G=None,wnb_range=3,init=False):
        if G != None:
            self.Gauges_init=G
        elif init==True:
            self.Gauges_init=self.Spectra[start].Gauges
        self.list_nspec=[]
        self.Spectra=[]
        self.Summary=pd.DataFrame()
        if end == None:
            end =self.N
        if end <= self.N :
            new_jauges=copy.deepcopy(self.Gauges_init)
            for i in tqdm(range(start,end), desc="Progress"):
                new_jauges=self.Select_Gauges(i,new_jauges)
                new_spec=Spectre(self.data_Spectres[0],self.data_Spectres.drop(columns=[0])[i+1],new_jauges)                
                new_spec.FIT(wnb_range=wnb_range)
                new_jauges=copy.deepcopy(new_spec.Gauges)
                self.Summary=pd.concat([self.Summary,pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),new_spec.study],ignore_index=False,axis=1)],ignore_index=True)
                if (self.Kinetic == False and i%10==0) or i<2:
                    self.Spectra.append(new_spec)
                    self.list_nspec.append(int(i))
                if i ==1 :
                    figs=[spec.Print(return_fig=True) for spec in self.Spectra]
                    dlg = ConfirmationDialog(figs)
                    result=dlg.exec_()
                    result=dlg.get_user_choice()
                    if result:
                        self.list_nspec=[]
                        self.Spectra=[]
                        self.Summary=pd.DataFrame()
                        break
            print("Fit DONE")
        else:
            print("end > number of spectre !")

    def Select_Gauges(self,x,Gauges):
        for i ,select_G in enumerate(self.Gauges_select):
            if not select_G:
                Gauges[i].state="Y"
                continue
            for intervale in select_G:
                if  intervale[0] <=x  <= intervale[1]:
                    Gauges[i].state="Y"
                    continue
            else:
                Gauges[i].state="IN_NOISE"
        return Gauges
    
    def FIT_Corr(self,end=None,start=0,select_gauge=False,inter=None):
        if end is None:
            end =self.N
        if end <= self.N :
            if self.Kinetic is True:
                new_jauges=copy.deepcopy(self.Gauges_init)
                for i in tqdm(range(start+1,end), desc="Progress"):
                    if select_gauge is True:
                        new_jauges=self.Select_Gauges(i,new_jauges)
                    new_spec=Spectre(self.data_Spectres[0],self.data_Spectres.drop(columns=[0])[i],new_jauges)
                    new_spec.FIT(model_jauge=new_jauges,inter=inter)
                    new_jauges=copy.deepcopy(new_spec.Gauges)
                    self.Summary.iloc[i]=pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),new_spec.study],ignore_index=False,axis=1)
                print("Fit Corr Fast Kinetic DONE")
            else:
                new_jauges=copy.deepcopy(self.Spectra[start].Gauges)
                for i in tqdm(range(start+1,end), desc="Progress"):
                    if select_gauge is True:
                        new_jauges=self.Select_Gauges(i,new_jauges)
                    self.Spectra[i].FIT(model_jauge=new_jauges,inter=inter)
                    new_jauges=copy.deepcopy(self.Spectra[i].Gauges)
                    self.Summary.iloc[i]=pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),self.Spectra[i].study],ignore_index=False,axis=1)
                print("Fit Corr DONE")
        else:
            print("end > number of spectre !")

    def Corr_Summary(self, num_spec=None, All=True, lambda_error=None):
        """
        Met à jour self.Summary à partir des Spectra[i].study.

        - Si All=True : reconstruit entièrement Summary à partir de tous les spectres.
        - Sinon : ne met à jour que la ligne correspondant à num_spec (en créant ou remplaçant).
        """

        def _build_row_df(idx_spec, study):
            """
            Construit un DataFrame 1 ligne avec au minimum la colonne 'n°Spec'.
            Si 'study' est un DataFrame/Series non vide, on le concatène horizontalement.
            """
            base_df = pd.DataFrame({"n°Spec": [int(idx_spec)]})

            if study is None:
                # Rien à ajouter
                return base_df

            # Series -> DataFrame 1 ligne
            if isinstance(study, pd.Series):
                study = study.to_frame().T

            if isinstance(study, pd.DataFrame):
                if study.empty:
                    # Pas de données exploitables
                    return base_df
                study = study.reset_index(drop=True)
                return pd.concat([base_df, study], axis=1)

            # Type inattendu : on log et on ne concatène pas
            print("Corr_Summary: unexpected type for 'study':", type(study))
            return base_df

        # -------------------------
        # CAS 1 : Rebuild complet
        # -------------------------
        if All:
            rows = []  # liste de Series (une par spectre)

            for i in range(len(self.Spectra)):
                spe = self.Spectra[i]

                # Mise à jour des paramètres des jauges depuis Gauges_init
                for j in range(len(self.Gauges_init)):
                    if j < len(spe.Gauges):   # sécurité si nb de jauges a changé
                        spe.Gauges[j].lamb0   = self.Gauges_init[j].lamb0
                        spe.Gauges[j].f_P     = self.Gauges_init[j].f_P
                        spe.Gauges[j].inv_f_P = self.Gauges_init[j].inv_f_P

                if lambda_error is not None:
                    spe.lambda_error = lambda_error

                # Calcul de l'étude pour ce spectre
                try:
                    spe.Calcul_study()
                except Exception as e:
                    print(f"Corr_Summary: erreur dans Calcul_study pour spectre {i}: {e}")
                    spe.study = None

                # Construction de la ligne pour ce spectre
                row_df = _build_row_df(i, getattr(spe, "study", None))
                # row_df est un DF 1 ligne -> on récupère la Series
                rows.append(row_df.iloc[0])

            # On reconstruit complètement Summary à partir des lignes collectées
            if rows:
                self.Summary = pd.DataFrame(rows).reset_index(drop=True)
            else:
                # Aucun spectre exploitable : on crée un Summary vide avec au moins la colonne n°Spec
                self.Summary = pd.DataFrame(columns=["n°Spec"])

            return  # très important : on s'arrête là dans le cas All

        # -------------------------
        # CAS 2 : Mise à jour d'un seul spectre
        # -------------------------
        if num_spec is not None:
            # Sécurité : s'assurer que Summary existe
            if not hasattr(self, "Summary") or self.Summary is None:
                self.Summary = pd.DataFrame()

            # Recherche de la ligne par n°Spec, si la colonne existe
            if "n°Spec" in self.Summary.columns and not self.Summary.empty:
                idx_rows = np.where(np.array(self.Summary["n°Spec"]) == num_spec)[0]
            else:
                idx_rows = []

            # Calcul de study pour ce spectre
            # On suppose que num_spec correspond à l'indice dans self.Spectra (à adapter sinon)
            if 0 <= num_spec < len(self.Spectra):
                spe = self.Spectra[num_spec]
            else:
                print(f"Corr_Summary: num_spec {num_spec} hors limites pour self.Spectra.")
                return

            try:
                spe.Calcul_study()
            except Exception as e:
                print(f"Corr_Summary: erreur dans Calcul_study pour spectre {num_spec}: {e}")
                spe.study = None

            row_df = _build_row_df(num_spec, getattr(spe, "study", None))
            row = row_df.iloc[0]

            if len(idx_rows) == 0:
                # Pas encore de ligne pour ce n°Spec -> on ajoute
                self.Summary = pd.concat(
                    [self.Summary, pd.DataFrame([row])],
                    ignore_index=True
                )
            else:
                # On remplace la ligne existante
                i = idx_rows[0]
                # Si l'index i est hors limites (cas bizarre), on ajoute
                if i >= len(self.Summary):
                    self.Summary = pd.concat(
                        [self.Summary, pd.DataFrame([row])],
                        ignore_index=True
                    )
                else:
                    # on aligne les colonnes pour éviter des KeyError
                    for col in row.index:
                        if col not in self.Summary.columns:
                            self.Summary[col] = np.nan
                    # et on remplace la ligne
                    self.Summary.iloc[i] = row

    def Corr_Movie(self,folder_Movie=None,fps=None):
            if fps != None and folder_Movie==None:
                self.fps=fps
            self.time_movie=[self.t0_moviei/self.fps for i in range(len(self.list_Movie))]
            if folder_Movie!=None:
                self.folder_Movie=folder_Movie
                self.list_Movie=[os.path.join(self.folder_Movie, name) for name in os.listdir(self.folder_Movie) ]
                self.Movie=[]
                for i in range(len(self.list_Movie)):
                    if os.path.isfile(self.list_Movie[i]):
                        try:
                            self.Movie.append(Image.open(self.list_Movie[i]))
                            self.time_movie.append(i/self.fps +self.t0_movie)
                        except IOError:
                            pass
                    if os.path.isfile(self.folder_Movie):
                        try:
                            cap = cv2.VideoCapture(self.folder_Movie)
                            self.fps = cap.get(cv2.CAP_PROP_FPS)
                            num_frames = int( cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            self.time_movie=[i/self.fps+self.t0_movie for i in range(num_frames)]
                        except IOError:
                                    pass

    def Extract_time(self,time_index=None,data_time=None):
        if type(data_time) != type(None):
            self.data_Oscillo = pd.read_csv(data_time, sep='\s+', skipfooter=0, engine='python')
        if time_index:
            self.time_index=time_index

        if self.data_Oscillo is not None:
            c_nam=self.data_Oscillo.columns.tolist()
            temps, signale_spec ,signale_cam=np.array(self.data_Oscillo[c_nam[0]]), np.array(self.data_Oscillo[c_nam[self.time_index[0]]]),np.array(self.data_Oscillo[c_nam[self.time_index[1]]])

            marche_cam=max(signale_cam)/2
            if marche_cam <0.5:
                self.t0_movie=0
            else:
                self.t0_movie=temps[np.where(signale_cam > marche_cam)[0][0]]
            print(f"self.t0_movie={self.t0_movie*1e3}ms")
            marche_spec=max(signale_spec)/2
            signale_spec_bit=(signale_spec >=marche_spec).astype(int)
            front=np.diff(signale_spec_bit)
            self.Time_spectrum= (temps[np.where(front >=0.9)]+temps[np.where(front <= -0.9)])/2 #[(temps[up]+temps[down])/2 for up,down in zip(np.where(front >=0.9),np.where(front <= -0.9))]
            print(f"self.Time_spectre len={len(self.Time_spectrum)} LOAD")
        else:
            print("CALCULE FAILED NO DATA")

    def Print(self,num_spec=0,data=[],Oscilo=False):
        if data ==[] and Oscilo==False:
            self.Spectra[num_spec].Print()
        else:
            fig,(ax1,ax2) =plt.subplots(1,2)
            for i in range(len(data)):
                index=np.where(np.array(self.Summary[data[i]]) != None)[0]
                ax1.plot(self.Summary["n°Spec"][index],self.Summary[data[i]][index],'+-',label=data[i])
            if self.data_Oscillo is not None and Oscilo is True:
                c_nam=self.data_Oscillo.columns.tolist()
                for name in c_nam:
                    ax2.plot(self.data_Oscillo[c_nam[0]],self.data_Oscillo[name],label=name)
                y=np.zeros_like(self.Time_spectrum)+1
                ax2.plot(self.Time_spectrum,y,' k',marker=2,label="spec_time")
                ax2.axvline(self.t0_movie,label="t0_movie")
            ax1.legend(loc="best")
            ax1.set_title("Summary print")
            ax2.legend(loc="best")
            ax2.set_title("Oscilo print")
            plt.show()

    def Play_Movie(self):
        if self.folder_Movie is None:
            return print("No Folder Movie !")
        # Affichage initial
        fig, ax = plt.subplots()
        # Lecture vidéo avec OpenCV
        cap = cv2.VideoCapture(self.folder_Movie)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Texte pour afficher le temps et le numéro de frame
        info_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color="white", fontsize=12, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

        # Fonction pour mettre à jour le texte du temps et le numéro de la frame
        def update_info_text(frame_number):
            time_in_seconds =(frame_number / fps + self.t0_movie)*1e3
            info_text.set_text(f"time : {time_in_seconds:.3f} ms | frame : {frame_number}/{num_frames - 1}")

        # Fonction pour lire une image donnée de la vidéo
        def read_frame(frame_number):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            else:
                return None

        # Initialisation de la première image
        current_frame = 0
        img = read_frame(current_frame)

        
        image = ax.imshow(img,vmax=np.max(img),vmin=np.min(img))

        # Fonction de mise à jour pour slider
        def update(val):
            global current_frame
            current_frame = int(val)
            img = read_frame(current_frame)
            if img is not None:
                image.set_data(img)
                image.set_clim(vmax=np.max(img),vmin=np.min(img))
                update_info_text(current_frame)
            plt.draw()

        # Fonction pour aller à l'image suivante
        def next_frame(event):
            global current_frame
            if current_frame < num_frames - 1:
                current_frame += 1
                slider.set_val(current_frame)

        # Fonction pour aller à l'image précédente
        def prev_frame(event):
            global current_frame
            if current_frame > 0:
                current_frame -= 1
                slider.set_val(current_frame)

        # Barre de défilement pour choisir les images
        slider_ax = plt.axes([0.150, 0.95, 0.7, 0.03])
        slider = Slider(ax=slider_ax, label=f'kfps:{int(fps*1e-3)}', valmin=0, valmax=num_frames - 1, valinit=0, valstep=1,facecolor="#0099cc",track_color="gray")
        slider.on_changed(update)

        # SUIVANT
        next_button = plt.axes([0.6, 0.9, 0.15, 0.04])
        next_button = Button(next_button,"\u2192",color="#0099cc",hovercolor="#0A5DA5")
        next_button.label.set_fontsize(30)
        next_button.on_clicked(next_frame)
        #PRECEDENT
        previous_button = plt.axes([0.3, 0.9, 0.15, 0.04])
        previous_button = Button(previous_button,"\u2190",color="#0099cc",hovercolor="#0A5DA5")
        previous_button.label.set_fontsize(30)
        previous_button.on_clicked(prev_frame)
        update(0)

        plt.show()
        cap.release()

    def Help(self):
        print("sorry")

""" ------------------------------------- CLASS OSCILOSCOPE LECROY-------------------------------------"""
class Oscillo_autosave:
    def __init__(self,IP="100.100.143.2",folder=r"F:\Aquisition_Banc_CEDd\Aquisition_LECROY_Banc_CEDd"):
        self.scope = lecroyscope.Scope(IP)  # IP address of the scope
        self.folder=folder
        print(f"Scope ID: {self.scope.id}")
        print("dossier d'enregistrement"+self.folder)

    def Print(self):
        # Afficher les nouvelles traces sur le graphique
        trace_group = self.scope.read(1, 2, 3, 4)
        for i in range(1, len(trace_group) + 1):
            plt.plot(trace_group[i].x, trace_group[i].y, '.-')
        plt.xlabel('Time')
        plt.ylabel(f'Channel {i}')
        plt.title(f'Trace for Channel {i}')
        plt.grid(True)
        plt.show()


    def save(self,name):
        trace_group = self.scope.read(1, 2, 3, 4)
        time = trace_group.time  # time values are the same for all traces
        df = pd.DataFrame({"Time" :pd.Series(time), 
                   "Channel1" :pd.Series(trace_group[1].y),
                   "Channel2" :pd.Series(trace_group[2].y),
                   "Channel3" :pd.Series(trace_group[3].y),
                   "Channel4" :pd.Series(trace_group[4].y),
                  })
        file_path =os.path.join(self.folder,name)
        if file_path:
            with open(file_path, 'w') as file2write:
                file2write.write(df.to_string())
            print(f"Data saved to {file_path}")
