import re

import numpy as np
import pandas as pd
from pynverse import inversefunc

from Bibli_python.CL_FD_Update import Pics
from Bibli_python.Physic_law import (
    Raman_Dc12__2003_Occelli,
    Raman_Dc13_1997_Schieferl,
    Rhodamine_6G_2024_Dembele,
    Ruby_1986_Mao,
    Ruby_2020_Shen,
    Sm_1997_Datchi,
    Sm_2015_Rashenko,
    SrFCl,
    T_Ruby_Sm_1997_Datchi,
)


class Gauge:
    def __init__(self, name="", lamb0=None, nb_pic=None, X=None, Y=None, deltaP0i=None, spe=None, f_P=None, name_spe=""):
        if deltaP0i is None:
            deltaP0i = []

        self.lamb0 = lamb0
        self.nb_pic = nb_pic
        self.deltaP0i = deltaP0i
        self.pics = []
        self.name = name
        self.X = X
        self.Y = Y
        self.dY = None
        self.spe = spe

        self.f_P = f_P
        if f_P is not None:
            self.inv_f_P = inversefunc((lambda x: f_P(x)), domain=[10, 1000])
        else:
            self.inv_f_P = None
        self.P = 0

        self.T = 0
        self.name_spe = name_spe

        self.bit_model = False
        self.model = None
        self.color_print = [None, None]
        self.bit_fit = False
        self.lamb_fit = None
        self.indexX = None
        self.fit = "Fit Non effectué"
        self.study = pd.DataFrame()
        self.study_add = pd.DataFrame()

        self.state = "Y"  # Y, IN_NOISE , N 1,0,None
        self.Jauge_choice = ["Ruby", "Sm", "SrFCl", "Rhodamine6G"]

        self.help = (
            "Gauge: G de spectroscopie (Ruby,Samarium,SrFCL) prochainement jauge X A CODER "
            "\n #Calib1_251024_16h39_d11_0411_26 Coefficients ajustés X=x_lamb0 et P=aX**2+bX+c: a = 0, "
            "b = 0.04165619359945429,,lambda0 = 551.3915961705188 "
        )

        if "Ruby" in self.name:
            self.Ruby()
        elif ("Sm" in self.name) or ("Samarium" in self.name):
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
        self.pics = []
        for i in range(self.nb_pic):
            if "Sig" in self.name_spe:
                match = re.search("Sig(\\d+)", self.name_spe)
                sigma = float(match.group(1))
            else:
                sigma = 0.25
            if "Mod:" in self.name_spe:
                match = re.search(r"Mod:(\\w+)", self.name_spe)
                model_fit = str(match.group(1))
            else:
                model_fit = "PseudoVoigt"
            new_pics = Pics(
                name=self.name + '_p' + str(i + 1),
                ctr=self.lamb0 + self.deltaP0i[i][0],
                sigma=sigma,
                model_fit=model_fit,
            )
            self.pics.append(new_pics)
            if i == 0:
                self.model = new_pics.model
            else:
                self.model = self.model + new_pics.model

        self.bit_model = True

    #JAUGE PREREMPLIS
    def Ruby(self):
        self.lamb0 = 694.4
        self.nb_pic = 2
        self.name = "Ruby"
        self.name_spe = "Ru"
        self.deltaP0i = [[0, 1], [-1.5, 0.75]]
        self.color_print = ['darkred', ['darkred', 'brown']]

        if self.f_P is None:
            self.f_P = Ruby_2020_Shen
            self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[690, 750])

        self.Init_perso()

    def Calcul_Ruby(self, input_spe):
        if 'RuSmT' in self.name_spe:
            if input_spe[self.spe].fit.params[input_spe[self.spe].name + '_p1center'].stderr is not None:
                sigmals = input_spe[self.spe].fit.params[input_spe[self.spe].name + '_p1center'].stderr
            else:
                sigmals = 0

            if self.fit.params[self.name + '_p1center'].stderr is not None:
                sigmal = self.fit.params[self.name + '_p1center'].stderr
            else:
                sigmal = 0

            T, sigmaT = T_Ruby_Sm_1997_Datchi(
                self.lamb_fit,
                input_spe[self.spe].lamb_fit,
                lamb0S=input_spe[self.spe].lamb0,
                lamb0R=self.lamb0,
                sigmalambda=[sigmal, sigmals],
            )
            self.study_add = pd.DataFrame(np.array([[self.deltaP0i[1][0], T, sigmaT]]), columns=['Deltap12', 'T', 'sigmaT'])
        elif "RuT" in self.name_spe:
            T, sigmaT = T_Ruby_Sm_1997_Datchi(
                self.lamb_fit,
                lamb0R=self.lamb0,
                sigmalambda=[sigmal, input_spe[self.spe].sigmaP],
                P=input_spe[self.spe].P,
            )
            self.study_add = pd.DataFrame(np.array([[self.deltaP0i[1][0], T, sigmaT]]), columns=['Deltap12', 'T', 'sigmaT'])
        else:
            self.study_add = pd.DataFrame(np.array([self.deltaP0i[1][0]]), columns=['Deltap12'])

    def Sm(self, all=False):
        self.lamb0 = 685.39
        self.nb_pic = 1
        self.name = "Sm"
        if all is False:
            self.deltaP0i = [[0, 1]]
        else:
            print("to code")

        if self.f_P is None:
            self.f_P = Sm_2015_Rashenko
            self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[683, 730])
        self.color_print = ['darkblue', ['darkblue']]
        self.Init_perso()

    def Diamond(self, isotope='c12'):
        if isotope == 'c12':
            self.lamb0 = 1333
            self.f_P = Raman_Dc12__2003_Occelli
            self.color_print = ['silver', ['silver']]
        elif isotope == 'c13':
            self.lamb0 = 1287
            self.f_P = Raman_Dc13_1997_Schieferl
            self.color_print = ['dimgrey', ['dimgrey']]
        self.nb_pic = 1
        self.name_spe = "Sig20"
        self.name = "D_" + isotope
        self.deltaP0i = [[0, 1]]
        self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[self.lamb0 - 10, self.lamb0 + 200])

        self.Init_perso()

    def SrFCl(self, all=False):
        self.lamb0 = 685.39
        self.nb_pic = 1
        self.name = "SrFCl"
        if all is False:
            self.deltaP0i = [[0, 1]]
        else:
            print("to code")
        if self.f_P is None:
            self.f_P = SrFCl
            self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[683, 730])

        self.color_print = ['darkorange', ['darkorange']]
        self.Init_perso()

    def Rhodamine6G(self):
        self.lamb0 = 550
        self.nb_pic = 2
        self.name = "Rhodamine6G"
        self.deltaP0i = [[0, 1], [52, 0.5, 4]]
        self.color_print = ['limegreen', ['limegreen', "darkgreen"]]
        self.name_spe = "Lw50Hg150Sig20Mod:Gaussian"

        if self.f_P is None:
            self.f_P = Rhodamine_6G_2024_Dembele
            self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[530, 600])

        self.Init_perso()

    def Calcul(self, input_spe=None, mini=False, lambda_error=0):
        if self.bit_fit is False:
            return print("NO FIT of ", self.name, "do it !")
        else:
            if "DRX" in self.name_spe:
                if self.state == "IN_NOISE":
                    self.a, self.b, self.c, self.rca, self.V, self.P = self.Element_ref.A, self.Element_ref.B, self.Element_ref.C, self.Element_ref.RCA, self.Element_ref.AV0, 0
                else:
                    self.CALCUL(mini=mini)
                self.study = pd.concat(
                    [
                        pd.DataFrame(
                            np.array([[self.a, self.b, self.c, self.rca, self.V, self.P]]),
                            columns=[
                                'a_' + self.name,
                                'b_' + self.name,
                                'c_' + self.name,
                                'c/a_' + self.name,
                                'V_' + self.name,
                                'P_' + self.name,
                            ],
                        ),
                        self.study_add,
                    ],
                    axis=1,
                )
                return print("CODER DRX")

            self.lamb_fit = round(self.pics[0].ctr[0], 3)

            sigmal = lambda_error
            if self.state == "Y":
                if hasattr(self.fit, "params"):
                    if self.fit.params[self.name + '_p1center'].stderr is not None:
                        if self.fit.params[self.name + '_p1center'].stderr > lambda_error:
                            sigmal = self.fit.params[self.name + '_p1center'].stderr

                self.fwhm = round(self.pics[0].sigma[0], 5)

                for i in range(1, len(self.deltaP0i)):
                    self.deltaP0i[i][0] = round(self.pics[i].ctr[0] - self.lamb_fit, 5)
                self.P, self.sigmaP = self.f_P(self.lamb_fit, self.lamb0, sigmalambda=sigmal)

                if self.name == "Ruby":
                    self.Calcul_Ruby(input_spe)

                elif self.name == "Rhodamine6G":
                    self.study_add = pd.DataFrame(np.array([abs(self.deltaP0i[1][0])]), columns=['Deltap12'])

                elif self.name == "Sm":
                    if "RuSm" in self.name_spe:
                        self.P, self.sigmaP = Sm_1997_Datchi(self.deltaP0i[1][0], lamb0=self.lamb0, sigmalambda=sigmal)
                    if "RuSmT" in self.name_spe:
                        T, sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit, input_spe[self.spe].lamb_fit, sigmalambda=[sigmal, input_spe[self.spe].sigmaP])
                        self.study_add = pd.DataFrame(np.array([[self.deltaP0i[1][0], T, sigmaT]]), columns=['Deltap12', 'T', 'sigma_T'])
                elif self.name == "SrFCl":
                    self.study_add = pd.DataFrame(np.array([abs(self.deltaP0i[1][0])]), columns=['Deltap12'])

                self.study = pd.concat(
                    [
                        pd.DataFrame(
                            np.array([[self.P, self.sigmaP, self.lamb_fit, self.fwhm, self.state]]),
                            columns=[
                                'P_' + self.name,
                                'sigma_P_' + self.name,
                                'lambda_' + self.name,
                                'fwhm_' + self.name,
                                "State_" + self.name,
                            ],
                        ),
                        self.study_add,
                    ],
                    axis=1,
                )
            else:
                self.P, self.sigmaP = Ruby_1986_Mao(self.lamb_fit, lamb0=self.lamb0, sigmalambda=sigmal, hydro=False)

                if "Sm" in self.name_spe:
                    self.lamb0 = Sm_1997_Datchi(self.P)
                elif "SrFCl" in self.name_spe:
                    self.lamb0 = SrFCl(self.P)
                else:
                    self.lamb0 = Raman_Dc12__2003_Occelli(self.P)
                self.indexX = None
                self.study_add = pd.DataFrame(np.array([[abs(self.deltaP0i[1][0])]]), columns=['Deltap12'])

                self.study = pd.concat(
                    [
                        pd.DataFrame(
                            np.array([[self.P, self.sigmaP, self.lamb_fit, self.fwhm, self.state]]),
                            columns=[
                                'P_' + self.name,
                                'sigma_P_' + self.name,
                                'lambda_' + self.name,
                                'fwhm_' + self.name,
                                "State_" + self.name,
                            ],
                        ),
                        self.study_add,
                    ],
                    axis=1,
                )
