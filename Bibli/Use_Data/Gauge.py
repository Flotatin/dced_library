import re
from typing import List, Optional

import numpy as np
import pandas as pd
from pynverse import inversefunc

from Bibli.Data.Data_Gauge import Data_Gauge
from Bibli.Data.Data_Peak import Data_Peak
from Bibli.Physic_law import (
    Raman_Dc12__2003_Occelli,
    Raman_Dc13_1997_Schieferl,
    Rhodamine_6G_2024_Dembele,
    Ruby_2020_Shen,
    Sm_2015_Rashenko,
    SrFCl,
    T_Ruby_Sm_1997_Datchi,
)
from Bibli.Use_Data.Peak import Peak


class Gauge:
    """Logique d'utilisation associée à :class:`Data_Gauge`."""

    def __init__(self, gauge_data: Data_Gauge, spe: Optional[str] = None, f_P=None):
        self.data = gauge_data
        self.spe = spe
        self.f_P = f_P or gauge_data.f_P
        self.name = gauge_data.name
        self.lamb0 = gauge_data.lamb0
        self.nb_pic = gauge_data.nb_pic or len(gauge_data.peaks)
        self.deltaP0i = gauge_data.deltaP0i or []
        self.name_spe = gauge_data.name_spe
        self.color_print = gauge_data.color_print or [None, None]

        self.pics: List[Peak] = []
        self.model = None
        self.bit_model = False
        self.bit_fit = False
        self.fit = "Fit Non effectué"
        self.P = 0
        self.T = 0
        self.lamb_fit = None
        self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[10, 1000]) if self.f_P else None
        self.study = pd.DataFrame()
        self.study_add = pd.DataFrame()
        self.state = "Y"

        if gauge_data.peaks:
            for p_data in gauge_data.peaks:
                self._append_peak(Peak(p_data))
        else:
            self._init_by_name()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _append_peak(self, peak: Peak):
        self.pics.append(peak)
        if self.model is None:
            self.model = peak.model
        else:
            self.model = self.model + peak.model
        self.bit_model = True

    def _init_by_name(self):
        if "Ruby" in self.name:
            self._init_ruby()
        elif "Sm" in self.name or "Samarium" in self.name:
            self._init_sm()
        elif "SrFCl" in self.name:
            self._init_srfcl()
        elif "Rhodamine" in self.name:
            self._init_rhodamine()
        elif "Diamond_c12" in self.name:
            self._init_diamond(isotope="c12")
        elif "Diamond_c13" in self.name:
            self._init_diamond(isotope="c13")
        elif self.nb_pic:
            self._init_custom()

    def _init_custom(self):
        self.pics = []
        for i in range(self.nb_pic):
            sigma = 0.25
            model_fit = "PseudoVoigt"
            if "Sig" in self.name_spe:
                match = re.search("Sig(\d+)", self.name_spe)
                if match:
                    sigma = float(match.group(1))
            if "Mod:" in self.name_spe:
                match = re.search(r"Mod:(\w+)", self.name_spe)
                if match:
                    model_fit = str(match.group(1))
            delta = self.deltaP0i[i][0] if i < len(self.deltaP0i) else 0
            data_peak = Data_Peak(
                name=f"{self.name}_p{i+1}",
                center=(self.lamb0 or 0) + delta,
                sigma=sigma,
                model=model_fit,
                height=1.0,
            )
            self._append_peak(Peak(data_peak))

    def _init_ruby(self):
        self.lamb0 = 694.4
        self.nb_pic = 2
        self.name = "Ruby"
        self.name_spe = "Ru"
        self.deltaP0i = [[0, 1], [-1.5, 0.75]]
        self.color_print = ['darkred', ['darkred', 'brown']]
        if self.f_P is None:
            self.f_P = Ruby_2020_Shen
            self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[690, 750])
        self._init_custom()

    def _init_sm(self, all=False):
        self.lamb0 = 685.39
        self.nb_pic = 1
        self.name = "Sm"
        self.deltaP0i = [[0, 1]]
        if self.f_P is None:
            self.f_P = Sm_2015_Rashenko
            self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[683, 730])
        self.color_print = ['darkblue', ['darkblue']]
        self._init_custom()

    def _init_diamond(self, isotope='c12'):
        if isotope == 'c12':
            self.lamb0 = 1333
            self.f_P = Raman_Dc12__2003_Occelli
            self.color_print = ['silver', ['silver']]
        else:
            self.lamb0 = 1287
            self.f_P = Raman_Dc13_1997_Schieferl
            self.color_print = ['dimgrey', ['dimgrey']]
        self.nb_pic = 1
        self.name_spe = "Sig20"
        self.name = f"D_{isotope}"
        self.deltaP0i = [[0, 1]]
        self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[self.lamb0 - 10, self.lamb0 + 200])
        self._init_custom()

    def _init_srfcl(self):
        self.lamb0 = 685.39
        self.nb_pic = 1
        self.name = "SrFCl"
        self.deltaP0i = [[0, 1]]
        if self.f_P is None:
            self.f_P = SrFCl
            self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[683, 730])
        self.color_print = ['darkorange', ['darkorange']]
        self._init_custom()

    def _init_rhodamine(self):
        self.lamb0 = 550
        self.nb_pic = 2
        self.name = "Rhodamine6G"
        self.deltaP0i = [[0, 1], [52, 0.5, 4]]
        self.color_print = ['limegreen', ['limegreen', 'darkgreen']]
        self.name_spe = "Lw50Hg150Sig20Mod:Gaussian"
        if self.f_P is None:
            self.f_P = Rhodamine_6G_2024_Dembele
            self.inv_f_P = inversefunc((lambda x: self.f_P(x)), domain=[530, 600])
        self._init_custom()

    # ------------------------------------------------------------------
    # Calculs
    # ------------------------------------------------------------------
    def Calcul_Ruby(self, input_spe):
        if 'RuSmT' in self.name_spe:
            sigmals = input_spe[self.spe].fit.params.get(f"{input_spe[self.spe].name}_p1center").stderr or 0
            sigmal = self.fit.params.get(f"{self.name}_p1center").stderr or 0
            T, sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit, input_spe[self.spe].lamb_fit, lamb0S=input_spe[self.spe].lamb0, lamb0R=self.lamb0, sigmalambda=[sigmal, sigmals])
            self.study_add = pd.DataFrame(np.array([[self.deltaP0i[1][0], T, sigmaT]]), columns=['Deltap12', 'T', 'sigmaT'])
        elif "RuT" in self.name_spe:
            sigmal = self.fit.params.get(f"{self.name}_p1center").stderr or 0
            T, sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit, lamb0R=self.lamb0, sigmalambda=[sigmal, input_spe[self.spe].sigmaP], P=input_spe[self.spe].P)
            self.study_add = pd.DataFrame(np.array([[self.deltaP0i[1][0], T, sigmaT]]), columns=['Deltap12', 'T', 'sigmaT'])
        else:
            self.study_add = pd.DataFrame(np.array([self.deltaP0i[1][0]]), columns=['Deltap12'])

    def Calcul(self, input_spe=None, mini=False, lambda_error=0):
        if not self.bit_fit:
            print("NO FIT of", self.name, "do it !")
            return
        self.lamb_fit = round(self.pics[0].ctr[0], 3)
        sigmal = lambda_error
        if self.state == "Y":
            if hasattr(self.fit, "params"):
                stderr = self.fit.params.get(f"{self.name}_p1center").stderr
                if stderr is not None and stderr > lambda_error:
                    sigmal = stderr
            self.fwhm = round(self.pics[0].sigma[0], 5)
            for i in range(1, len(self.deltaP0i)):
                self.deltaP0i[i][0] = round(self.pics[i].ctr[0] - self.lamb_fit, 5)
            self.P, self.sigmaP = self.f_P(self.lamb_fit, self.lamb0, sigmalambda=sigmal)
        else:
            self.fwhm, self.P, self.sigmaP = 0.1, 0, 0

        if 'Ru' in self.name_spe:
            self.study_add = pd.DataFrame(np.array([abs(self.deltaP0i[1][0])]), columns=['Deltap12'])
            if 'SmT' in self.name_spe and input_spe is not None:
                sigmals = lambda_error
                if self.state == "Y" and hasattr(input_spe[self.spe].fit, "params"):
                    stderr = input_spe[self.spe].fit.params.get(f"{input_spe[self.spe].name}_p1center").stderr
                    if stderr is not None and stderr > lambda_error:
                        sigmals = stderr
                T, sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit, input_spe[self.spe].lamb_fit, lamb0S=input_spe[self.spe].lamb0, lamb0R=self.lamb0, sigmalambda=[sigmal, sigmals])
                self.study_add = pd.DataFrame(np.array([[self.deltaP0i[1][0], T, sigmaT]]), columns=['Deltap12', 'T', 'sigma_T'])

        self.study = pd.concat(
            [pd.DataFrame(np.array([[self.P, self.sigmaP, self.lamb_fit, self.fwhm, self.state]]), columns=[f"P_{self.name}", f"sigma_P_{self.name}", f"lambda_{self.name}", f"fwhm_{self.name}", f"State_{self.name}"]), self.study_add],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def Clear(self, c=None):
        self.study.loc[:, :] = c
        self.bit_model = False
        self.model = None
        self.bit_fit = False
        self.lamb_fit = None
        self.fit = "Fit Non effectué"

    def Update_Fit(self, crt, ampH, coef_spe=None, sigma=None, inter=None, model_fit=None, Delta_ctr=None):
        for i in range(len(self.pics)):
            local_sigma = sigma
            if len(self.deltaP0i[i]) > 2:
                if local_sigma is None:
                    local_sigma = self.pics[0].sigma[0]
                local_sigma = local_sigma * self.deltaP0i[i][2]
            self.pics[i].Update(ctr=crt + self.deltaP0i[i][0], ampH=ampH * self.deltaP0i[i][1], coef_spe=coef_spe, sigma=local_sigma, inter=inter, model_fit=model_fit, Delta_ctr=Delta_ctr)
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
