import numpy as np
import pandas as pd
import peakutils as pk
from lmfit.models import LinearModel
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from typing import List, Optional

from Bibli.Data.Spectrum import Spectrum as SpectrumData
from Bibli.Use_Data.Gauge import Gauge
from Bibli.Data.Data_Gauge import Data_Gauge


def Gen_sum_F(list_f):
    def sum_function(x, *params):
        y = np.zeros_like(x)
        index = 0
        for f in list_f:
            num_params = f.__code__.co_argcount - 1
            y += f(x, *params[index:index + num_params])
            index += num_params
        return y
    return sum_function


class Spectrum:
    """Logique de traitement/fits associé à :class:`SpectrumData`."""

    def __init__(self, data: SpectrumData):
        self.data = data
        self.wnb = np.array(data.wnb)
        self.spec = np.array(data.spec)
        self.spec_brut = np.array(data.spec)
        self.param_f = data.param_f
        self.deg_baseline = data.deg_baseline
        self.type_filtre = data.type_filtre
        self.y_filtre, self.blfit = None, None
        self.x_corr = self.wnb
        self.Data_treatement(print_data=False)

        self.E = data.E
        self.X = None
        self.Y = None
        self.dY = None
        self.bit_model = False
        self.model = None
        self.fit = "Fit Non effectué"
        self.bit_fit = False
        self.lamb_fit = None
        self.indexX = None

        self.Gauges: List[Gauge] = [Gauge(g) if isinstance(g, Data_Gauge) else g for g in data.Gauges]
        self.lambda_error = round((self.wnb[-1] - self.wnb[0]) * 0.5 / len(self.wnb), 4)
        self.study = pd.DataFrame()
        self.help = "Spectre: etude de spectre"

    # ------------------------------------------------------------------
    def Corr(self, list_lamb0):
        for i in range(len(self.Gauges)):
            if list_lamb0[i] is not None:
                self.Gauges[i].lamb0 = list_lamb0[i]
            self.Gauges[i].Calcul(input_spe=self.Gauges, lambda_error=self.lambda_error)
        self.study = pd.concat([x.study for x in self.Gauges], axis=1)

    # ------------------------------------------------------------------
    def Print(self, ax=None, ax2=None, return_fig=False):
        import matplotlib.pyplot as plt

        if ax is None:
            print_fig = True
            fig, (ax, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 4), gridspec_kw={'height_ratios': [0.85, 0.15]})
        else:
            print_fig = False
        ax.plot(self.x_corr, self.blfit, '-.', c='g', markersize=1)
        ax.plot(self.wnb, self.spec, '-', color='lightgray', markersize=4)
        ax.plot(self.x_corr, self.y_corr + self.blfit, '.', color='black', markersize=3)

        for G in self.Gauges:
            if G.bit_fit:
                titre_fiti = f"{G.name}:$\\lambda_0=$" + str(G.lamb0)
                bf = self.blfit if G.indexX is None else self.blfit[G.indexX]
                if G.color_print[0] is not None:
                    ax.plot(G.X, G.Y, '--', label=titre_fiti, markersize=1, c=G.color_print[0])
                    if ax2 is not None:
                        ax2.plot(G.X, G.dY / max(np.abs(G.dY)), '-', c=G.color_print[0])
                else:
                    ax.plot(G.X, G.Y, '--', label=titre_fiti, markersize=1)
                    if ax2 is not None:
                        ax2.plot(G.X, G.dY / max(np.abs(G.dY)), '-')
                for i, pic in enumerate(G.pics):
                    if pic.best_fit is not None:
                        y_p = pic.best_fit[G.indexX] + bf
                        if G.color_print[1] is not None:
                            titre_pic = rf" $p_{i+1}^{{(G.name[0])}}= {round(pic.ctr[0], 3)}$"
                            ax.fill_between(G.X, y_p, bf, where=y_p > min(y_p), alpha=0.3, label=titre_pic, color=G.color_print[1][i])
                        else:
                            ax.fill_between(G.X, y_p, bf, where=y_p > min(y_p), alpha=0.1)

        ax.minorticks_on()
        ax.tick_params(which='major', length=10, width=1.5, direction='in')
        ax.tick_params(which='minor', length=5, width=1.5, direction='in')
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        ax.set_title(f'$Spectre,\\Delta\\lambda=$' + str(self.lambda_error))
        ax.set_ylabel('Amplitude (U.A.)')
        ax.set_xlim([min(self.x_corr), max(self.x_corr)])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if ax2 is not None:
            ax2.axhline(0, color="k", ls='-.')
            ax2.minorticks_on()
            ax2.tick_params(which='major', length=10, width=1.5, direction='in')
            ax2.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax2.tick_params(which='both', bottom=True, top=True, left=True, right=True)
            ax2.set_xlabel(f'$\\lambda$ (nm)')
            ax2.set_ylabel(f'$(Data-Fit)/max (U.A.)$')
            ax2.set_xlim([min(self.x_corr), max(self.x_corr)])
            ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        else:
            ax.set_xlabel(f'$\\lambda$ (nm)')
        ax.legend(loc="best")
        if return_fig:
            return fig
        if print_fig:
            plt.show()
        else:
            return ax

    # ------------------------------------------------------------------
    def FIT_One_Jauge(self, num_jauge=0, peakMax0=None, wnb_range=3, coef_spe=None, sigma=None, inter=None, model_fit=None, manuel=False, model_jauge=None):
        G = self.Gauges[num_jauge]
        y_sub = self.y_corr
        if peakMax0 is not None and model_jauge is None:
            peakMax = peakMax0
        elif model_jauge is not None:
            G = model_jauge[num_jauge]
            wnb_range_model = (self.wnb[G.indexX][-1] - self.wnb[G.indexX][0]) / 2
            if ("Lw" and "Hg") not in G.name_spe:
                if wnb_range_model < self.lambda_error * 10:
                    wnb_range = self.lambda_error * 10
                elif wnb_range_model <= wnb_range:
                    wnb_range = wnb_range_model + self.lambda_error
            peakMax = G.lamb_fit
        else:
            peakMax = G.lamb0
        dpic = [dp[0] for dp in G.deltaP0i]
        if ("Lw" and "Hg") in G.name_spe:
            match = re.search("Lw(\d+)", G.name_spe)
            Dwnb_low = float(match.group(1))
            match = re.search("Hg(\d+)", G.name_spe)
            Dwnb_hight = float(match.group(1))
            G.indexX = np.where((self.wnb > (peakMax - Dwnb_low)) & (self.wnb < (peakMax + Dwnb_hight)))[0]
            wnb_range = Dwnb_hight + Dwnb_low
        else:
            G.indexX = np.where((self.wnb > peakMax - (wnb_range + abs(min(min(dpic), 0)))) & (self.wnb < peakMax + (wnb_range + max(max(dpic), 0))))[0]
        x_sub = self.wnb[G.indexX]
        y_sub = y_sub[G.indexX]
        if ("Lw" and "Hg") in G.name_spe:
            G.deltaP0i[1][0] = float(find_peaks(self.y_corr[G.indexX])[0][0]) * self.lambda_error - wnb_range / 2
        # génération des pics
        if manuel is False:
            peakMax_calc = x_sub[pk.indexes(y_sub, thres=0.00008, min_dist=int(len(x_sub) / 5))]
            print("tres petite amplitude", peakMax_calc)
            if len(peakMax_calc) == 0:
                peakMax_calc = x_sub[pk.indexes(y_sub, thres=0.1, min_dist=int(len(x_sub) / 5))]
            if (peakMax_calc is None) or (len(peakMax_calc) == 0):
                peakMax = peakMax
            else:
                peakMax = peakMax_calc[0]
        else:
            peakMax = peakMax0
        if peakMax is not None:
            G.lamb_fit = peakMax
            G.Update_Fit(peakMax, max(y_sub), coef_spe=coef_spe, sigma=sigma, inter=inter, model_fit=model_fit)
        G.Update_model()
        if G.model is None:
            G.model = LinearModel()
            G.model.set_param_hint(G.name + 'slope', value=0)
            G.model.set_param_hint(G.name + 'intercept', value=min(y_sub))
        param = G.model.make_params()
        G.fit = G.model.fit(y_sub, x=x_sub, params=param)
        G.model = G.fit.model
        G.Y = G.fit.best_fit + self.blfit[G.indexX]
        G.dY = G.fit.best_fit - y_sub
        G.X = x_sub
        G.lamb_fit = G.fit.best_values[G.name + '_p1center'] if f"{G.name}_p1center" in G.fit.best_values else peakMax
        G.bit_fit = True
        for p in G.pics:
            new_param = p.Out_model(out=G.fit)
            p.Update(ctr=float(new_param[0]), ampH=float(new_param[1]), coef_spe=new_param[3], sigma=float(new_param[2]))
            param = p.model.make_params()
            p.best_fit = p.model.eval(param, x=self.wnb)
        self.Gauges[num_jauge] = G

    def FIT(self, wnb_range=2, coef_spe=None, sigma=None, inter=None, model_fit=None, model_jauge=None):
        for i, G in enumerate(self.Gauges):
            if G.state == "Y":
                try:
                    self.FIT_One_Jauge(num_jauge=i, peakMax0=G.lamb_fit, wnb_range=wnb_range, coef_spe=coef_spe, sigma=sigma, inter=inter, model_fit=model_fit, model_jauge=model_jauge)
                except Exception as e:
                    G.state = "IN_NOISE"
                    print("error:", e, "in fit of :", G.name)
            G.bit_fit = True
        for G in self.Gauges:
            G.Calcul(input_spe=self.Gauges, lambda_error=self.lambda_error)
        self.study = pd.concat([x.study for x in self.Gauges], axis=1)
        self.bit_fit = True

    def FIT_Curv(self, inter=1):
        list_F = []
        initial_guess = []
        bounds_min, bounds_max = [], []
        x_min, x_max = float(self.Gauges[0].lamb0), float(self.Gauges[0].lamb0)
        for j, G in enumerate(self.Gauges):
            for i, p in enumerate(G.pics):
                x_min, x_max = min(x_min, p.ctr[0] - p.sigma[0] * 5), max(x_max, p.ctr[0] + p.sigma[0] * 5)
                list_F.append(p.f_model)
                initial_guess += [p.ctr[0], p.ampH[0], p.sigma[0]]
                for c in p.coef_spe:
                    initial_guess += [c[0]]
                bounds_min += [p.ctr[1][0], p.ampH[1][0], p.sigma[1][0]]
                bounds_max += [p.ctr[1][1], p.ampH[1][1], p.sigma[1][1]]
                for c in p.coef_spe:
                    bounds_min += [c[1][0]]
                    bounds_max += [c[1][1]]
            G.Update_model()
        bounds = [bounds_min, bounds_max]
        for i, G in enumerate(self.Gauges):
            if i == 0:
                self.model = G.model
            else:
                self.model += G.model
        self.Data_treatement()
        self.indexX = np.where((self.wnb >= x_min) & (self.wnb <= x_max))[0]
        x_corr = self.wnb[self.indexX]
        y_sub = self.y_corr[self.indexX]
        blfit = self.blfit[self.indexX]
        sum_function = Gen_sum_F(list_F)
        params, params_covar = curve_fit(sum_function, x_corr, y_sub, p0=initial_guess, bounds=bounds)
        fit = sum_function(x_corr, *params)
        self.Y = fit + blfit
        self.X = x_corr
        self.dY = y_sub - fit
        self.lamb_fit = params[0]
        ij_3, ij_4, ij_5 = 0, 0, 0
        params_list = list(params)
        for i, J in enumerate(self.Gauges):
            for j, p in enumerate(J.pics):
                n_c = len(p.coef_spe)
                start_idx = 3 * ij_3 + 4 * ij_4 + 5 * ij_5
                end_idx = start_idx + 3
                if n_c == 0:
                    params_pic = params_list[start_idx:end_idx]
                    ij_3 += 1
                elif n_c == 1:
                    params_pic = params_list[start_idx:end_idx] + [np.array([params_list[end_idx]])]
                    ij_4 += 1
                elif n_c == 2:
                    params_pic = params_list[start_idx:end_idx] + [np.array(params_list[end_idx:end_idx + 2])]
                    ij_5 += 1
                p.Update(ctr=float(params_pic[0]), ampH=float(params_pic[1]), coef_spe=params_pic[3], sigma=float(params_pic[2]), inter=float(inter))
                param = p.model.make_params()
                p.best_fit = p.model.eval(param, x=self.wnb)
            if j == 0:
                J.lamb_fit = params_pic[0]
            param = J.model.make_params()
            J.Y = p.model.eval(param, x=self.wnb)
            J.X = self.wnb
            J.dY = J.Y - self.y_corr
            J.bit_fit = True
        for G in self.Gauges:
            G.Calcul(input_spe=self.Gauges, lambda_error=self.lambda_error)
        self.study = pd.concat([x.study for x in self.Gauges], axis=1)
        self.bit_fit = True

    def Clear_study(self, num_jauge):
        self.Gauges[num_jauge].study.loc[:, :] = None

    def Calcul_study(self, mini=False):
        for i in range(len(self.Gauges)):
            if ("DRX" in self.Gauges[i].name_spe) and (self.bit_fit is True):
                self.Gauges[i].bit_fit = True
            self.Gauges[i].Calcul(input_spe=self.Gauges, mini=mini, lambda_error=self.lambda_error)
        self.study = pd.concat([x.study for x in self.Gauges], axis=1)

    def Data_treatement(self, deg_baseline=None, type_filtre=None, param_f=None, print_data=False, ax=None, ax2=None):
        if deg_baseline is not None:
            self.deg_baseline = deg_baseline
        if type_filtre is not None:
            self.type_filtre = type_filtre
        if param_f is not None:
            self.param_f = param_f
        if self.type_filtre == 'svg':
            if len(self.wnb) >= self.param_f[0]:
                self.y_filtre = savgol_filter(self.spec, window_length=self.param_f[0], polyorder=self.param_f[1], mode='interp')
            else:
                self.y_filtre = self.spec
        else:
            self.y_filtre = self.spec
        # baseline
        sample_rate = (self.wnb[-1] - self.wnb[0]) / (len(self.wnb) - 1)
        print_baseline = True
        if print_data:
            print_baseline = False
        vec = np.linspace(0, len(self.spec), len(self.spec))
        idx_baseline = pk.indexes(self.y_filtre, thres=0.3, min_dist=int(len(self.y_filtre) / 20))
        baseline = np.poly1d(np.polyfit(vec[idx_baseline], self.y_filtre[idx_baseline], deg=self.deg_baseline))(vec)
        self.y_corr = self.spec - baseline
        self.blfit = baseline
        if ax is not None:
            ax.plot(self.wnb, baseline, '-.', c='g', markersize=1)
            ax.plot(self.wnb, self.spec, '-', color='lightgray', markersize=1)
            ax.plot(self.wnb, self.y_corr + baseline, '-', color='black', markersize=1)
