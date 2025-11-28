import numpy as np
from lmfit.models import (
    GaussianModel,
    MoffatModel,
    Pearson4Model,
    PseudoVoigtModel,
    SplitLorentzianModel,
)

from Bibli.Data.Data_Peak import Data_Peak
from Bibli.Physic_law import *  # noqa: F401,F403


class Peak:
    """Logique de fit et d'update d'un pic à partir d'un :class:`Data_Peak`."""

    def __init__(self, peak: Data_Peak):
        self.data = peak
        self.name = peak.name.replace("-", "_")
        self.model_fit = peak.model
        self.name_coef_spe = []
        self.f_model = None
        self.f_amp = None

        self.inter = peak.inter
        self.inter_min = max(1 - self.inter, 0)
        self.delta_ctr = peak.delta_ctr

        # bornes initiales
        self.sigma = [peak.sigma, [peak.sigma * self.inter_min, peak.sigma * (1 + self.inter)]]
        self.ampH = [peak.height, [peak.height * self.inter_min, peak.height * (1 + self.inter)]]
        self.coef_spe = [[c, [c * self.inter_min, c * (1 + self.inter)]] for c in peak.coef_spe]

        self.model = None
        self.best_fit = None

        self._configure_model()
        self._prepare_special_params()
        amp = peak.amplitude
        if amp is None:
            amp = self._compute_amplitude(self.ampH[0], [c[0] for c in self.coef_spe], self.sigma[0])
        self.amp = [amp, [amp * self.inter_min, amp * (1 + self.inter)]]

        self.ctr = [peak.center, [peak.center - self.delta_ctr, peak.center + self.delta_ctr]]
        self._set_param_hints()

    # ------------------------------------------------------------------
    # Configuration du modèle
    # ------------------------------------------------------------------
    def _configure_model(self):
        if self.model_fit == "PseudoVoigt":
            self.f_amp = self._amp_psd
            self.name_coef_spe = ["fraction"]
            self.model = PseudoVoigtModel(prefix=self.name)
            self.f_model = PseudoVoigt
        elif self.model_fit == "Moffat":
            self.f_amp = self._amp_moffat
            self.name_coef_spe = ["beta"]
            self.model = MoffatModel(prefix=self.name)
            self.f_model = Moffat
        elif self.model_fit == "SplitLorentzian":
            self.f_amp = self._amp_split_lorentzian
            self.name_coef_spe = ["sigma_r"]
            self.model = SplitLorentzianModel(prefix=self.name)
            self.f_model = SplitLorentzian
        elif self.model_fit == "Gaussian":
            self.f_amp = self._amp_gaussian
            self.name_coef_spe = []
            self.model = GaussianModel(prefix=self.name)
            self.f_model = Gaussian
        elif self.model_fit == "PearsonIV":
            self.f_amp = self._amp_pearson_iv
            self.name_coef_spe = ["expon", "skew"]
            self.model = Pearson4Model(prefix=self.name)
            self.f_model = PearsonIV
        else:
            raise ValueError(f"Modèle inconnu : {self.model_fit}")

    def _prepare_special_params(self):
        if self.model_fit == "PseudoVoigt":
            frac = self.coef_spe[0][0]
            frac = max(min(1, frac), 0)
            self.coef_spe = [[frac, [frac * self.inter_min, max(0.1, min(1, frac * (1 + self.inter)))]]]
        elif self.model_fit == "PearsonIV":
            if len(self.coef_spe) < 2:
                base = self.coef_spe[0][0]
                self.coef_spe = [[base, [0.5, max(1, base * (1 + self.inter))]], [base, [0.5, max(1, base * (1 + self.inter))]]]
            skew = self.coef_spe[1][0]
            limits = [skew * (1 - self.inter) - 0.1, skew * (1 + self.inter) + 0.1]
            m_val = max(0.505, self.coef_spe[0][0])
            self.coef_spe = [
                [m_val, [max(0.501, m_val * self.inter_min), max(0.51, m_val * (1 + self.inter))]],
                [skew, [min(limits), max(limits)]],
            ]
            new_ctr = self.ctr[0] + self.sigma[0] * self.coef_spe[1][0] / (2 * self.coef_spe[0][0])
            self.ctr = [new_ctr, [new_ctr - self.delta_ctr, new_ctr + self.delta_ctr]]

    def _compute_amplitude(self, ampH, coef_spe, sigma):
        if self.model_fit == "PseudoVoigt":
            return self._amp_psd(ampH, coef_spe, sigma)
        if self.model_fit == "Gaussian":
            return self._amp_gaussian(ampH, coef_spe, sigma)
        if self.model_fit == "Moffat":
            return self._amp_moffat(ampH, coef_spe, sigma)
        if self.model_fit == "SplitLorentzian":
            return self._amp_split_lorentzian(ampH, coef_spe, sigma)
        if self.model_fit == "PearsonIV":
            return self._amp_pearson_iv(ampH, coef_spe, sigma)
        return ampH

    # ------------------------------------------------------------------
    # Fonctions d'amplitude
    # ------------------------------------------------------------------
    @staticmethod
    def _amp_psd(ampH, coef_spe, sigma):
        return ampH / (((1 - coef_spe[0])) / (sigma * np.sqrt(np.pi / np.log(2))) + (coef_spe[0]) / (np.pi * sigma))

    @staticmethod
    def _amp_gaussian(ampH, coef_spe, sigma):
        return ampH * sigma * np.sqrt(np.pi * 2)

    @staticmethod
    def _amp_moffat(ampH, coef_spe, sigma):
        return ampH * coef_spe[0]

    @staticmethod
    def _amp_split_lorentzian(ampH, coef_spe, sigma):
        return ampH / 2 * np.pi * (sigma + coef_spe[0])

    @staticmethod
    def _amp_pearson_iv(ampH, coef_spe, sigma):
        m_val, skew = coef_spe[0], coef_spe[1]
        normalization = np.abs(np.math.gamma(m_val + 1j * skew / 2) / np.math.gamma(m_val)) ** 2 / (sigma * np.math.beta(m_val - 0.5, 0.5))
        return ampH / (normalization * (1 + (skew / (2 * m_val)) ** 2) ** -m_val * np.exp(-skew * np.arctan(-skew / (2 * m_val))))

    # ------------------------------------------------------------------
    # Paramètres du modèle
    # ------------------------------------------------------------------
    def _set_param_hints(self):
        for i, name_spe in enumerate(self.name_coef_spe):
            self.model.set_param_hint(
                self.name + name_spe,
                value=self.coef_spe[i][0],
                min=self.coef_spe[i][1][0],
                max=self.coef_spe[i][1][1],
            )
        self.model.set_param_hint(self.name + "amplitude", value=self.amp[0], min=self.amp[1][0], max=self.amp[1][1])
        self.model.set_param_hint(self.name + "sigma", value=self.sigma[0], min=self.sigma[1][0], max=self.sigma[1][1])
        self.model.set_param_hint(self.name + "center", value=self.ctr[0], min=self.ctr[1][0], max=self.ctr[1][1])

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def Update(self, ctr=None, ampH=None, coef_spe=None, sigma=None, inter=None, model_fit=None, Delta_ctr=None, amp=None):
        if Delta_ctr is None:
            Delta_ctr = self.delta_ctr
        if inter is not None:
            self.inter = inter
        self.inter_min = max(1 - self.inter, 0)

        if coef_spe is not None:
            self.coef_spe = [[c, [c * self.inter_min, c * (1 + self.inter)]] for c in coef_spe]

        if ctr is not None:
            self.ctr = [ctr, [ctr - Delta_ctr, ctr + Delta_ctr]]
        if sigma is not None:
            self.sigma = [sigma, [sigma * self.inter_min, sigma * (1 + self.inter)]]
        else:
            self.sigma = [self.sigma[0], [self.sigma[0] * self.inter_min, self.sigma[0] * (1 + self.inter)]]

        if model_fit is not None and model_fit != self.model_fit:
            self.model_fit = model_fit
            self._configure_model()
        self._prepare_special_params()

        if ampH is not None and amp is None:
            self.ampH = [ampH, [ampH * self.inter_min, ampH * (1 + self.inter)]]
            amp = self._compute_amplitude(self.ampH[0], [c[0] for c in self.coef_spe], self.sigma[0])
            self.amp = [amp, [amp * self.inter_min, amp * (1 + self.inter)]]
        elif amp is not None:
            self.amp = [amp, [amp * self.inter_min, amp * (1 + self.inter)]]
        else:
            amp = self._compute_amplitude(self.ampH[0], [c[0] for c in self.coef_spe], self.sigma[0])
            self.amp = [amp, [amp * self.inter_min, amp * (1 + self.inter)]]

        self._set_param_hints()

    def Out_model(self, out=None, l_params=None):
        if out is not None:
            ctr = round(out.params[self.name + "center"].value, 3)
            sigma = round(out.params[self.name + "sigma"].value, 3)
            ampH = round(out.params[self.name + "height"].value, 3)
            coef_spe = np.array([round(out.params[self.name + name_spe].value, 3) for name_spe in self.name_coef_spe])
            if self.model_fit == "PearsonIV":
                ctr = round(out.params[self.name + "position"].value, 3)
            return [ctr, ampH, sigma, coef_spe]
        elif l_params is not None:
            ctr = round(l_params[0], 3)
            sigma = round(l_params[2], 3)
            ampH = round(l_params[1], 3)
            coef_spe = np.array([round(p, 3) for p in l_params[3]])
            if self.model_fit == "PearsonIV":
                ctr = ctr - sigma * coef_spe[1] / (4 * (coef_spe[0] - 1))
            return [ctr, ampH, sigma, coef_spe]
        else:
            params = self.model.make_params()
            ctr = round(params[self.name + "center"].value, 3)
            sigma = round(params[self.name + "sigma"].value, 3)
            ampH = round(params[self.name + "height"].value, 3)
            coef_spe = np.array([round(params[self.name + name_spe].value, 3) for name_spe in self.name_coef_spe])
            if self.model_fit == "PearsonIV":
                ctr = round(params[self.name + "position"].value, 3)
            return [ctr, ampH, sigma, coef_spe], params

    def Help(self, request=None):
        if request is None:
            print("Choisissez un modèle pour obtenir des informations détaillées :")
            print("Disponible : 'PseudoVoigt', 'Moffat', 'SplitLorentzian', 'Gaussian', 'PearsonIV'.")
        elif request == "param":
            if self.model_fit == "PseudoVoigt":
                print("PseudoVoigt : center, amplitude, sigma, fraction.")
            elif self.model_fit == "Moffat":
                print("Moffat : center, amplitude, sigma, beta.")
            elif self.model_fit == "SplitLorentzian":
                print("SplitLorentzian : center, amplitude, sigma, sigma_r.")
            elif self.model_fit == "Gaussian":
                print("Gaussian : center, amplitude, sigma.")
            elif self.model_fit == "PearsonIV":
                print("PearsonIV : center, amplitude, sigma, expon (m), skew.")
        elif request == "test":
            print("OK")
