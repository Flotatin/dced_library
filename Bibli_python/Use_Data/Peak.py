import numpy as np
from lmfit.models import (
    GaussianModel,
    MoffatModel,
    Pearson4Model,
    PseudoVoigtModel,
    SplitLorentzianModel,
)

from Bibli_python.Data.Data_Peak import data_Peak


class Peak:
    def __init__(self, peak: data_Peak, inter: float = 3.0):
        """
        peak : objet Peaks (données brutes)
        inter: facteur pour construire les bornes à partir de l'incertitude.
        """
        self.peak = data_Peak
        self.name = data_Peak.name
        self.model_fit = data_Peak.model
        self.inter = inter
        self.inter_min = max(1 - inter, 0)

        self.model = None              # lmfit.Model
        self.name_coef_spe = []        # noms des params spéciaux (fraction, beta, skew...)
        self.coef_spe = []             # valeur + min/max
        self.center = None             # idem
        self.sigma = None
        self.amplitude = None

        self._configure_model()
        self._build_internal_params_from_dict()
        self._set_param_hints()

    # -------------------------
    # 1) Choix du modèle lmfit
    # -------------------------
    def _configure_model(self):
        if self.model_fit == "PseudoVoigt":
            self.model = PseudoVoigtModel(prefix=self.name)
        elif self.model_fit == "Moffat":
            self.model = MoffatModel(prefix=self.name)
        elif self.model_fit == "SplitLorentzian":
            self.model = SplitLorentzianModel(prefix=self.name)
        elif self.model_fit == "Gaussian":
            self.model = GaussianModel(prefix=self.name)
        elif self.model_fit == "PearsonIV":
            self.model = Pearson4Model(prefix=self.name)
        else:
            raise ValueError(f"Modèle inconnu : {self.model_fit}")

    # -------------------------
    # 2) Transforme (value, spread) -> [val,[min,max]]
    # -------------------------
    @staticmethod
    def _value_min_max_from_tuple(val_tuple, inter):
        """
        val_tuple = (value, sigma) ou (value, (min,max)).
        """
        if len(val_tuple) != 2:
            raise ValueError("Chaque entrée de params doit être (value, sigma) ou (value, (min,max)).")

        value, spread = val_tuple

        if isinstance(spread, (tuple, list)) and len(spread) == 2:
            vmin, vmax = spread
        else:
            vmin = value - inter * spread
            vmax = value + inter * spread

        return [value, [vmin, vmax]]

    def _build_internal_params_from_dict(self):
        p = self.peak.params

        # center / sigma / amplitude génériques
        self.center = self._value_min_max_from_tuple(p.get("center", (0.0, 0.5)), self.inter)
        self.sigma = self._value_min_max_from_tuple(p.get("sigma", (0.25, 0.25)), self.inter)
        self.amplitude = self._value_min_max_from_tuple(p.get("amplitude", (1.0, 1.0)), self.inter)

        # paramètres "spéciaux" propres au modèle
        self.coef_spe = []
        for name in self.name_coef_spe:
            if name in p:
                self.coef_spe.append(self._value_min_max_from_tuple(p[name], self.inter))
            else:
                # valeur par défaut si rien fourni
                self.coef_spe.append(self._value_min_max_from_tuple((0.5, 0.5), self.inter))

        # ajustements spécifiques (ex PseudoVoigt fraction entre 0 et 1)
        if self.model_fit == "PseudoVoigt":
            frac_val = self.coef_spe[0][0]
            frac_val = max(0.0, min(1.0, frac_val))
            vmin = frac_val * self.inter_min
            vmax = min(1.0, frac_val * (1 + self.inter))
            self.coef_spe[0] = [frac_val, [vmin, max(0.1, vmax)]]

        if self.model_fit == "PearsonIV":
            # petit encadrement du param m (expon) et skew, très simplifié
            if len(self.coef_spe) < 2:
                # si un seul param spec fourni, on duplique
                self.coef_spe = [
                    self.coef_spe[0],
                    self._value_min_max_from_tuple((self.coef_spe[0][0], 0.5), self.inter)
                ]

    # -------------------------
    # 3) Applique les hints lmfit
    # -------------------------
    def _set_param_hints(self):
        # spéciaux
        for param in self.peak.param.key():
            v0=self.peak.param[param][0]
            sig=self.peak.param[param][1]
            self.model.set_param_hint(
            self.name + param,
            value=v0,
            min=v0-sig/2,
            max=v0+sig/2
        )
            
    # -------------------------
    # 4) API type Pics : Update / Out_model / Help
    # -------------------------
    def Update(self, **kwargs):
        """
        Mise à jour de certains paramètres, puis recalcul des hints.

        Ex:
            pic.Update(center=650.2, amplitude=1200., fraction=0.7)
        """
        for key, value in kwargs.items():
            # si scalaire -> on garde l'ancien spread
            if not isinstance(value, (tuple, list)):
                old = self.peak.params.get(key, (value, value * 0.1))
                _, spread = old
                self.peak.params[key] = (value, spread)
            else:
                self.peak.params[key] = tuple(value)

        self._build_internal_params_from_dict()
        self._set_param_hints()

    def Out_model(self, out=None, l_params=None):
        """
        Renvoie [center, height, sigma, coef_spe] pour rester compatible
        avec ton ancienne signature.

        - si out est un objet de fit lmfit
        - si l_params est une liste [ctr, ampH, sigma, coef_spe]
        """
        if out is not None:
            ctr = round(out.params[self.name + "center"].value, 3)
            sigma = round(out.params[self.name + "sigma"].value, 3)
            # lmfit calcule height à partir d'amplitude
            ampH = round(out.params[self.name + "height"].value, 3) \
                if self.name + "height" in out.params else None
            coef_spe = np.array([
                round(out.params[self.name + name_spe].value, 3)
                for name_spe in self.name_coef_spe
            ])
            return [ctr, ampH, sigma, coef_spe]

        elif l_params is not None:
            ctr = round(l_params[0], 3)
            ampH = round(l_params[1], 3)
            sigma = round(l_params[2], 3)
            coef_spe = np.array([round(p, 3) for p in l_params[3]])
            return [ctr, ampH, sigma, coef_spe]

        else:
            params = self.model.make_params()
            ctr = round(params[self.name + "center"].value, 3)
            sigma = round(params[self.name + "sigma"].value, 3)
            ampH = round(params[self.name + "height"].value, 3) \
                if self.name + "height" in params else None
            coef_spe = np.array([
                round(params[self.name + name_spe].value, 3)
                for name_spe in self.name_coef_spe
            ])
            return [ctr, ampH, sigma, coef_spe], params

    def Help(self, request=None):
        if request is None:
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
