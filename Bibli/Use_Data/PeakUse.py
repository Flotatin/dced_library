# Bibli/Use_Data/PeakUse.py

import copy
import numpy as np
from math import sqrt, log
from typing import Optional, List, Dict, Any

from lmfit.models import (
    PseudoVoigtModel,
    MoffatModel,
    SplitLorentzianModel,
    GaussianModel,
    Pearson4Model,
)

from Bibli.Data.PeakData import PeakData



"""------------------------------------- LOI FIT -------------------------------------"""      
def PseudoVoigt(x,center,ampH,sigma,fraction):
    amp=ampH/(((1-fraction))/(sigma*sqrt(np.pi/log(2)))+(fraction)/(np.pi*sigma))
    sigma_g=(sigma/(sqrt(2*log(2))))
    return (1-fraction)*amp/(sigma_g*sqrt(2*np.pi))*np.exp(-(x-center)**2/(2*sigma_g**2))+fraction*amp/np.pi*(sigma/((x-center)**2+sigma**2))

def Moffat(x,center,ampH,sigma,beta):
    amp=ampH*beta
    return amp*(((x-center)/sigma)**2+1)**(-beta)

def SplitLorentzian(x, center, ampH, sigma, sigma_r):
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

# ---------------------------------------------------------------------------
# FONCTIONS D'AMPLITUDE (à partir de la hauteur ampH)
# ---------------------------------------------------------------------------

def amp_pseudovoigt(ampH: float, fraction: float, sigma: float) -> float:
    return ampH / (((1 - fraction)) / (sigma * sqrt(np.pi / log(2))) +
                   (fraction) / (np.pi * sigma))


def amp_gaussian(ampH: float, sigma: float) -> float:
    return ampH * sigma * sqrt(2 * np.pi)


def amp_moffat(ampH: float, beta_m: float) -> float:
    return ampH * beta_m


def amp_splitlor(ampH: float, sigma_r: float, sigma: float) -> float:
    return ampH / (2 * np.pi) * (sigma + sigma_r)


def amp_pearsonIV_simple(ampH: float, sigma: float) -> float:
    """
    Version simplifiée pour l'amplitude intégrée.
    Si tu veux ta formule exacte avec beta/gamma, tu pourras la remettre ici.
    """
    return amp_gaussian(ampH, sigma)


# ---------------------------------------------------------------------------
# Helper : nom des paramètres "spécifiques" selon le modèle
# ---------------------------------------------------------------------------

def _coef_param_names(model_fit: str) -> List[str]:
    if model_fit == "PseudoVoigt":
        return ["fraction"]
    if model_fit == "Moffat":
        return ["beta"]
    if model_fit == "SplitLorentzian":
        return ["sigma_r"]
    if model_fit == "Gaussian":
        return []
    if model_fit == "PearsonIV":
        # cohérent avec ton ancien code (name_coef_spe=["expon","skew"])
        return ["expon", "skew"]
    return []


# ---------------------------------------------------------------------------
# 1. Création d'un PeakData à partir de valeurs physiques
# ---------------------------------------------------------------------------

def peak_create(
    name: str = '',
    ctr: float = 0.0,
    ampH: float = 1.0,
    coef_spe: Optional[List[float]] = None,
    sigma: float = 0.25,
    inter: float = 3.0,
    model_fit: str = "PseudoVoigt",
    delta_ctr: float = 0.5,
) -> PeakData:
    """
    Construit un PeakData inerte à partir de valeurs centrales.
    Pas de bornes, pas de Model ici : uniquement les params["..."]["value"].
    """
    name_clean = name.replace("-", "_")

    if coef_spe is None:
        # choix génériques de départ
        if model_fit == "PseudoVoigt":
            coef_spe = [0.5]
        elif model_fit == "Moffat":
            coef_spe = [1.0]
        elif model_fit == "SplitLorentzian":
            coef_spe = [sigma]   # sigma_r ~ sigma
        elif model_fit == "PearsonIV":
            coef_spe = [2.0, 0.0]  # m, skew
        else:
            coef_spe = []

    coef_names = _coef_param_names(model_fit)
    coef_spe = list(coef_spe)

    # on remplit le dico params
    params: Dict[str, Dict[str, Any]] = {
        "center": {"value": float(ctr), "error": None},
        "sigma":  {"value": float(sigma), "error": None},
        "ampH":   {"value": float(ampH), "error": None},
    }

    for i, cname in enumerate(coef_names):
        if i < len(coef_spe):
            params[cname] = {"value": float(coef_spe[i]), "error": None}
        else:
            # au cas où coef_spe est plus court
            params[cname] = {"value": 0.0, "error": None}

    peak = PeakData(
        name=name_clean,
        model_fit=model_fit,
        params=params,
        inter=inter,
        delta_ctr=delta_ctr,
        sigma_min_factor=0.05,
        sigma_max_factor=8.0,
        best_fit=None,
    )
    return peak


# ---------------------------------------------------------------------------
# 2. Calcul des bornes à partir d'un PeakData
# ---------------------------------------------------------------------------

def peak_compute_bounds(peak: PeakData) -> Dict[str, tuple]:
    """
    Calcule des bornes min/max pour chaque paramètre de peak.params
    en utilisant peak.inter, peak.delta_ctr, peak.sigma_min_factor, sigma_max_factor.
    """
    inter = peak.inter
    inter_min = max(1 - inter, 0)

    p = peak.params
    bounds: Dict[str, tuple] = {}

    # center
    if "center" in p:
        c = p["center"]["value"]
        bounds["center"] = (c - peak.delta_ctr, c + peak.delta_ctr)

    # sigma
    if "sigma" in p:
        s = p["sigma"]["value"]
        sigma_phys_min = s * peak.sigma_min_factor
        sigma_phys_max = s * peak.sigma_max_factor
        smin = max(s * inter_min, sigma_phys_min)
        smax = min(s * (1 + inter), sigma_phys_max)
        bounds["sigma"] = (smin, smax)

    # ampH
    if "ampH" in p:
        aH = p["ampH"]["value"]
        bounds["ampH"] = (aH * inter_min, aH * (1 + inter))

    # autres paramètres (coef_spe, etc.)
    core = {"center", "sigma", "ampH"}
    for name, info in p.items():
        if name in core:
            continue
        v = info["value"]
        bounds[name] = (v * inter_min, v * (1 + inter))

    # ajustements spécifiques au modèle (ex : fraction dans [0,1])
    if peak.model_fit == "PseudoVoigt" and "fraction" in bounds:
        vmin, vmax = bounds["fraction"]
        vmin = max(0.0, vmin)
        vmax = min(1.0, vmax)
        vmax = max(vmax, 0.1)  # reproduit l'idée de ton code
        bounds["fraction"] = (vmin, vmax)

    if peak.model_fit == "PearsonIV":
        # tu peux affiner ici (m>0.5, skew borné, etc.)
        if "expon" in bounds:
            mmin, mmax = bounds["expon"]
            mmin = max(0.5, mmin)
            bounds["expon"] = (mmin, max(mmax, mmin + 0.01))

    return bounds


# ---------------------------------------------------------------------------
# 3. Construction d'un lmfit.Model à partir d'un PeakData
# ---------------------------------------------------------------------------

def peak_build_lmfit_model(peak: PeakData):
    """
    Construit un lmfit.Model *éphémère* + params à partir d'un PeakData.
    PeakData reste complètement inerte.
    Retourne (model, params, coef_names).
    """
    prefix = peak.name
    coef_names = _coef_param_names(peak.model_fit)
    p = peak.params
    b = peak_compute_bounds(peak)
    inter = peak.inter
    inter_min = max(1 - inter, 0)

    # choix du modèle
    if peak.model_fit == "PseudoVoigt":
        model = PseudoVoigtModel(prefix=prefix)
    elif peak.model_fit == "Moffat":
        model = MoffatModel(prefix=prefix)
    elif peak.model_fit == "SplitLorentzian":
        model = SplitLorentzianModel(prefix=prefix)
    elif peak.model_fit == "Gaussian":
        model = GaussianModel(prefix=prefix)
    elif peak.model_fit == "PearsonIV":
        model = Pearson4Model(prefix=prefix)
    else:
        raise ValueError(f"model_fit inconnu : {peak.model_fit}")

    # valeurs centrales
    center = p["center"]["value"]
    sigma = p["sigma"]["value"]
    ampH = p["ampH"]["value"]

    # paramètres de forme
    coef_vals = [p[name]["value"] for name in coef_names] if coef_names else []

    # amplitude intégrée (en fonction du modèle)
    if peak.model_fit == "PseudoVoigt":
        frac = coef_vals[0] if coef_vals else 0.5
        amp = amp_pseudovoigt(ampH, frac, sigma)
    elif peak.model_fit == "Gaussian":
        amp = amp_gaussian(ampH, sigma)
    elif peak.model_fit == "Moffat":
        beta_m = coef_vals[0] if coef_vals else 1.0
        amp = amp_moffat(ampH, beta_m)
    elif peak.model_fit == "SplitLorentzian":
        sigma_r = coef_vals[0] if coef_vals else sigma
        amp = amp_splitlor(ampH, sigma_r, sigma)
    elif peak.model_fit == "PearsonIV":
        amp = amp_pearsonIV_simple(ampH, sigma)
    else:
        amp = ampH

    # bornes sur amplitude
    amp_min = amp * inter_min
    amp_max = amp * (1 + inter)

    # hints shape params
    for i, name_spe in enumerate(coef_names):
        v = coef_vals[i]
        vmin, vmax = b.get(name_spe, (v * inter_min, v * (1 + inter)))
        model.set_param_hint(prefix + name_spe, value=v, min=vmin, max=vmax)

    # hints amp / sigma / center
    model.set_param_hint(prefix + 'amplitude',
                         value=amp,
                         min=amp_min,
                         max=amp_max)

    if "sigma" in b:
        model.set_param_hint(prefix + 'sigma',
                             value=sigma,
                             min=b["sigma"][0],
                             max=b["sigma"][1])
    else:
        model.set_param_hint(prefix + 'sigma', value=sigma)

    if "center" in b:
        model.set_param_hint(prefix + 'center',
                             value=center,
                             min=b["center"][0],
                             max=b["center"][1])
    else:
        model.set_param_hint(prefix + 'center', value=center)

    params = model.make_params()
    return model, params, coef_names


# ---------------------------------------------------------------------------
# 4. Update "manuel" d'un paramètre dans PeakData
# ---------------------------------------------------------------------------

def peak_set_param(
    peak: PeakData,
    name: str,
    value: float,
    error: Optional[float] = None,
    in_place: bool = True,
) -> PeakData:
    """
    Met à jour un paramètre dans PeakData.params (center, sigma, ampH, fraction, ...).
    """
    if not in_place:
        peak = copy.deepcopy(peak)

    if name not in peak.params:
        peak.params[name] = {"value": float(value), "error": error}
    else:
        peak.params[name]["value"] = float(value)
        peak.params[name]["error"] = error

    return peak


# ---------------------------------------------------------------------------
# 5. Mise à jour à partir du résultat lmfit (équivalent Out_model)
# ---------------------------------------------------------------------------

def peak_update_from_lmfit(
    peak: PeakData,
    out,
    coef_names: Optional[List[str]] = None,
    store_best_fit: Optional[np.ndarray] = None,
    in_place: bool = True,
) -> PeakData:
    """
    Met à jour PeakData.params à partir d'un résultat lmfit (out).
    - centre, sigma, ampH (height)
    - coef_spe (fraction, beta, sigma_r, expon, skew, ...)
    Optionnel : stocke aussi best_fit dans peak.best_fit.
    """
    if not in_place:
        peak = copy.deepcopy(peak)

    prefix = peak.name
    if coef_names is None:
        coef_names = _coef_param_names(peak.model_fit)

    # center
    if prefix + "center" in out.params:
        p = out.params[prefix + "center"]
        peak = peak_set_param(peak, "center", p.value, p.stderr)

    # sigma
    if prefix + "sigma" in out.params:
        p = out.params[prefix + "sigma"]
        peak = peak_set_param(peak, "sigma", p.value, p.stderr)

    # ampH = height
    if prefix + "height" in out.params:
        p = out.params[prefix + "height"]
        peak = peak_set_param(peak, "ampH", p.value, p.stderr)

    # coef_spe
    for cname in coef_names:
        full_name = prefix + cname
        if full_name in out.params:
            p = out.params[full_name]
            peak = peak_set_param(peak, cname, p.value, p.stderr)

    # best_fit éventuel
    if store_best_fit is not None:
        peak.best_fit = np.array(store_best_fit)

    return peak


# ---------------------------------------------------------------------------
# 6. Helper "lecture" simple (équivalent de ton Out_model retournant une liste)
# ---------------------------------------------------------------------------

def peak_get_params_for_export(
    peak: PeakData,
    coef_names: Optional[List[str]] = None,
):
    """
    Retourne (center, ampH, sigma, [coef_spe...]) pour export rapide.
    """
    if coef_names is None:
        coef_names = _coef_param_names(peak.model_fit)

    center = peak.params["center"]["value"]
    sigma = peak.params["sigma"]["value"]
    ampH = peak.params["ampH"]["value"]
    coef_spe = [peak.params[name]["value"] for name in coef_names if name in peak.params]

    return center, ampH, sigma, np.array(coef_spe, dtype=float)
