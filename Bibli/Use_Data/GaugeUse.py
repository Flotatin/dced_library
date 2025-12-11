# Bibli/Use_Data/GaugeUse.py

import copy
from typing import Optional, List, Tuple, Callable, Dict, Any

import numpy as np
from pynverse import inversefunc
import pandas as pd
from Bibli.Data.GaugeData import GaugeData
from Bibli.Data.PeakData import PeakData
from Bibli.Use_Data.PeakUse import peak_create, peak_set_param

# Import de tes lois de pression (à adapter aux chemins réels)
from Bibli.Physique_Law import *


# ------------------------- TEMPLATES DE JAUGE ------------------------- #
# Tout ce qui dépend uniquement du "type" de jauge est ici.

GAUGE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Ruby": {
        "lamb0": 694.4,
        # (delta_lambda, amp_factor, sigma_factor)
        "peaks": [
            (0.0, 1.0, 1.0),
            (-1.5, 0.75, 1.0),
        ],
        "model_fit": "PseudoVoigt",
        "pressure_law": Ruby_2020_Shen,
        "pressure_domain": (692.0, 750.0),
        "color_print": ['darkred', ['darkred', 'brown']],
    },
    "Sm": {
        "lamb0": 685.39,
        "peaks": [
            (0.0, 1.0, 1.0),
        ],
        "model_fit": "PseudoVoigt",
        "pressure_law": Sm_2015_Rashenko,
        "pressure_domain": (683.0, 730.0),
        "color_print": ['darkblue', ['darkblue']],
    },
    "SrFCl": {
        "lamb0": 690.1,   # à ajuster si besoin
        "peaks": [
            (0.0, 1.0, 1.0),
        ],
        "model_fit": "PseudoVoigt",
        "pressure_law": SrFCl_law,
        "pressure_domain": (683.0, 730.0),
        "color_print": ['darkorange', ['darkorange']],
    },
    "Rhodamine6G": {
        "lamb0": 550.0,
        "peaks": [
            (0.0, 1.0, 1.0),
            (52.0, 0.5, 4.0),   # ton ancien [52, 0.5, 4]
        ],
        "model_fit": "Gaussian",
        "pressure_law": Rhodamine_6G_2024_Dembele,
        "pressure_domain": (530.0, 600.0),
        "color_print": ['limegreen', ['limegreen', 'darkgreen']],
    },
    "Diamond_c12": {
        "lamb0": 1333.0,
        "peaks": [
            (0.0, 1.0, 1.0),
        ],
        "model_fit": "PseudoVoigt",
        "pressure_law": Raman_Dc12__2003_Occelli,
        "pressure_domain": (1323.0, 1533.0),
        "color_print": ['silver', ['silver']],
    },
    "Diamond_c13": {
        "lamb0": 1287.0,
        "peaks": [
            (0.0, 1.0, 1.0),
        ],
        "model_fit": "PseudoVoigt",
        "pressure_law": Raman_Dc13_1997_Schieferl,
        "pressure_domain": (1277.0, 1487.0),
        "color_print": ['dimgrey', ['dimgrey']],
    },
}

def gauge_get_template(gauge: GaugeData) -> Dict[str, Any]:
    if gauge.name not in GAUGE_TEMPLATES:
        raise ValueError(f"Aucun template trouvé pour la jauge : {gauge.name}")
    return GAUGE_TEMPLATES[gauge.name]
# ------------------------- CREATION D'UNE GAUGE ------------------------- #

def gauge_create_from_name(name: str, name_spe: str = "") -> GaugeData:
    tpl = gauge_get_template(name)

    lamb0 = tpl["lamb0"]
    model_fit = tpl["model_fit"]

    peaks = []
    for i, (dlam, amp_fac, sig_fac) in enumerate(tpl["peaks"]):
        ctr = lamb0 + dlam
        sigma = 0.25 * sig_fac

        peak = peak_create(
            name=f"{name}_p{i+1}",
            ctr=ctr,
            ampH=1.0,
            coef_spe=None,
            sigma=sigma,
            inter=3.0,
            model_fit=model_fit,
            delta_ctr=0.5,
        )
        peaks.append(peak)

    return GaugeData(
        name=name,
        name_spe=name_spe,
        peaks=peaks,
        meta={"lamb0": lamb0}
    )


# ------------------------- PATTERN RELATIF SUR LES PICS ------------------------- #

def gauge_compute_pattern(gauge: GaugeData) -> List[Tuple[float, float, float]]:
    """
    Calcule le pattern (delta_center, amp_factor, sigma_factor) actuel à partir des peaks.
    Cela remplace deltaP0i stocké.

    Pattern défini relativement au pic 0 (pic principal).
    """
    if not gauge.peaks:
        return []

    p0 = gauge.peaks[0]
    center0 = p0.params["center"]["value"]
    ampH0 = p0.params["ampH"]["value"]
    sigma0 = p0.params["sigma"]["value"]

    pattern = []
    for p in gauge.peaks:
        c = p.params["center"]["value"]
        a = p.params["ampH"]["value"]
        s = p.params["sigma"]["value"]

        delta_center = c - center0
        amp_factor = a / ampH0 if ampH0 != 0 else 1.0
        sigma_factor = s / sigma0 if sigma0 != 0 else 1.0

        pattern.append((delta_center, amp_factor, sigma_factor))

    return pattern


# ------------------------- UPDATE_FIT (remplace Gauge.Update_Fit) ------------------------- #

def gauge_update_fit(
    gauge: GaugeData,
    ctr_main: float,
    ampH_main: float,
    sigma_main: Optional[float] = None,
    coef_spe_main: Optional[List[float]] = None,
    inter: Optional[float] = None,
    model_fit: Optional[str] = None,
    delta_ctr: Optional[float] = None,
    in_place: bool = True,
) -> GaugeData:
    """
    Applique un nouveau (ctr, ampH, sigma, coef_spe) au pic principal et propage
    aux autres pics selon le pattern relatif courant.

    -> plus besoin de deltaP0i dans GaugeData.
    """
    if not in_place:
        gauge = copy.deepcopy(gauge)

    if not gauge.peaks:
        return gauge

    pattern = gauge_compute_pattern(gauge)

    # si sigma_main non fourni, on prend le sigma du pic principal
    if sigma_main is None:
        sigma_main = gauge.peaks[0].params["sigma"]["value"]

    for i, peak in enumerate(gauge.peaks):
        d_center, amp_fac, sigma_fac = pattern[i]

        center_i = ctr_main + d_center
        ampH_i = ampH_main * amp_fac
        sigma_i = sigma_main * sigma_fac

        # update des params de base
        peak = peak_set_param(peak, "center", center_i)
        peak = peak_set_param(peak, "ampH", ampH_i)
        peak = peak_set_param(peak, "sigma", sigma_i)

        # changement de modèle éventuel
        if model_fit is not None and model_fit != peak.model_fit:
            # on recrée un PeakData avec le même center/ampH/sigma, mais un autre model_fit
            peak = peak_create(
                name=peak.name,
                ctr=center_i,
                ampH=ampH_i,
                coef_spe=coef_spe_main,
                sigma=sigma_i,
                inter=peak.inter if inter is None else inter,
                model_fit=model_fit,
                delta_ctr=peak.delta_ctr if delta_ctr is None else delta_ctr,
            )
        else:
            # on ne change que inter / delta_ctr si fourni
            if inter is not None:
                peak.inter = inter
            if delta_ctr is not None:
                peak.delta_ctr = delta_ctr

            # param spécifiques (fraction, beta, etc.)
            if coef_spe_main is not None:
                # noms des coef_spe selon le modèle
                from Bibli.Use_Data.PeakUse import _coef_param_names
                coef_names = _coef_param_names(peak.model_fit)
                for cname, val in zip(coef_names, coef_spe_main):
                    peak = peak_set_param(peak, cname, val)

        gauge.peaks[i] = peak

    return gauge


# ------------------------- LOI DE PRESSION / COULEURS ------------------------- #

def gauge_get_pressure_law(gauge: GaugeData) -> Tuple[Callable, Callable]:
    """
    Renvoie (f_P, inv_f_P) en fonction de gauge.gauge_type.
    Rien n'est stocké dans GaugeData.
    """
    tpl = gauge_get_template(gauge.gauge_type)
    f_P = tpl["pressure_law"]
    domain = tpl["pressure_domain"]
    inv_f = inversefunc(lambda x: f_P(x), domain=domain)
    return f_P, inv_f


def gauge_get_colors(gauge: GaugeData):
    """Couleurs pour affichage (global, liste des pics)."""
    tpl = gauge_get_template(gauge.gauge_type)
    return tpl["color_print"]


def gauge_compute(
    gauge: GaugeData,
    all_gauges: list = None,
    lambda_error: float = 0.0,
):
    """
    Version finale et propre du calcul pour une jauge spectro.
    - lit la valeur + erreur des PeakData (params["center"]["value/error"])
    - calcule lambda_fit, P, sigmaP, fwhm
    - gère les cas Ruby, RuT, RuSmT
    """

    # --------------------------------------------------------------
    #   1) Vérification minimale
    # --------------------------------------------------------------
    if not gauge.bit_fit or len(gauge.peaks) == 0:
        print(f"NO FIT for gauge {gauge.name}, skip compute.")
        return gauge

    # --------------------------------------------------------------
    #   2) Extraction du pic principal
    # --------------------------------------------------------------
    main_peak = gauge.peaks[0]

    ctr = float(main_peak.params["center"]["value"])
    gauge.lamb_fit = round(ctr, 3)

    sigma0 = float(main_peak.params["sigma"]["value"])
    gauge.fwhm = round(sigma0, 5)

    # Erreur sur lambda = erreur du pic principal ou fallback
    sigmal = float(main_peak.params["center"].get("error", lambda_error))
    if sigmal < lambda_error:
        sigmal = lambda_error

    # --------------------------------------------------------------
    #   3) Loi de pression
    # --------------------------------------------------------------
    tpl = gauge_get_template(gauge)
    f_P = tpl["pressure_law"]
    lamb0 = tpl["lamb0"]

    P, sigmaP = f_P(gauge.lamb_fit, lamb0, sigmalambda=sigmal)

    if gauge.state == "Y":
        gauge.P = float(P)
        gauge.sigmaP = float(sigmaP)
    else:
        gauge.P = 0.0
        gauge.sigmaP = 0.0
        gauge.fwhm = 0.1

    # --------------------------------------------------------------
    #   4) Δλ entre pics (equivalent deltaP0i!)
    # --------------------------------------------------------------
    deltas = []
    for i in range(1, len(gauge.peaks)):
        c_i = float(gauge.peaks[i].params["center"]["value"])
        deltas.append(round(c_i - gauge.lamb_fit, 5))

    # --------------------------------------------------------------
    #   5) Ruby : Deltap12
    # --------------------------------------------------------------
    gauge.study_add = pd.DataFrame()

    if "Ru" in gauge.name_spe and len(gauge.peaks) >= 2:
        # Δλ12 = pic2 - pic1
        c1 = float(gauge.peaks[0].params["center"]["value"])
        c2 = float(gauge.peaks[1].params["center"]["value"])
        delt12 = round(c2 - c1, 5)

        gauge.study_add = pd.DataFrame(
            np.array([[abs(delt12)]]),
            columns=["Deltap12"]
        )

        # ----------------------------------------------------------
        #   5b) RuSmT : calcul de la température Ruby–Sm
        # ----------------------------------------------------------
        if "SmT" in gauge.name_spe and all_gauges is not None:
            sm_index = gauge.meta.get("spe", None)
            if sm_index is not None and 0 <= sm_index < len(all_gauges):
                sm_g = all_gauges[sm_index]

                # valeur du pic Sm
                lamb_fit_S = float(sm_g.peaks[0].params["center"]["value"])

                # erreur sur pic Sm
                sigmals = float(sm_g.peaks[0].params["center"].get("error", lambda_error))
                if sigmals < lambda_error:
                    sigmals = lambda_error

                # λ0 Ruby / Sm
                tpl_sm = gauge_get_template(sm_g)
                lamb0S = tpl_sm["lamb0"]

                tpl_ru = gauge_get_template(gauge)
                lamb0R = tpl_ru["lamb0"]
                
                # calcul T
                T, sigmaT = T_Ruby_Sm_1997_Datchi(
                    gauge.lamb_fit,
                    lamb_fit_S,
                    lamb0S=lamb0S,
                    lamb0R=lamb0R,
                    sigmalambda=[sigmal, sigmals],
                )

                gauge.T = float(T)

                # remplace study_add par tableau complet
                gauge.study_add = pd.DataFrame(
                    np.array([[delt12, T, sigmaT]]),
                    columns=["Deltap12", "T", "sigma_T"]
                )

    # --------------------------------------------------------------
    #   6) Construction du tableau principal
    # --------------------------------------------------------------
    base_df = pd.DataFrame(
        np.array([[gauge.P, gauge.sigmaP, gauge.lamb_fit, gauge.fwhm, gauge.state]]),
        columns=[
            f"P_{gauge.name}",
            f"sigma_P_{gauge.name}",
            f"lambda_{gauge.name}",
            f"fwhm_{gauge.name}",
            f"State_{gauge.name}",
        ]
    )

    # Merge final
    gauge.study = (
        pd.concat([base_df, gauge.study_add], axis=1)
        if not gauge.study_add.empty else base_df
    )

    return gauge
