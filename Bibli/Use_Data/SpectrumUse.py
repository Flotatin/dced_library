# Bibli/Use_Data/SpectrumUse.py

import copy
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import peakutils as pk
from scipy.signal import savgol_filter

from Bibli.Data.SpectrumData import SpectrumData
from Bibli.Use_Data.GaugeUse import (
    gauge_compute,
    gauge_get_template,
)
from Bibli.Use_Data.PeakUse import (
    peak_build_lmfit_model,
    peak_update_from_lmfit,
)

# -------------------------------------------------------------------------
# 1) CREATION + PRETRAITEMENT DU SPECTRE
# -------------------------------------------------------------------------

def spectrum_create(
    x_raw,
    y_raw,
    gauges_init=None,
    type_filtre="svg",
    param_f=None,
    deg_baseline=0,
) -> SpectrumData:
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    if param_f is None:
        param_f = [9, 2]

    gauges = copy.deepcopy(gauges_init) if gauges_init is not None else []

    spe = SpectrumData(
        x_raw=x_raw,
        y_raw=y_raw,
        type_filtre=type_filtre,
        param_f=param_f,
        deg_baseline=deg_baseline,
        gauges=gauges,
    )

    spectrum_preprocess(spe)
    return spe

def spectrum_preprocess(
    spe: SpectrumData,
    deg_baseline: Optional[int] = None,
    type_filtre: Optional[str] = None,
    param_f: Optional[List[float]] = None,
) -> SpectrumData:
    if deg_baseline is not None:
        spe.deg_baseline = int(deg_baseline)
    if type_filtre is not None:
        spe.type_filtre = type_filtre
    if param_f is not None:
        spe.param_f = list(param_f)

    x = spe.x_raw
    y = spe.y_raw

    # ------------------- Baseline ------------------- #
    baseline = pk.baseline(y, deg=spe.deg_baseline)
    baseline = np.asarray(baseline, dtype=float)

    if spe.deg_baseline == 0:
        offset = np.min(y) - baseline[0]
        if offset < 0:
            baseline = baseline + offset * 1.05

    spe.baseline = baseline

    # ------------------- Filtrage ------------------- #
    if spe.type_filtre == "svg":
        win, poly = int(spe.param_f[0]), int(spe.param_f[1])
        if win < 3:
            win = 3
        if win % 2 == 0:
            win += 1
        y_filtered = savgol_filter(y, window_length=win, polyorder=poly)

    elif spe.type_filtre == "fft":
        spectre_fft = np.fft.fft(y)
        freq = np.fft.fftfreq(len(y), d=(x[1] - x[0]))

        low, high = spe.param_f
        mask = (np.abs(freq) > low) & (np.abs(freq) < high)
        spectre_fft_filtered = spectre_fft.copy()
        spectre_fft_filtered[mask] = 0

        y_filtered = np.real(np.fft.ifft(spectre_fft_filtered))

    else:
        y_filtered = y.copy()

    spe.y_filtered = y_filtered

    # ------------------- Correction spectre ------------------- #
    spe.y_corr = y_filtered - baseline
    spe.x_corr = x.copy()

    # ------------------- lambda_error ------------------- #
    if len(x) > 1:
        spe.lambda_error = round(abs(x[-1] - x[0]) * 0.5 / len(x), 4)
    else:
        spe.lambda_error = 0.0

    return spe

# -------------------------------------------------------------------------
# 2) CALCUL DES JAUGES (P, T, etc.)
# -------------------------------------------------------------------------

def spectrum_compute_gauges(
    spe: SpectrumData,
    mini: bool = False,
) -> SpectrumData:
    if not spe.gauges:
        spe.study = pd.DataFrame()
        return spe

    studies = []
    all_gauges = spe.gauges

    for g in spe.gauges:
        g = gauge_compute(g, all_gauges=all_gauges, lambda_error=spe.lambda_error)
        studies.append(g.study)

    spe.study = pd.concat(studies, axis=1) if studies else pd.DataFrame()
    return spe

# -------------------------------------------------------------------------
# 3) FIT D’UNE SEULE JAUGE (équivalent FIT_One_Jauge)
# -------------------------------------------------------------------------

def _get_x_y_for_fit(spe: SpectrumData):
    x = spe.x_corr if spe.x_corr is not None else spe.x_raw
    if spe.y_corr is not None:
        y = spe.y_corr
    else:
        y = spe.y_raw
    return x, y

def spectrum_fit_one_gauge(
    spe: SpectrumData,
    idx_gauge: int,
    wnb_range: float = 3.0,
    manuel: bool = False,
) -> SpectrumData:
    """
    Fit d’une jauge :
    - sélectionne une fenêtre autour λ0
    - construit un modèle lmfit pour tous les pics de la jauge
    - effectue le fit
    - met à jour les PeakData
    - met à jour gauge.bit_fit et lamb_fit
    """

    # ------------------ Vérifications ------------------
    if idx_gauge < 0 or idx_gauge >= len(spe.gauges):
        print("Index jauge invalide.")
        return spe

    gauge = spe.gauges[idx_gauge]

    if not gauge.peaks:
        print("Jauge sans pics.")
        return spe

    # ------------------ Position initiale ------------------
    tpl = gauge_get_template(gauge)
    lamb0 = tpl["lamb0"]

    x = spe.x_corr
    y = spe.y_corr

    # Fenêtre de travail
    mask = (x > lamb0 - wnb_range) & (x < lamb0 + wnb_range)
    x_sub = x[mask]
    y_sub = y[mask]

    if len(x_sub) < 6:
        print("Fenêtre de fit trop petite.")
        return spe

    # ------------------ Construction du modèle total ------------------
    total_model = None
    total_params = None
    peaks_info = []   # pour stocker (peak, coef_names)

    for peak in gauge.peaks:
        model_i, params_i, coef_names = peak_build_lmfit_model(peak)
        peaks_info.append((peak, coef_names))

        if total_model is None:
            total_model = model_i
            total_params = params_i
        else:
            total_model = total_model + model_i
            total_params.update(params_i)

    # ------------------ Fit ------------------
    try:
        out = total_model.fit(y_sub, total_params, x=x_sub)
    except Exception as e:
        print("Erreur fit jauge :", e)
        return spe

    # ------------------ Mise à jour des pics ------------------
    for k, (peak, coef_names) in enumerate(peaks_info):
        best_fit_sub = out.best_fit  # la somme totale

        peak = peak_update_from_lmfit(
            peak,
            out,
            coef_names=coef_names,
            store_best_fit=best_fit_sub
        )

        gauge.peaks[k] = peak

    # ------------------ Mise à jour jauge ------------------
    # Pic principal = first peak
    gauge.bit_fit = True
    gauge.lamb_fit = gauge.peaks[0].params["center"]["value"]
    gauge.X = x_sub
    gauge.Y = out.best_fit + spe.baseline[mask]
    gauge.dY = out.best_fit - y_sub
    gauge.indexX = np.where(mask)[0]

    return spe

# -------------------------------------------------------------------------
# 4) FIT DE TOUTES LES JAUGES + CALCUL (équivalent FIT)
# -------------------------------------------------------------------------

def spectrum_fit_all_gauges(
    spe: SpectrumData,
    wnb_range: float = 2.0,
) -> SpectrumData:
    for i, G in enumerate(spe.gauges):
        if G.state == "Y":
            try:
                spe = spectrum_fit_one_gauge(
                    spe,
                    num_jauge=i,
                    peakMax0=G.lamb_fit,
                    wnb_range=wnb_range,
                )
            except Exception as e:
                G.state = "IN_NOISE"
                print("error:", e, "in fit of :", G.name)
                spe.gauges[i] = G
        else:
            G.bit_fit = False

    spe = spectrum_compute_gauges(spe)
    spe.bit_fit = True
    return spe

# -------------------------------------------------------------------------
# 5) FIT GLOBAL PAR curve_fit (équivalent FIT_Curv)
# -------------------------------------------------------------------------

def spectrum_fit_curv(
    spe: SpectrumData,
    inter: float = 1.0,
) -> SpectrumData:
    """
    Fit global par courbe (ancien FIT_Curv), réécrit avec la nouvelle archi :
    - somme de tous les pics de toutes les jauges (lmfit)
    - fit sur une fenêtre x_min..x_max englobant tous les pics
    - mise à jour de chaque PeakData
    - reconstruction des contributions par jauge
    - calcul final des P, sigmaP, etc. via spectrum_compute_gauges
    """

    # ------------------------------------------------------------------
    # 0) Vérifs de base
    # ------------------------------------------------------------------
    if not spe.gauges:
        print("spectrum_fit_curv : aucun gauge dans le spectre.")
        return spe

    x = spe.x_corr
    y = spe.y_corr
    if x is None or y is None:
        print("spectrum_fit_curv : spectre non prétraité (x_corr / y_corr manquants).")
        return spe

    # ------------------------------------------------------------------
    # 1) Déterminer la fenêtre globale à fitter (x_min, x_max)
    #    à partir des centres + sigma de tous les pics
    # ------------------------------------------------------------------
    x_min = np.inf
    x_max = -np.inf

    for g in spe.gauges:
        for p in g.peaks:
            c = float(p.params["center"]["value"])
            s = float(p.params["sigma"]["value"])
            x_min = min(x_min, c - 5.0 * s)
            x_max = max(x_max, c + 5.0 * s)

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        print("spectrum_fit_curv : fenêtre globale invalide.")
        return spe

    mask = (x >= x_min) & (x <= x_max)
    x_sub = x[mask]
    y_sub = y[mask]

    if len(x_sub) < 10:
        print("spectrum_fit_curv : trop peu de points dans la fenêtre globale.")
        return spe

    # ------------------------------------------------------------------
    # 2) Construire le modèle lmfit global = somme de tous les pics
    # ------------------------------------------------------------------
    total_model = None
    total_params = None

    # pour retrouver les modèles individuels par gauge/pic
    peak_models = []  # liste de (i_gauge, i_peak, model_i, coef_names)

    for ig, g in enumerate(spe.gauges):
        for ip, peak in enumerate(g.peaks):
            # au besoin, on peut ajuster inter ici
            peak.inter = inter

            model_i, params_i, coef_names = peak_build_lmfit_model(peak)
            peak_models.append((ig, ip, model_i, coef_names))

            if total_model is None:
                total_model = model_i
                total_params = params_i
            else:
                total_model = total_model + model_i
                total_params.update(params_i)

    if total_model is None:
        print("spectrum_fit_curv : aucun modèle de pic construit.")
        return spe

    # ------------------------------------------------------------------
    # 3) Fit global
    # ------------------------------------------------------------------
    try:
        out = total_model.fit(y_sub, total_params, x=x_sub)
    except Exception as e:
        print("spectrum_fit_curv : erreur dans le fit global :", e)
        return spe

    best_sum = out.best_fit  # somme de tous les pics

    # ------------------------------------------------------------------
    # 4) Mise à jour de chaque PeakData à partir du fit
    #    + récupération des best_fit individuels
    # ------------------------------------------------------------------
    # On va aussi reconstruire, pour chaque gauge, la somme de ses pics
    # afin d'avoir gauge.Y, gauge.dY, etc.
    nb_gauges = len(spe.gauges)
    gauge_contribs = [np.zeros_like(x_sub) for _ in range(nb_gauges)]

    for (ig, ip, model_i, coef_names) in peak_models:
        gauge = spe.gauges[ig]
        peak = gauge.peaks[ip]

        # contribution de ce pic seul
        peak_fit = model_i.eval(out.params, x=x_sub)

        # mise à jour des paramètres du pic
        peak = peak_update_from_lmfit(
            peak,
            out,
            coef_names=coef_names,
            store_best_fit=peak_fit,  # best_fit uniquement pour ce pic
        )

        gauge.peaks[ip] = peak
        gauge_contribs[ig] += peak_fit

    # ------------------------------------------------------------------
    # 5) Mise à jour des jauges (X, Y, dY, lamb_fit, bit_fit, indexX)
    # ------------------------------------------------------------------
    idx_global = np.where(mask)[0]
    baseline_sub = spe.baseline[mask] if spe.baseline is not None else np.zeros_like(x_sub)

    for ig, g in enumerate(spe.gauges):
        if not g.peaks:
            continue

        g.X = x_sub
        g.Y = gauge_contribs[ig] + baseline_sub
        g.dY = gauge_contribs[ig] - y_sub
        g.indexX = idx_global
        g.bit_fit = True

        # lamb_fit = centre du pic principal (peak 0)
        g.lamb_fit = float(g.peaks[0].params["center"]["value"])

    # ------------------------------------------------------------------
    # 6) Mise à jour globale du spectre
    # ------------------------------------------------------------------
    spe.X = x_sub
    spe.Y = best_sum + baseline_sub
    spe.dY = best_sum - y_sub
    spe.bit_fit = True

    # ------------------------------------------------------------------
    # 7) Calcul pression / T pour chaque jauge
    # ------------------------------------------------------------------
    spe = spectrum_compute_gauges(spe)

    return spe

from matplotlib import pyplot as plt

def spectrum_print(
    spe: SpectrumData,
    ax: Optional["plt.Axes"] = None,
    ax_resid: Optional["plt.Axes"] = None,
    return_fig: bool = False,
):
    """
    Affichage d'un spectre avec :
      - brut
      - baseline
      - (corrigé + baseline)
      - contributions des jauges et des pics (si fit faits)

    Paramètres
    ----------
    spe : SpectrumData
        Spectre à afficher.
    ax : plt.Axes ou None
        Axe principal. Si None, la fonction crée une figure et deux axes (spectre + résidus).
    ax_resid : plt.Axes ou None
        Axe pour les résidus normalisés. Ignoré si ax est None et return_fig=False.
    return_fig : bool
        Si True, retourne la figure (pour intégration dans GUI, etc.)

    Retour
    ------
    fig ou (ax, ax_resid) ou None
        Selon return_fig et si des axes sont fournis.
    """
    # -------------------- Préparation des données de base -------------------- #
    # x à afficher
    x_plot = spe.x_corr if spe.x_corr is not None else spe.x_raw
    if x_plot is None:
        print("spectrum_print : pas de x à afficher.")
        return

    # brut
    y_raw = spe.y_raw

    # baseline
    if spe.baseline is not None:
        blfit = spe.baseline
    else:
        blfit = np.zeros_like(x_plot)

    # corrigé + baseline (équivalent y_corr + blfit)
    if spe.y_corr is not None:
        y_corr_plus_bl = spe.y_corr + blfit
    else:
        y_corr_plus_bl = y_raw

    # -------------------- Création des axes si nécessaire -------------------- #
    created_fig = False
    if ax is None:
        created_fig = True
        fig, (ax, ax_resid) = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(8, 4),
            gridspec_kw={"height_ratios": [0.85, 0.15]},
        )
    else:
        fig = ax.figure if hasattr(ax, "figure") else None
        # si ax_resid n'est pas fourni, on ne trace pas les résidus
        # (utile pour embedding dans des GUIs)
    
    # -------------------- Tracé de base -------------------- #
    ax.plot(x_plot, blfit, "-.", c="g", markersize=1, label="Baseline")
    ax.plot(x_plot, y_raw, "-", color="lightgray", markersize=2, label="Brut")
    ax.plot(x_plot, y_corr_plus_bl, ".", color="black", markersize=2, label="Corr + bkg")

    # -------------------- Tracé des jauges / pics (si fits) -------------------- #
    for G in spe.gauges:
        if not getattr(G, "bit_fit", False):
            continue

        # On récupère les couleurs via le template
        try:
            tpl = gauge_get_template(G)
            color_main, color_peaks = tpl["color_print"]
        except Exception:
            color_main, color_peaks = None, None

        # Courbe de fit de la jauge
        if getattr(G, "X", None) is not None and getattr(G, "Y", None) is not None:
            titre_fit = f"{G.name}: λ0={tpl['lamb0']}" if "lamb0" in tpl else G.name
            if color_main is not None:
                ax.plot(G.X, G.Y, "--", label=titre_fit, markersize=1, c=color_main)
            else:
                ax.plot(G.X, G.Y, "--", label=titre_fit, markersize=1)

            # Résidus normalisés sur ax_resid si dispo
            if (ax_resid is not None) and (getattr(G, "dY", None) is not None):
                denom = np.max(np.abs(G.dY)) if np.max(np.abs(G.dY)) != 0 else 1.0
                if color_main is not None:
                    ax_resid.plot(G.X, G.dY / denom, "-", c=color_main)
                else:
                    ax_resid.plot(G.X, G.dY / denom, "-")

        # Pics individuels
        if getattr(G, "peaks", None):
            # baseline sur la fenêtre de la jauge
            if getattr(G, "indexX", None) is not None and spe.baseline is not None:
                bl_sub = spe.baseline[G.indexX]
            else:
                # fallback : interpolation de la baseline sur G.X
                if spe.baseline is not None and getattr(G, "X", None) is not None:
                    bl_sub = np.interp(G.X, x_plot, blfit)
                else:
                    bl_sub = None

            for i, peak in enumerate(G.peaks):
                if getattr(peak, "best_fit", None) is None:
                    continue
                if getattr(G, "X", None) is None or bl_sub is None:
                    continue

                y_p = peak.best_fit + bl_sub

                # couleur du pic
                if (color_peaks is not None) and (i < len(color_peaks)):
                    c_pic = color_peaks[i]
                else:
                    c_pic = None

                # label avec centre du pic
                ctr_val = peak.params.get("center", {}).get("value", None)
                if ctr_val is not None:
                    titre_pic = rf"$p_{{{i+1}}}^{({G.name[0]})} = {ctr_val:.3f}$"
                else:
                    titre_pic = rf"$p_{{{i+1}}}^{({G.name[0]})}$"

                if c_pic is not None:
                    ax.fill_between(
                        G.X,
                        y_p,
                        bl_sub,
                        where=y_p > np.min(y_p),
                        alpha=0.3,
                        label=titre_pic,
                        color=c_pic,
                    )
                else:
                    ax.fill_between(
                        G.X,
                        y_p,
                        bl_sub,
                        where=y_p > np.min(y_p),
                        alpha=0.1,
                    )

    # -------------------- Mise en forme axes principaux -------------------- #
    ax.minorticks_on()
    ax.tick_params(which="major", length=10, width=1.5, direction="in")
    ax.tick_params(which="minor", length=5, width=1.5, direction="in")
    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.set_title(f"Spectre, Δλ = {getattr(spe, 'lambda_error', 0.0)}")
    ax.set_ylabel("Amplitude (U.A.)")
    ax.set_xlim([float(np.min(x_plot)), float(np.max(x_plot))])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Résidus
    if ax_resid is not None:
        ax_resid.axhline(0.0, color="k", ls="-.")
        ax_resid.minorticks_on()
        ax_resid.tick_params(which="major", length=10, width=1.5, direction="in")
        ax_resid.tick_params(which="minor", length=5, width=1.5, direction="in")
        ax_resid.tick_params(which="both", bottom=True, top=True, left=True, right=True)
        ax_resid.set_xlabel(r"$\lambda$ (nm)")
        ax_resid.set_ylabel(r"$(Data - Fit)/max$ (U.A.)")
        ax_resid.set_xlim([float(np.min(x_plot)), float(np.max(x_plot))])
        ax_resid.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    else:
        ax.set_xlabel(r"$\lambda$ (nm)")

    ax.legend(loc="best")

    # -------------------- Retour / affichage -------------------- #
    if return_fig and created_fig:
        return fig
    elif return_fig:
        return ax, ax_resid
    else:
        if created_fig:
            plt.show()
        return
