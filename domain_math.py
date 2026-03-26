"""Fonctions mathématiques pures réutilisables hors couche UI."""

from __future__ import annotations

import numpy as np
from pynverse import inversefunc

from Bibli_python import CL_FD_Update as CL


def gauge_lambda_from_pressure(gauge, pressure: float) -> tuple[float, float]:
    """Retourne (lambda_fit, delta_lambda_p) à partir de la pression."""
    x = round(float(gauge.inv_f_P(pressure)), 3)
    return x, x - gauge.lamb0


def gauge_pressure_from_lambda(gauge, lambda_fit: float, delta_lambda_t: float) -> float:
    """Retourne la pression à partir de lambda (corrigé température)."""
    return round(float(gauge.f_P(lambda_fit - delta_lambda_t)), 3)


def ruby_lambda_from_temperature(gauge, temperature: float) -> tuple[float, float]:
    """Retourne (lambda_fit, delta_lambda_t) pour une jauge Ruby."""
    x = round(
        float(
            inversefunc(
                lambda x_: CL.T_Ruby_by_P(x_, P=gauge.P, lamb0R=gauge.lamb0),
                temperature,
            )
        ),
        3,
    )
    xp = round(
        float(
            inversefunc(
                lambda x_: CL.Ruby_2020_Shen(x_, lamb0=gauge.lamb0),
                gauge.P,
                [690, 720],
            )
        ),
        3,
    )
    return x, x - xp


def fit_score(residuals, *, use_abs: bool = False) -> float:
    """Calcule un score scalaire d'ajustement (plus petit = meilleur)."""
    resid = np.asarray(residuals, dtype=float)
    if use_abs:
        return float(np.sum(np.abs(resid)))
    return float(np.sum(resid**2))
