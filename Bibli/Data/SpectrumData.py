# Bibli/Data/Spectrum.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from Bibli.Data.GaugeData import GaugeData


@dataclass
class SpectrumData:
    """
    Données inertes d'un spectre :
    - x_raw / y_raw : jamais modifiés
    - x_corr / y_corr : versions modifiées par preprocessing
    - y_filtered / baseline : produits intermédiaires
    """

    # Données brutes (immuables)
    x_raw: np.ndarray
    y_raw: np.ndarray

    # Données filtrées / corrigées (modifiées)
    x_corr: Optional[np.ndarray] = None
    y_corr: Optional[np.ndarray] = None

    # Produits intermédiaires
    baseline: Optional[np.ndarray] = None
    y_filtered: Optional[np.ndarray] = None

    # Paramètres
    type_filtre: str = "svg"
    param_f: List[float] = field(default_factory=lambda: [9, 2])
    deg_baseline: int = 0

    # Liste de jauges
    gauges: List[GaugeData] = field(default_factory=list)

    # Résultats globaux
    lambda_error: float = 0.0
    study: pd.DataFrame = field(default_factory=pd.DataFrame)

    meta: Dict[str, Any] = field(default_factory=dict)
