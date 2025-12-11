# Bibli/Data/RunData.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from Bibli.Data.SpectrumData import SpectrumData
from Bibli.Data.GaugeData import GaugeData


@dataclass
class RunData:
    """
    Données inertes d'une expérience CED :
    - aucun traitement
    - pas de lecture disque
    - pas de fit
    """

    # Données spectres (tableau brut : col0 = x, autres = spectres)
    data_spectres: pd.DataFrame

    # Liste de spectres individuels (SpectrumData)
    spectra: List[SpectrumData] = field(default_factory=list)

    # Jauges "modèle" pour initialiser les spectres
    gauges_init: List[GaugeData] = field(default_factory=list)

    # Nombre de spectres
    N: int = 0

    # Résumé global des fits (P, T, etc.)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Liste des index des spectres effectivement gardés (pour affichage / stockage)
    list_nspec: List[int] = field(default_factory=list)

    # Mode cinétique ou non
    kinetic: bool = False

    # Oscilloscope
    time_index: List[int] = field(default_factory=lambda: [2, 4])
    data_oscillo: Optional[pd.DataFrame] = None
    time_spectrum: Optional[np.ndarray] = None

    # Movie
    folder_movie: Optional[str] = None
    movie: Optional[list] = None        # liste d'images ou None
    list_movie: Optional[List[str]] = None
    time_movie: List[float] = field(default_factory=list)
    fps: Optional[float] = None
    t0_movie: float = 0.0

    # Divers
    ced_path: str = "not_save"
    gauges_select: List[Optional[list]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    help: str = "CEDData : données d'une expérience CED (brutes + résultats)."
