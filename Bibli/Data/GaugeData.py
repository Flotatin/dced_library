# Bibli/Data/Gauge.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from PeakData import PeakData

@dataclass
class GaugeData:
    """
    Données inertes d'une jauge pour un spectre.
    Le type réel de la jauge est déterminé par son nom.
    Toute la logique est dans GaugeUse.
    """
    name: str = ""          # "Ruby", "Sm", "Rhodamine6G", etc.
    name_spe: str = ""      # informations optionnelles (Sig20, Mod:Gaussian, etc.)

    peaks: List[PeakData] = field(default_factory=list)

    # Données du spectre local
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    dY: Optional[np.ndarray] = None

    # Résultats
    lamb_fit: Optional[float] = None
    P: float = 0.0
    sigmaP: float = 0.0
    T: float = 0.0
    fwhm: float = 0.0

    state: str = "Y"

    study: pd.DataFrame = field(default_factory=pd.DataFrame)
    study_add: pd.DataFrame = field(default_factory=pd.DataFrame)

    bit_fit: bool = False

    meta: Dict[str, Any] = field(default_factory=dict)
