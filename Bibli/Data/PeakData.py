# Bibli/Data/Peak.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class PeakData:
    """
    Objet inerte :
    - stocke les paramètres (valeur, erreur éventuelle)
    - aucune logique de calcul
    - aucune dépendance à lmfit
    """
    name: str = ''
    model_fit: str = "PseudoVoigt"

    # Dictionnaire des paramètres du Peak
    # Exemple :
    # params["center"] = {"value": 694.2, "error": 0.03}
    # params["sigma"]  = {"value": 0.25,  "error": 0.01}
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # paramètres généraux pour la construction utilisatrice
    inter: float = 3.0
    delta_ctr: float = 0.5

    # bornes générales de sigma (comme dans ton ancien code)
    sigma_min_factor: float = 0.05
    sigma_max_factor: float = 8.0

    # résultat brut du fit (optionnel)
    best_fit: Optional[np.ndarray] = None
