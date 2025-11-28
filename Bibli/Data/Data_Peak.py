from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Data_Peak:
    """Conteneur de données pour un pic spectral.

    Cette classe stocke les valeurs initiales nécessaires pour construire
    un modèle de pic (centre, hauteur, sigma, paramètres spécifiques, etc.)
    sans inclure la logique de fit.
    """

    name: str
    center: float
    height: float = 1.0
    sigma: float = 0.25
    model: str = "PseudoVoigt"
    coef_spe: List[float] = field(default_factory=lambda: [0.5])
    inter: float = 3.0
    delta_ctr: float = 0.5
    amplitude: Optional[float] = None
