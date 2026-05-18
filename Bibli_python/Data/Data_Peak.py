
from dataclasses import dataclass, field


@dataclass
class data_Peak:
    """
    Conteneur de données pour un pic.

    name  : nom logique du pic (préfixe lmfit).
    model : nom du modèle ('PseudoVoigt', 'Gaussian', ...).
    params: dict param -> (value, spread)
            - spread = sigma -> on fabrique min/max avec l'argument `inter`
            - OU spread = (min, max) -> utilisé tel quel.
    """

    name: str
    model: str
    params: dict = field(default_factory=dict)