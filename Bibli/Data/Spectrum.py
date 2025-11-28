from dataclasses import dataclass, field
from typing import List, Optional

from Bibli.Data.Data_Gauge import Data_Gauge


@dataclass
class Spectrum:
    """Données brutes d'un spectre (x, y) et méta-informations simples."""

    wnb: List[float]
    spec: List[float]
    Gauges: List[Data_Gauge] = field(default_factory=list)
    type_filtre: str = "svg"
    param_f: List[float] = field(default_factory=lambda: [9, 2])
    deg_baseline: int = 0
    E: Optional[float] = None
