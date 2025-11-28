from dataclasses import dataclass, field
from typing import Callable, List, Optional

from Bibli.Data.Data_Peak import Data_Peak


@dataclass
class Data_Gauge:
    """Données brutes décrivant une jauge (ensemble de pics)."""

    name: str
    lamb0: Optional[float] = None
    nb_pic: Optional[int] = None
    deltaP0i: List[List[float]] = field(default_factory=list)
    name_spe: str = ""
    f_P: Optional[Callable] = None
    color_print: List[Optional[List[str]]] = field(default_factory=list)
    peaks: List[Data_Peak] = field(default_factory=list)
