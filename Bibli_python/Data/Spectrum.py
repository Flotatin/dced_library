import numpy as np
import pandas as pd


class Spectrum:
    def __init__(self, wnb, spec, Gauges=None, type_filtre="svg", param_f=None, deg_baseline=0):
        if Gauges is None:
            Gauges = []
        if param_f is None:
            param_f = [9, 2]

        self.x_raw = np.array(wnb)
        self.y_raw = spec
        self.y_corr = spec
        self.x_corr = wnb
        self.param_f = param_f
        self.deg_baseline = deg_baseline
        self.type_filtre = type_filtre
        self.y_filtre, self.blfit = None, None
        # FIT PIC
        self.Gauges = Gauges
        self.lambda_error = round((self.wnb[-1] - self.wnb[0]) * 0.5 / len(self.wnb), 4)
        # SYNTHESE
        self.study = pd.DataFrame()