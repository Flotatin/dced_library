# Bibli/Use_Data/CEDUse.py

import copy
import os
from typing import Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from Bibli.Data.RunData import RunData
from Bibli.Data.GaugeData import GaugeData
from Bibli.Data.SpectrumData import SpectrumData
from Bibli.Use_Data.SpectrumUse import (
    spectrum_create,
    spectrum_preprocess,
    spectrum_fit_curv,
    spectrum_compute_gauges,
)


class CEDUse:
    """
    Classe "active" qui manipule un RunData :
    - création à partir de fichiers
    - fit des spectres
    - reconstruction du Summary
    - oscillo, movie, print
    """

    def __init__(self, run_data: RunData):
        self.data = run_data

    # ------------------------------------------------------------------
    # FABRIQUE : création d'un RunData à partir des fichiers
    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        data_path: str,
        gauges_init: List[GaugeData],
        N: Optional[int] = None,
        data_oscillo: Optional[str] = None,
        folder_Movie: Optional[str] = None,
        time_index: List[int] = [2, 4],
        fps: Optional[float] = None,
        skiprow_spec: int = 43,
        kinetic: bool = False,
    ) -> "CEDUse":
        """
        Lit le fichier de spectres, prépare un RunData inerte, puis renvoie CEDUse.
        Aucun fit n'est fait ici.
        """

        # Lecture brute des spectres
        df_spec = pd.read_csv(
            data_path,
            sep=r"\s+",
            header=None,
            skipfooter=0,
            skiprows=skiprow_spec,
            engine="python",
        )

        # Cas 2 colonnes (lambda, I) -> reshape en plusieurs spectres
        if len(df_spec.columns) == 2:
            wave = df_spec.iloc[:, 0].to_numpy()
            Iua = df_spec.iloc[:, 1].to_numpy()
            wave_unique = np.unique(wave)
            num_spec = len(wave) // len(wave_unique)

            if num_spec >= 1:
                Iua = Iua.reshape(num_spec, len(wave_unique)).T
                cols = [0] + [i + 1 for i in range(num_spec)]
                df_spec = pd.DataFrame(
                    np.column_stack([wave_unique, Iua]),
                    columns=cols,
                )

        # N = nb de colonnes - 1
        if N is None:
            N = df_spec.shape[1] - 1

        # Oscillo
        if data_oscillo is not None:
            df_osc = pd.read_csv(
                data_oscillo,
                sep=r"\s+",
                skipfooter=0,
                engine="python",
            )
        else:
            df_osc = None

        ced = RunData(
            data_spectres=df_spec,
            spectra=[],
            gauges_init=gauges_init,
            N=N,
            summary=pd.DataFrame(),
            list_nspec=[],
            kinetic=kinetic,
            time_index=time_index,
            data_oscillo=df_osc,
            time_spectrum=None,
            folder_movie=folder_Movie,
            movie=None,
            list_movie=None,
            time_movie=[],
            fps=fps,
            t0_movie=0.0,
            ced_path="not_save",
            gauges_select=[None for _ in range(len(gauges_init))],
        )

        return cls(ced)

    # ------------------------------------------------------------------
    # INITIALISATION DES SPECTRES (avec ou sans fit global)
    # ------------------------------------------------------------------
    def init_spectra(
        self,
        fit: bool = False,
        type_filtre: str = "svg",
        param_f: Optional[List[float]] = None,
        deg_baseline: int = 0,
    ):
        """
        Crée les SpectrumData à partir de data_spectres.
        Si fit=True : applique un fit global (spectrum_fit_curv) à chaque spectre.
        """
        d = self.data

        if param_f is None:
            param_f = [9, 2]

        wave = d.data_spectres.iloc[:, 0].to_numpy()

        d.spectra = []
        d.summary = pd.DataFrame()
        d.list_nspec = []

        if fit:
            new_gauges = copy.deepcopy(d.gauges_init)

            for i in tqdm(range(0, d.N), desc="Spectra + FIT"):
                y = d.data_spectres.iloc[:, i + 1].to_numpy()

                spe = spectrum_create(
                    wave,
                    y,
                    gauges_init=new_gauges,
                    type_filtre=type_filtre,
                    param_f=param_f,
                    deg_baseline=deg_baseline,
                )

                try:
                    spe = spectrum_fit_curv(spe)
                    stu = spe.study
                    new_gauges = copy.deepcopy(spe.gauges)
                except Exception as e:
                    print("ERROR:", e, "\n Spec n°:", str(i))
                    if len(d.spectra) >= 1:
                        spe = d.spectra[-1]
                        stu = spe.study
                        new_gauges = copy.deepcopy(spe.gauges)
                    else:
                        stu = pd.DataFrame()

                row = pd.concat(
                    [pd.DataFrame({"n°Spec": [int(i)]}), stu],
                    axis=1,
                )
                d.summary = pd.concat([d.summary, row], ignore_index=True)

                if (not d.kinetic) or (i % 10 == 0) or (i < 2):
                    d.spectra.append(spe)
                    d.list_nspec.append(int(i))

            print("Spectra loading & global fit DONE")

        else:
            for i in tqdm(range(0, d.N), desc="Spectra loading"):
                y = d.data_spectres.iloc[:, i + 1].to_numpy()

                spe = spectrum_create(
                    wave,
                    y,
                    gauges_init=d.gauges_init,
                    type_filtre=type_filtre,
                    param_f=param_f,
                    deg_baseline=deg_baseline,
                )
                d.spectra.append(spe)
                d.list_nspec.append(int(i))

            print("Spectra loading DONE")

        # Temps / oscillo
        if d.data_oscillo is not None:
            self.extract_time()

        # Movie
        if d.folder_movie is not None:
            self._load_movie()

    # ------------------------------------------------------------------
    # FIT (réanalyse) sur un intervalle de spectres (type ancien FIT)
    # ------------------------------------------------------------------
    def fit_range(
        self,
        end: Optional[int] = None,
        start: int = 0,
        gauges_init: Optional[List[GaugeData]] = None,
        type_filtre: str = "svg",
        param_f: Optional[List[float]] = None,
        deg_baseline: int = 0,
    ):
        d = self.data

        if param_f is None:
            param_f = [9, 2]

        if gauges_init is not None:
            d.gauges_init = gauges_init
        elif d.spectra and start < len(d.spectra):
            d.gauges_init = d.spectra[start].gauges

        d.summary = pd.DataFrame()
        d.spectra = []
        d.list_nspec = []

        if end is None:
            end = d.N
        if end > d.N:
            print("fit_range : end > number of spectra !")
            return

        wave = d.data_spectres.iloc[:, 0].to_numpy()
        new_gauges = copy.deepcopy(d.gauges_init)

        for i in tqdm(range(start, end), desc="Re-FIT"):
            y = d.data_spectres.iloc[:, i + 1].to_numpy()

            spe = spectrum_create(
                wave,
                y,
                gauges_init=new_gauges,
                type_filtre=type_filtre,
                param_f=param_f,
                deg_baseline=deg_baseline,
            )

            try:
                spe = spectrum_fit_curv(spe)
                stu = spe.study
                new_gauges = copy.deepcopy(spe.gauges)
            except Exception as e:
                print("error:", e, "in fit of spectrum:", i)
                if len(d.spectra) > 0:
                    spe = d.spectra[-1]
                    stu = spe.study
                    new_gauges = copy.deepcopy(spe.gauges)
                else:
                    stu = pd.DataFrame()

            row = pd.concat(
                [pd.DataFrame({"n°Spec": [int(i)]}), stu],
                axis=1,
            )
            d.summary = pd.concat([d.summary, row], ignore_index=True)

            if (not d.kinetic and i % 10 == 0) or i < 2:
                d.spectra.append(spe)
                d.list_nspec.append(int(i))

        print("Re-FIT DONE")

    # ------------------------------------------------------------------
    # Corr / Refits rapides en propageant les jauges (type ancien FIT_Corr)
    # ------------------------------------------------------------------
    def fit_corr(
        self,
        end: Optional[int] = None,
        start: int = 0,
        inter: Optional[float] = None,
        type_filtre: str = "svg",
        param_f: Optional[List[float]] = None,
        deg_baseline: int = 0,
    ):
        d = self.data

        if param_f is None:
            param_f = [9, 2]

        if end is None:
            end = d.N
        if end > d.N:
            print("fit_corr : end > number of spectra !")
            return

        wave = d.data_spectres.iloc[:, 0].to_numpy()

        if d.kinetic:
            new_gauges = copy.deepcopy(d.gauges_init)
            for i in tqdm(range(start + 1, end), desc="FIT_Corr Kinetic"):
                y = d.data_spectres.iloc[:, i + 1].to_numpy()

                spe = spectrum_create(
                    wave,
                    y,
                    gauges_init=new_gauges,
                    type_filtre=type_filtre,
                    param_f=param_f,
                    deg_baseline=deg_baseline,
                )
                spe = spectrum_fit_curv(spe, inter=inter if inter is not None else 1.0)
                new_gauges = copy.deepcopy(spe.gauges)

                stu = spe.study
                row = pd.concat(
                    [pd.DataFrame({"n°Spec": [int(i)]}), stu],
                    axis=1,
                )
                if len(d.summary) <= i:
                    d.summary = pd.concat([d.summary, row], ignore_index=True)
                else:
                    d.summary.iloc[i] = row.iloc[0]

            print("FIT_Corr Fast Kinetic DONE")

        else:
            if not d.spectra or start >= len(d.spectra):
                print("fit_corr : start >= len(spectra), rien à faire.")
                return

            new_gauges = copy.deepcopy(d.spectra[start].gauges)
            for i in tqdm(range(start + 1, end), desc="FIT_Corr"):
                y = d.data_spectres.iloc[:, i + 1].to_numpy()

                spe = spectrum_create(
                    wave,
                    y,
                    gauges_init=new_gauges,
                    type_filtre=type_filtre,
                    param_f=param_f,
                    deg_baseline=deg_baseline,
                )
                spe = spectrum_fit_curv(spe, inter=inter if inter is not None else 1.0)
                new_gauges = copy.deepcopy(spe.gauges)

                if i < len(d.spectra):
                    d.spectra[i] = spe

                stu = spe.study
                row = pd.concat(
                    [pd.DataFrame({"n°Spec": [int(i)]}), stu],
                    axis=1,
                )
                if len(d.summary) <= i:
                    d.summary = pd.concat([d.summary, row], ignore_index=True)
                else:
                    d.summary.iloc[i] = row.iloc[0]

            print("FIT_Corr DONE")

    # ------------------------------------------------------------------
    # Reconstruction / correction du Summary
    # ------------------------------------------------------------------
    def corr_summary(self, All: bool = True, num_spec: Optional[int] = None):
        d = self.data

        if All:
            rows = []
            for i, spe in enumerate(d.spectra):
                if spe.study is None or spe.study.empty:
                    spe = spectrum_compute_gauges(spe)
                row = pd.concat(
                    [pd.DataFrame({"n°Spec": [int(i)]}), spe.study],
                    axis=1,
                )
                rows.append(row.iloc[0])

            d.summary = pd.DataFrame(rows).reset_index(drop=True)
        else:
            if num_spec is None or num_spec >= len(d.spectra):
                print("corr_summary: num_spec incorrect.")
                return

            spe = d.spectra[num_spec]
            if spe.study is None or spe.study.empty:
                spe = spectrum_compute_gauges(spe)

            row = pd.concat(
                [pd.DataFrame({"n°Spec": [int(num_spec)]}), spe.study],
                axis=1,
            ).iloc[0]

            if d.summary is None or d.summary.empty:
                d.summary = pd.DataFrame([row])
            else:
                if "n°Spec" in d.summary.columns:
                    idx = np.where(d.summary["n°Spec"] == num_spec)[0]
                else:
                    idx = []

                if len(idx) == 0:
                    d.summary = pd.concat(
                        [d.summary, pd.DataFrame([row])],
                        ignore_index=True,
                    )
                else:
                    d.summary.iloc[idx[0]] = row

    # ------------------------------------------------------------------
    # Oscilloscope / temps
    # ------------------------------------------------------------------
    def extract_time(self, time_index: Optional[List[int]] = None, data_time: Optional[str] = None):
        d = self.data

        if data_time is not None:
            d.data_oscillo = pd.read_csv(
                data_time,
                sep=r"\s+",
                skipfooter=0,
                engine="python",
            )
        if time_index:
            d.time_index = time_index

        if d.data_oscillo is not None:
            c_nam = d.data_oscillo.columns.tolist()
            temps = np.array(d.data_oscillo[c_nam[0]])
            signale_spec = np.array(d.data_oscillo[c_nam[d.time_index[0]]])
            signale_cam = np.array(d.data_oscillo[c_nam[d.time_index[1]]])

            marche_cam = max(signale_cam) / 2
            if marche_cam < 0.5:
                d.t0_movie = 0
            else:
                d.t0_movie = temps[np.where(signale_cam > marche_cam)[0][0]]
            print(f"t0_movie = {d.t0_movie * 1e3} ms")

            marche_spec = max(signale_spec) / 2
            signale_spec_bit = (signale_spec >= marche_spec).astype(int)
            front = np.diff(signale_spec_bit)
            d.time_spectrum = (
                temps[np.where(front >= 0.9)] + temps[np.where(front <= -0.9)]
            ) / 2
            print(f"Time_spectrum len = {len(d.time_spectrum)}")
        else:
            print("extract_time : pas de data_oscillo")

    # ------------------------------------------------------------------
    # Movie
    # ------------------------------------------------------------------
    def _load_movie(self):
        d = self.data
        d.time_movie = []

        if os.path.isdir(d.folder_movie):
            d.list_movie = [
                os.path.join(d.folder_movie, name)
                for name in os.listdir(d.folder_movie)
            ]
            d.movie = []
            for i, path in enumerate(d.list_movie):
                if os.path.isfile(path):
                    try:
                        img = Image.open(path)
                        d.movie.append(img)
                        if d.fps is not None:
                            d.time_movie.append(i / d.fps + d.t0_movie)
                    except IOError:
                        pass
            print("Movie (image sequence) DONE")

        elif os.path.isfile(d.folder_movie):
            try:
                cap = cv2.VideoCapture(d.folder_movie)
                d.fps = cap.get(cv2.CAP_PROP_FPS)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                d.time_movie = [i / d.fps + d.t0_movie for i in range(num_frames)]
                print("Movie (video) DONE")
            except IOError:
                print("Movie load FAILED")

    # ------------------------------------------------------------------
    # Print simple d'un spectre
    # ------------------------------------------------------------------
    def print_spectrum(self, num_spec: int = 0):
        d = self.data

        if num_spec < 0 or num_spec >= len(d.spectra):
            print("print_spectrum: num_spec hors limites.")
            return

        spe = d.spectra[num_spec]

        x = spe.x_corr if spe.x_corr is not None else spe.x_raw
        y_raw = spe.y_raw
        y_corr = spe.y_corr if spe.y_corr is not None else y_raw
        bkg = spe.baseline if spe.baseline is not None else np.zeros_like(x)

        fig, ax = plt.subplots()
        ax.plot(x, y_raw, "-", color="gray", label="Brut")
        ax.plot(x, bkg, "--", color="green", label="Baseline")
        ax.plot(x, y_corr + bkg, "-k", label="Corr + bkg")
        ax.set_xlabel(r"$\lambda$ (nm)")
        ax.set_ylabel("U.A.")
        ax.legend(loc="best")
        ax.set_title(f"Spectrum #{num_spec}")
