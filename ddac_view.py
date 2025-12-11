from __future__ import annotations

import copy
import os
from typing import Optional

import cv2
import matplotlib.colors as mcolors
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from Bibli_python import CL_FD_Update as CL


class DdacViewMixin:
    def _setup_ddac_box(self):
        group_graphique = QGroupBox("dDAC")
        layout_graphique = QVBoxLayout()

        movie_layout = QHBoxLayout()
        # Slider Qt pour index d'image
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)  # sera mis à jour après chargement
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self._on_slider_movie_changed)
        movie_layout.addWidget(self.slider)

        # Boutons précédent / play / suivant
        self.previous_button = QPushButton("⟵")
        self.previous_button.clicked.connect(self.previous_image)
        movie_layout.addWidget(self.previous_button)

        self.play_stop_button = QPushButton("play/stop")
        self.play_stop_button.clicked.connect(self.f_play_stop_movie)
        movie_layout.addWidget(self.play_stop_button)

        self.next_button = QPushButton("⟶")
        self.next_button.clicked.connect(self.next_image)
        movie_layout.addWidget(self.next_button)

        layout_graphique.addLayout(movie_layout)
        # FPS

        self.fps_play_spinbox = QSpinBox()
        self.fps_play_spinbox.setRange(1, 1000)
        self.fps_play_spinbox.setValue(100)
        movie_layout.addWidget(QLabel("fps:"))
        movie_layout.addWidget(self.fps_play_spinbox)

        # ================== WIDGET PyQtGraph ==================
        self.pg_ddac = pg.GraphicsLayoutWidget()

        # Col 0 : P, dP/dt/T, sigma
        self.pg_P = self.pg_ddac.addPlot(row=0, col=0)
        self.pg_P.setLabel('bottom', 'Time (s)')
        self.pg_P.setLabel('left', 'P (GPa)')

        self.pg_dPdt = self.pg_ddac.addPlot(row=1, col=0)
        self.pg_dPdt.setLabel('bottom', 'Time (s)')
        self.pg_dPdt.setLabel('left', 'dP/dt (GPa/ms), T (K)')

        self.pg_sigma = self.pg_ddac.addPlot(row=2, col=0)
        self.pg_sigma.setLabel('bottom', 'Time (s)')
        self.pg_sigma.setLabel('left', 'sigma (nm)')

        # Col 1 : IMAGE + Δλ
        self.pg_movie = self.pg_ddac.addPlot(row=0, col=1)
        self.pg_movie.setAspectLocked(True)
        self.pg_movie.hideAxis('bottom')
        self.pg_movie.hideAxis('left')
        self.img_item = pg.ImageItem()
        self.pg_movie.addItem(self.img_item)

        self.pg_text = self.pg_ddac.addViewBox(row=1, col=1)
        self.pg_text.setAspectLocked(False)
        self.pg_text.enableAutoRange(False)

        self.pg_text_label = pg.TextItem(color='w')
        font = QFont("Arial", 14)   # plus lisible
        self.pg_text_label.setFont(font)

        self.pg_text.addItem(self.pg_text_label)

        # Positionnement centré
        self.pg_text_label.setAnchor((0.5, 0.5))
        self.pg_text_label.setPos(0, 0)
        self.pg_text.setRange(xRange=(-1, 1), yRange=(-1, 1))


        self.pg_dlambda = self.pg_ddac.addPlot(row=2, col=1)
        self.pg_dlambda.setLabel('bottom', 'Spectrum index')
        self.pg_dlambda.setLabel('left', 'Δλ12 (nm)')

        # ================== COURBES PERSISTANTES ==================
        self.curves_P = []        # une courbe par jauge
        self.curves_dPdt = []
        self.curves_sigma = []
        self.curves_T = []
        self.curve_piezo_list = []
        self.curve_corr_list = []
        self.curves_dlambda = []

        # Ligne verticale pour t sélectionné
        self.line_t_P = pg.InfiniteLine(angle=90, movable=False)
        self.line_t_dPdt = pg.InfiniteLine(angle=90, movable=False)
        self.line_t_sigma = pg.InfiniteLine(angle=90, movable=False)
        self.line_p0 = pg.InfiniteLine(0,angle=0, movable=False)
        self.pg_P.addItem(self.line_p0)
        self.pg_P.addItem(self.line_t_P)
        self.pg_dPdt.addItem(self.line_t_dPdt)
        self.pg_sigma.addItem(self.line_t_sigma)

        self.line_nspec = pg.InfiniteLine(angle=90, movable=False)
        self.pg_dlambda.addItem(self.line_nspec)

        # Zone temporelle film (bornes)
        self.zone_movie = [None, None]
        self.zone_movie_lines = [
            pg.InfiniteLine(angle=90, movable=False),
            pg.InfiniteLine(angle=90, movable=False)
        ]
        for line in self.zone_movie_lines:
            self.pg_P.addItem(line)


                # ================== MARQUEURS DE CLIC (SCATTER CROIX) ==================
        # Un scatter par graphe pour montrer la position exacte du clic

        self.scatter_P = pg.ScatterPlotItem(
            x=[], y=[],
            pen=None,
            brush=None,          # pas de remplissage
            size=10,
            symbol='+'           # croix
        )
        self.pg_P.addItem(self.scatter_P)

        self.scatter_dPdt = pg.ScatterPlotItem(
            x=[], y=[],
            pen=None,
            brush=None,
            size=10,
            symbol='+'
        )
        self.pg_dPdt.addItem(self.scatter_dPdt)

        self.scatter_sigma = pg.ScatterPlotItem(
            x=[], y=[],
            pen=None,
            brush=None,
            size=10,
            symbol='+'
        )
        self.pg_sigma.addItem(self.scatter_sigma)

        self.scatter_dlambda = pg.ScatterPlotItem(
            x=[], y=[],
            pen=None,
            brush=None,
            size=10,
            symbol='+'
        )
        self.pg_dlambda.addItem(self.scatter_dlambda)


        # ================== ÉTAT DYNAMIQUE (film) ==================
        self.current_index = 0
        self.index_playing = 0
        self.playing_movie = False
        self.t_cam = []       # liste par CEDd
        self.index_cam = []
        self.cap = []         # VideoCapture si tu gardes OpenCV
        self.correlations = []
        self.time = []
        self.spectre_number = []

        # ================== INTÉGRATION WIDGET GRAPHIQUE ==================
        layout_graphique.addWidget(self.pg_ddac)

        # ================== CONTROLES Qt (slider + boutons) ==================
        controls_layout = QHBoxLayout()

        self.label_CED = QLabel('CEDd file select:', self)
        self.label_CED.setFont(QFont("Arial", 8))
        controls_layout.addWidget(self.label_CED)

        self.movie_select_box = QCheckBox("clic frame (m)", self)
        self.movie_select_box.setChecked(True)
        controls_layout.addWidget(self.movie_select_box)

        self.spectrum_select_box = QCheckBox("clic spectrum (h)", self)
        self.spectrum_select_box.setChecked(True)
        controls_layout.addWidget(self.spectrum_select_box)

        layout_graphique.addLayout(controls_layout)

        group_graphique.setLayout(layout_graphique)

        self.grid_layout.addWidget(group_graphique, 0, 3, 3, 1)

        # Timer Qt pour le mode "lecture"
        self.timerMovie = QTimer(self)
        self.timerMovie.timeout.connect(self.play_movie)

        self.pg_P.scene().sigMouseClicked.connect(self._on_ddac_click)
        self.pg_dPdt.scene().sigMouseClicked.connect(self._on_ddac_click)
        self.pg_sigma.scene().sigMouseClicked.connect(self._on_ddac_click)
        self.pg_dlambda.scene().sigMouseClicked.connect(self._on_ddac_click)

        self._spectrum_limits_initialized = False
        self._zoom_limits_initialized = False

        # ================== FACTEURS DE LIGNES (2, 2, 1) ==================
        self._ddac_row_factors = (4, 1, 1)
        # exemple : colonne temps / colonne image de même largeur
        self._ddac_col_factors = (1, 1)

        self._update_graphicslayout_sizes(
            self.pg_ddac,
            row_factors=self._ddac_row_factors,
            col_factors=self._ddac_col_factors,
        )

    def _get_state_for_run(self, run_id: Optional[str] = None) -> Optional[RunViewState]:
        if run_id is None:
            run_id = self.current_run_id
        return self.runs.get(run_id)

    def _save_current_run(self):
        """Sauvegarde le RUN courant dans son état RunViewState."""

        state = self._get_state_for_run()
        if state is not None and self.RUN is not None:
            state.ced = copy.deepcopy(self.RUN)

    def _finalize_run_selection(self, state: RunViewState, name_select: str):
        """Replace l'ancienne logique basée sur index_select par l'état RunViewState."""
        self.current_run_id = self._get_run_id(state.ced)
        self.RUN = copy.deepcopy(state.ced)
        self.index_select = self.liste_objets_widget.row(state.list_item) if state.list_item is not None else -1

        if hasattr(self.RUN,"fps") and self.RUN.fps is not None:
            try:
                text_fps="Movie :1e"+str(round(np.log10(self.RUN.fps),2))+"fps"
            except Exception as e:
                print("fps log ERROR:",e)
                text_fps="Movie :"+str(round(self.RUN.fps,2))+"fps"
        else:
            text_fps="No Movie"

        if state.index_cam is not None and len(state.index_cam)>0:
            self.current_index = len(state.index_cam) // 2
            self.slider.setMaximum(max(0, len(state.index_cam) - 1))
            self.slider.setValue(self.current_index)

        if state.time is not None and len(state.time)>0:
            x_min, x_max = min(state.time), max(state.time)
            self.pg_P.setXRange(x_min, x_max, padding=0.01)
            self.pg_dPdt.setXRange(x_min, x_max, padding=0.01)
            self.pg_sigma.setXRange(x_min, x_max, padding=0.01)
            self.pg_dlambda.setXRange(x_min, x_max, padding=0.01)

        self._update_movie_frame()

        self.label_CED.setText( f"CEDd {name_select} fps :{text_fps}")

        self.CLEAR_ALL()
        # Mise à jour de la spinbox de spectre
        n_spec = len(self.RUN.list_nspec) if hasattr(self.RUN, "list_nspec") else len(self.RUN.Spectra)
        self.spinbox_spec_index.blockSignals(True)
        self.spinbox_spec_index.setRange(0, max(0, n_spec - 1))
        self.spinbox_spec_index.setValue(0)
        self.spinbox_spec_index.blockSignals(False)
        self.index_spec = 0
        self.bit_bypass = True
        self.LOAD_Spectrum()
        self.bit_bypass = False
        self.Update_Print()

    def _ensure_curve_at(self, curve_list, index, plot_widget, **plot_kwargs):
        """Garantit la présence d'une courbe à l'index donné et la retourne."""

        while len(curve_list) <= index:
            curve_list.append(None)

        curve = curve_list[index]
        if curve is None:
            curve = plot_widget.plot(**plot_kwargs)
            curve_list[index] = curve
        return curve

    def _get_or_create_curve(self, curve, plot_widget, **plot_kwargs):
        """Retourne la courbe existante ou en crée une nouvelle sur le plot fourni."""

        if curve is None:
            curve = plot_widget.plot(**plot_kwargs)
        return curve

    def _collect_curve_arrays(self, curves):
        """Récupère les tableaux Y valides des courbes PyQtGraph fournies."""

        arrays = []
        for c in curves:
            if c is None:
                continue
            _, y = c.getData()
            if y is not None and len(y) > 0:
                arrays.append(np.asarray(y, dtype=float))
        return arrays

    def _apply_viewbox_limits(self, viewbox, x_data, curves, extra_curves=None):
        """Applique des limites à partir des courbes fournies sur une ViewBox."""

        y_arrays = self._collect_curve_arrays(curves)

        if extra_curves:
            if not isinstance(extra_curves, (list, tuple)):
                extra_curves = [extra_curves]
            y_arrays.extend(self._collect_curve_arrays(extra_curves))

        if y_arrays:
            y_concat = np.concatenate(y_arrays)
            self._set_viewbox_limits_from_data(viewbox, x_data, y_concat, padding=0.02)

    def _refresh_ddac_limits(self, state: RunViewState) -> None:
        """Ajuste les limites des graphes PyQtGraph liés aux données dDAC.

        Les ViewBox P, dP/dt, σ et Δλ sont calées sur l'axe temps commun issu de
        l'état `RunViewState`. Les valeurs Y sont évaluées depuis les courbes
        déjà tracées, sans recalcul métier, afin de centrer les zooms et limiter
        les déplacements hors données.
        """
        # ----- Axe temps commun -----
        Time = getattr(state, "time", None)
        if Time is None or len(Time) < 2:
            return

        x_time = np.asarray(Time, dtype=float)

        vb_P = self.pg_P.getViewBox()
        self._apply_viewbox_limits(vb_P, x_time, state.curves_P, getattr(state, "piezo_curve", None))

        vb_dPdt = self.pg_dPdt.getViewBox()
        self._apply_viewbox_limits(vb_dPdt, x_time, state.curves_dPdt + state.curves_T)

        vb_sigma = self.pg_sigma.getViewBox()
        self._apply_viewbox_limits(vb_sigma, x_time, state.curves_sigma)

        vb_dlambda = self.pg_dlambda.getViewBox()
        self._apply_viewbox_limits(vb_dlambda, x_time, state.curves_dlambda)

    def f_zone_movie(self):
        """Sélection / reset d'une zone temporelle de film (2 bornes)."""
        state = self._get_state_for_run()
        if state is None or not state.t_cam:
            return

        t_current = state.t_cam[self.current_index]

        if self.zone_movie[0] is None:
            self.zone_movie[0] = t_current
            self.zone_movie_lines[0].setPos(t_current)
            return
        elif self.zone_movie[1] is None:
            self.zone_movie[1] = t_current
            self.zone_movie_lines[1].setPos(t_current)
            return
        else:
            # Reset
            self.zone_movie = [None, None]
            t_min = state.t_cam[0]
            t_max = state.t_cam[-1]
            self.zone_movie_lines[0].setPos(t_min)
            self.zone_movie_lines[1].setPos(t_max)

    def update(self,val): # pour la barre de défilmetn des images
        self.current_index=int(val)
        state = self._get_state_for_run()
        if state and state.index_cam:
            self.Num_im=state.index_cam[self.current_index]
        self._update_movie_frame()

    def read_frame(self,cap,frame_number,unit="rgb"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            if unit=="rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif unit=="gray":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        else:
            return None

    def REFRESH(self):
        self.RUN.Spectra[self.index_spec]=self.Spectrum
        self.RUN.Corr_Summary(All=True)
        state = self._get_state_for_run()
        if state is None:
            self.text_box_msg.setText("Aucun état RunViewState pour rafraîchir ce CEDd")
            return

        state.ced = copy.deepcopy(self.RUN)
        self._update_curves_for_run(state)

    def _update_curves_for_run(self, state: RunViewState):
        """Met à jour les courbes PyQtGraph à partir des données d'un `RunViewState`.

        Les listes temporelles et spectrales calculées par `Read_RUN` sont
        réaffectées sur les courbes déjà créées (P, dP/dt, T, Δλ, piézo,
        corrélation). Les attributs `state.time`, `state.spectre_number` et les
        tracés sont actualisés sans recréer d'objets graphiques.
        """

        (
            l_P,
            l_sigma_P,
            l_lambda,
            l_fwhm,
            l_spe,
            l_T,
            l_sigma_T,
            Time,
            spectre_number,
            time_amp,
            amp,
            Gauges_RUN,
        ) = self.Read_RUN(self.RUN)

        state.time = Time
        state.spectre_number = spectre_number

        for i, G in enumerate(Gauges_RUN):
            l_p_filtre = CL.savgol_filter(l_P[i], 10, 1) if len(l_P[i]) > 0 else np.array([])
            if len(l_p_filtre) > 4 and len(Time) > 4:
                dps = [
                    (l_p_filtre[x + 1] - l_p_filtre[x - 1]) / (Time[x + 1] - Time[x - 1]) * 1e-3
                    for x in range(2, len(l_p_filtre) - 2)
                ]
                time_dps = Time[2:-2]
            else:
                dps = []
                time_dps = []

            curve_kwargs = dict(
                pen=pg.mkPen(G.color_print[0], width=1),
                symbol='d',
                symbolPen=None,
                symbolBrush=G.color_print[0],
                symbolSize=4,
            )

            curve_P = self._ensure_curve_at(state.curves_P, i, self.pg_P, **curve_kwargs)
            curve_P.setData(Time, l_P[i])

            curve_dPdt = self._ensure_curve_at(state.curves_dPdt, i, self.pg_dPdt, **curve_kwargs)
            curve_dPdt.setData(time_dps, dps)

            curve_sigma = self._ensure_curve_at(state.curves_sigma, i, self.pg_sigma, **curve_kwargs)
            curve_sigma.setData(Time, l_fwhm[i])

        has_T = "RuSmT" in [x.name_spe for x in self.RUN.Gauges_init]
        if has_T:
            curve_T = self._ensure_curve_at(
                state.curves_T,
                0,
                self.pg_dPdt,
                pen=pg.mkPen('darkred'),
                symbol='t',
                symbolBrush='darkred',
                symbolSize=6,
            )
            curve_T.setData(Time, l_T[-1] if l_T else [])
        elif state.curves_T:
            state.curves_T[0].setData([], [])

        for i, spe in enumerate(l_spe):
            curve = self._ensure_curve_at(
                state.curves_dlambda,
                i,
                self.pg_dlambda,
                pen=None,
                symbol='+',
                symbolPen=pg.mkPen(state.color),
            )
            curve.setData(Time, spe)
        for extra_index in range(len(l_spe), len(state.curves_dlambda)):
            state.curves_dlambda[extra_index].setData([], [])

        if self.RUN.data_Oscillo is not None:
            state.piezo_curve = self._get_or_create_curve(state.piezo_curve, self.pg_P, pen=pg.mkPen(state.color))
            state.piezo_curve.setData(time_amp, amp)
        elif state.piezo_curve is not None:
            state.piezo_curve.setData([], [])

        if state.corr_curve is not None:
            if self.var_bouton[3].isChecked() and state.correlations:
                state.corr_curve.setData(state.t_cam, state.correlations)
            else:
                state.corr_curve.setData([], [])

        self._refresh_ddac_limits(state)

    def PRINT_CEDd(self, item=None, objet_run=None):
        """
        - Si item est donné (clic dans self.liste_fichiers) et objet_run est None :
            -> on charge depuis un fichier si c'est la première fois
            -> sinon on récupère l'objet déjà en mémoire (file_index_map)
        - Si objet_run est donné (création d'un nouveau CEDd) :
            -> on l'ajoute à la liste interne, sans passer par la liste de fichiers
        """
        # Sécu : s'assurer que la map existe
        if not hasattr(self, "file_index_map"):
            self.file_index_map = {}

        self._save_current_run()

        run_id = None
        state = None
        name_select = None

        # ------------------------------------------------------------------
        # CAS 1 : on vient d'un clic dans la liste de fichiers (item non None)
        # ------------------------------------------------------------------
        if objet_run is None and item is not None:
            chemin_fichier = os.path.join(self.dossier_selectionne, item.text())
            index_file = self.liste_fichiers.row(item)

            if index_file not in self.file_index_map:
                try:
                    objet_run = CL.LOAD_CEDd(chemin_fichier)
                except Exception as e:
                    print("Erreur LOAD_CEDd:", e)
                    self.text_box_msg.setText(f"Erreur chargement CEDd : {e}")
                    return
                run_id = self._get_run_id(objet_run)
                self.file_index_map[index_file] = run_id
                name_select = item.text()
            else:
                run_id = self.file_index_map[index_file]
                state = self.runs.get(run_id)
                name_select = item.text()

        # ------------------------------------------------------------------
        # CAS 2 : on reçoit directement un CEDd (objet_run != None)
        # ------------------------------------------------------------------
        elif objet_run is not None:
            run_id = self._get_run_id(objet_run)
            state = self.runs.get(run_id)
            name_select = CL.os.path.basename(objet_run.CEDd_path) if hasattr(objet_run, "CEDd_path") else "CEDd_new"
        else:
            return

        # Si un état existe déjà : on réutilise les courbes et on restaure l'UI
        if state is not None:
            self._finalize_run_selection(state, name_select)
            return

        # ------------------------------------------------------------------
        # Création d'un nouvel état RunViewState
        # ------------------------------------------------------------------
        if objet_run is None:
            return

        self.liste_objets.append(objet_run)
        self.index_select = len(self.liste_objets) - 1
        self.RUN = objet_run

        # Couleur dédiée à ce run
        c = self.c_m[0] if self.c_m else "#ffffff"
        self.color.append(c)
        if self.c_m:
            del self.c_m[0]

        name_select = name_select or CL.os.path.basename(self.RUN.CEDd_path)
        item_run = QListWidgetItem(name_select)
        item_run.setBackground(QColor(c))
        dark_c = mcolors.rgb2hex(
            tuple(min(1, cc * 0.5) for cc in mcolors.hex2color(c))
        )
        item_run.setForeground(QColor(dark_c))
        item_run.setData(Qt.UserRole, run_id)
        self.liste_objets_widget.addItem(item_run)

        # Lecture des données CEDd
        l_P, l_sigma_P, l_lambda, l_fwhm, l_spe, l_T, l_sigma_T, Time, spectre_number, time_amp, amp, Gauges_RUN = self.Read_RUN(self.RUN)

        # Axes temps
        self.pg_P.setXRange(min(Time), max(Time), padding=0.01)
        self.pg_dPdt.setXRange(min(Time), max(Time), padding=0.01)
        self.pg_sigma.setXRange(min(Time), max(Time), padding=0.01)
        self.pg_dlambda.setXRange(min(Time), max(Time), padding=0.01)

        state = RunViewState(ced=self.RUN, color=c)
        state.time = Time
        state.spectre_number = spectre_number
        state.curves_P = []
        state.curves_dPdt = []
        state.curves_sigma = []
        state.curves_T = []
        state.curves_dlambda = []
        state.piezo_curve = None
        state.corr_curve = None
        state.index_cam = []
        state.t_cam = []
        state.correlations = []
        state.list_item = item_run

        self.runs[run_id] = state
        self.current_run_id = run_id

        if self.RUN.data_movie is not None:
            try:
                t_cam, index_cam, _ = CL.Correl_Image_amp(
                    self.RUN.data_movie["t"],
                    self.RUN.data_movie["signal"],
                    time_amp,
                    amp,
                )
                state.t_cam = t_cam
                state.index_cam = [int(x) for x in index_cam]
                state.cap = self.RUN.Movie

                self.cap = self.RUN.Movie
                self.index_cam = state.index_cam
                self.t_cam = state.t_cam

                self.line_t_P.setPos(t_cam[len(t_cam)//2])
                self.line_t_dPdt.setPos(t_cam[len(t_cam)//2])
                self.line_t_sigma.setPos(t_cam[len(t_cam)//2])
                self.zone_movie_lines[0].setPos(t_cam[0])
                self.zone_movie_lines[1].setPos(t_cam[-1])

                self.correlations = _
                state.correlations = _

                state.corr_curve = self.pg_sigma.plot(pen=pg.mkPen(state.color))

                self.line_nspec.setPos(spectre_number[len(t_cam)//2])

                self.current_index = len(state.index_cam)//2 if state.index_cam else 0

                self.slider.setMaximum(max(0, len(state.index_cam) - 1))
                self.slider.setValue(self.current_index)

                state.cap.set(cv2.CAP_PROP_POS_FRAMES, state.index_cam[self.current_index])
                ret, frame = state.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.img_item.setImage(frame, autoLevels=True)

                # Première position des lignes de zone_movie aux bornes min/max temps
                if state.t_cam and len(state.t_cam) >= 2:
                    self.zone_movie[0] = state.t_cam[0]
                    self.zone_movie[1] = state.t_cam[-1]
                    self.zone_movie_lines[0].setPos(self.zone_movie[0])
                    self.zone_movie_lines[1].setPos(self.zone_movie[1])
            except Exception as e:
                print("Erreur correlation :", e)

        # Création des courbes PyQtGraph
        for i, G in enumerate(Gauges_RUN):
            l_p_filtre = CL.savgol_filter(l_P[i], 10, 1) if len(l_P[i]) > 0 else np.array([])
            if len(l_p_filtre) > 4 and len(Time) > 4:
                dps = [
                    (l_p_filtre[x + 1] - l_p_filtre[x - 1]) / (Time[x + 1] - Time[x - 1]) * 1e-3
                    for x in range(2, len(l_p_filtre) - 2)
                ]
                time_dps = Time[2:-2]
            else:
                dps = []
                time_dps = []

            curve_kwargs = dict(
                pen=pg.mkPen(G.color_print[0], width=1),
                symbol='d',
                symbolPen=None,
                symbolBrush=G.color_print[0],
                symbolSize=4,
            )

            cP = self.pg_P.plot(Time, l_P[i], **curve_kwargs)
            cSigma = self.pg_sigma.plot(Time, l_fwhm[i], **curve_kwargs)
            cDps = self.pg_dPdt.plot(time_dps, dps, **curve_kwargs)

            state.curves_P.append(cP)
            state.curves_sigma.append(cSigma)
            state.curves_dPdt.append(cDps)

        has_T = "RuSmT" in [x.name_spe for x in self.RUN.Gauges_init]
        if has_T:
            # Tracé de T
            cT = self.pg_dPdt.plot(
                Time,
                l_T[-1],
                pen=pg.mkPen('darkred'),
                symbol='t',
                symbolBrush='darkred',
                symbolSize=6,
            )
            state.curves_T.append(cT)

        if self.RUN.data_Oscillo is not None:
            cPiezo = self.pg_P.plot(time_amp, amp, pen=pg.mkPen(state.color))
            state.piezo_curve = cPiezo

        # Δλ
        for spe in l_spe:
            cdLambda = self.pg_dlambda.plot(
                Time,
                spe,
                pen=None,
                symbol='+',
                symbolPen=pg.mkPen(state.color),
            )
            state.curves_dlambda.append(cdLambda)

        self._refresh_ddac_limits(state)

        # Ajout d'une ligne horizontale pour le sigma (position y = 0)
        # (la ligne est déjà ajoutée à pg_P, on peut en ajouter une autre pour pg_sigma
        # si tu veux, mais ici on conserve le comportement actuel)

        # Mémorise l'item dans l'état
        state.list_item = item_run

        # Sélectionne l'item dans la liste
        item_run.setSelected(True)

        # Sélection initiale
        self._finalize_run_selection(state, name_select)

    def f_text_CEDd_print(self, t):
        """Met à jour le texte d'information dDAC dans le TextItem PyQtGraph."""
        dp = None
        if getattr(self, "tstart", None) is not None and getattr(self, "tend", None) is not None and self.tstart != self.tend:
            dp = (self.Pstart - self.Pend) / (self.tstart - self.tend) * 1e-3

        txt = (
            f"$t_spec$={self.x1*1e3:.3f}ms n°Spec={self.x5:.2f}\n\n"
            f"$t_frame$={t*1e6:.3f}µs n°Frame={self.Num_im}\n\n"
            f"P={self.y1:.2f}GPa  T or dP/dt={self.y3:.2f} K or GPa/ms\n\n"
            f"dP/dt={0.0 if dp is None else dp:.3f} GPa/ms"
            )
        self.pg_text_label.setText(txt)

    def f_CEDd_update_print(self):
        """Met à jour texte + sélection de spectre en fonction de x_clic / t."""
        state = self._get_state_for_run()
        if self.RUN is None or state is None:
            return

        # temps courant pour le film (si t_cam dispo)
        if self.RUN.Movie is not None and state.t_cam:
            t = state.t_cam[self.current_index]
        else:
            t = 0.0

        # spectre le plus proche en temps
        times = np.array(state.time)
        idx_spec = int(np.argmin(np.abs(times - self.x_clic)))
        spec_nb = state.spectre_number[idx_spec]

        if self.spectrum_select_box.isChecked() and int(spec_nb) != self.index_spec:
            self.index_spec = int(spec_nb)
            self.spinbox_spec_index.blockSignals(True)
            self.spinbox_spec_index.setValue(self.index_spec)
            self.spinbox_spec_index.blockSignals(False)
            self.bit_bypass = True
            try:
                self.LOAD_Spectrum()
            finally:
                self.bit_bypass = False

        self.f_text_CEDd_print(t)

    def _on_ddac_click(self, mouse_event):
        """Gère les clics sur les graphes temporels dDAC et synchronise l'état UI.

        Le clic est transformé des coordonnées scène vers le ViewBox ciblé (P,
        dP/dt, σ ou Δλ) pour extraire (t, valeur). Les lignes verticales, les
        marqueurs scatter et les textes sont mis à jour, puis la sélection de
        spectre/film est recalculée en fonction du temps cliqué.
        """
        self.setFocus()
        pos = mouse_event.scenePos()
        vb = None
        which = None

        # Sur quel plot on a cliqué ?
        if self.pg_P.sceneBoundingRect().contains(pos):
            vb = self.pg_P.getViewBox()
            which = "P"
        elif self.pg_dPdt.sceneBoundingRect().contains(pos):
            vb = self.pg_dPdt.getViewBox()
            which = "dPdt"
        elif self.pg_sigma.sceneBoundingRect().contains(pos):
            vb = self.pg_sigma.getViewBox()
            which = "sigma"
        elif self.pg_dlambda.sceneBoundingRect().contains(pos):
            vb = self.pg_dlambda.getViewBox()
            which = "dlambda"
        else:
            return

        mouse_point = vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        # Mise à jour des x1/x3/x5/x6 + lignes + scatters
        if which == "P":
            self.x1, self.y1 = x, y
            self.x_clic, self.y_clic = x, y
            self.line_t_P.setPos(x)
            self.line_t_sigma.setPos(x)
            self.line_t_dPdt.setPos(x)

            # scatter sur le graphe P
            if hasattr(self, "scatter_P"):
                self.scatter_P.setData([x], [y])

            # gestion du dP (start/end)
            if self.bit_dP == 0:
                self.Pstart, self.tstart, self.bit_dP = self.y1, self.x1, 1
            elif self.bit_dP == 1:
                self.Pend, self.tend, self.bit_dP = self.y1, self.x1, 0

        elif which == "dPdt":
            self.x3, self.y3 = x, y
            self.x_clic, self.y_clic = x, y
            self.line_t_P.setPos(x)
            self.line_t_sigma.setPos(x)
            self.line_t_dPdt.setPos(x)

            # scatter sur le graphe dP/dt
            if hasattr(self, "scatter_dPdt"):
                self.scatter_dPdt.setData([x], [y])

        elif which == "sigma":
            self.x5, self.y5 = x, y
            self.x_clic, self.y_clic = x, y
            self.line_t_P.setPos(x)
            self.line_t_sigma.setPos(x)
            self.line_t_dPdt.setPos(x)

            # scatter sur le graphe sigma
            if hasattr(self, "scatter_sigma"):
                self.scatter_sigma.setData([x], [y])

        elif which == "dlambda":
            self.x6, self.y6 = x, y
            self.x_clic, self.y_clic = x, y
            self.line_t_P.setPos(x)
            self.line_t_sigma.setPos(x)
            self.line_t_dPdt.setPos(x)
            self.line_nspec.setPos(x)

            # scatter sur le graphe Δλ
            if hasattr(self, "scatter_dlambda"):
                self.scatter_dlambda.setData([x], [y])

        # Mise à jour frame film si la case est cochée
        state = self._get_state_for_run()
        if (
            self.movie_select_box.isChecked()
            and self.RUN is not None
            and state is not None
            and state.t_cam
        ):

            t_array = np.array(state.t_cam)
            self.current_index = int(np.argmin(np.abs(t_array - self.x_clic)))
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_index)
            self.slider.blockSignals(False)
            self._update_movie_frame()

        self.f_CEDd_update_print()

    def _on_slider_movie_changed(self, idx: int):
        """Callback Qt du slider vidéo : change l'index courant et rafraîchit l'image."""
        self.current_index = idx
        self._update_movie_frame()

    def _update_movie_frame(self):
        """Affiche l'image courante du film et synchronise les repères temporels.

        Le frame indexé par `self.current_index` est lu via `read_frame`, puis
        affiché dans `img_item`. Les lignes verticales des graphes temporels sont
        alignées sur le temps de la frame et, si disponible, la position spectrale
        associée est rappelée via `line_nspec` et le texte d'info.
        """
        state = self._get_state_for_run()
        if state is None or not state.index_cam:
            return

        idx_list = state.index_cam
        if self.current_index < 0 or self.current_index >= len(idx_list):
            return

        frame_idx = idx_list[self.current_index]
        t = state.t_cam[self.current_index]

        frame = self.read_frame(state.cap, frame_idx)
        if frame is None:
            return

        # frame : np.array 2D ou 3D
        self.img_item.setImage(frame, autoLevels=True)  # ou sans .T suivant orientation

        # Lignes verticales
        self.line_t_P.setPos(t)
        self.line_t_dPdt.setPos(t)
        self.line_t_sigma.setPos(t)
        if state.spectre_number and len(state.spectre_number) > self.current_index:
            nspec = state.spectre_number[self.current_index]
            self.line_nspec.setPos(nspec)

        # Texte
        txt = (
            f"t = {t*1e3:.3f} ms\n"
            f"frame = {frame_idx}"
        )
        self.pg_text_label.setText(txt)

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_index)
            self.slider.blockSignals(False)
            self._update_movie_frame()

    def next_image(self):
        state = self._get_state_for_run()
        if state is None:
            return
        nb = len(state.index_cam)
        if self.current_index < nb - 1:
            self.current_index += 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_index)
            self.slider.blockSignals(False)
            self._update_movie_frame()

    def f_play_stop_movie(self):
        self.playing_movie = not self.playing_movie
        if self.playing_movie:
            fps = self.fps_play_spinbox.value()
            if fps <= 0:
                fps = 100
            interval_ms = int(1000 / fps)
            self.timerMovie.start(interval_ms)
        else:
            self.timerMovie.stop()

    def play_movie(self):
        state = self._get_state_for_run()
        if state is None:
            self.timerMovie.stop()
            self.playing_movie = False
            return

        nb = len(state.index_cam)
        if nb == 0:
            self.timerMovie.stop()
            self.playing_movie = False
            return

        # Déterminer les bornes de lecture en fonction du cadrage / zone_movie
        i_min, i_max = self._get_movie_bounds(state)
        if i_max < i_min:  # cas dégénéré
            self.timerMovie.stop()
            self.playing_movie = False
            return

        # Si l'index courant est en dehors de la zone, on se remet sur le début de la zone
        if self.current_index < i_min or self.current_index > i_max:
            self.current_index = i_min
        else:
            # Avancer d'une image dans la zone
            self.current_index += 1
            if self.current_index > i_max:
                # Boucle dans la zone définie
                self.current_index = i_min

        # Mise à jour du slider et de l'image
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)
        self._update_movie_frame()

    def _get_movie_bounds(self, state):
        """
        Retourne (i_min, i_max) : indices de lecture pour le film.
        - Si zone_movie n'est pas définie => toute la durée.
        - Sinon => plus petit et plus grand indices dont t_cam est dans [t0, t1].
        """
        nb = len(state.index_cam)
        if nb == 0:
            return 0, -1  # cas degénéré

        # Pas de zone définie : on lit tout
        if self.zone_movie[0] is None or self.zone_movie[1] is None:
            return 0, nb - 1

        t0, t1 = self.zone_movie
        if t0 > t1:
            t0, t1 = t1, t0

        t_cam = state.t_cam

        # Trouver i_min : premier index avec t_cam >= t0
        i_min = 0
        for i, t in enumerate(t_cam):
            if t >= t0:
                i_min = i
                break

        # Trouver i_max : dernier index avec t_cam <= t1
        i_max = nb - 1
        for i in range(nb - 1, -1, -1):
            if t_cam[i] <= t1:
                i_max = i
                break

        # Sécurité si la zone est hors des temps disponibles
        if i_min > i_max:
            i_min, i_max = 0, nb - 1

        return i_min, i_max
