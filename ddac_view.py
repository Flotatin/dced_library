from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QListWidget,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from Bibli_python import CL_FD_Update as CL
from theme_config import make_c_m



@dataclass
class RunViewState:
    """État graphique associé à un CEDd (remplace les listes parallèles)."""
    ced: CL.CEDd
    color_idx: str
    time: list = field(default_factory=list)
    spectre_number: list = field(default_factory=list)
    curves_P: list = field(default_factory=list)
    curves_P_extrap: list = field(default_factory=list)
    curves_dPdt: list = field(default_factory=list)
    curves_sigma: list = field(default_factory=list)
    curves_T: list = field(default_factory=list)
    curves_dlambda: list = field(default_factory=list)
    piezo_curve: object = None
    corr_curve: object = None
    cap: object = None
    t_cam: list = field(default_factory=list)
    index_cam: list = field(default_factory=list)
    correlations: list = field(default_factory=list)
    list_item: Optional[QListWidgetItem] = None
    pressure_series: list = field(default_factory=list)
    phase_tracks: dict = field(default_factory=dict)
    active_phase_element: str = "H2O"


class DdacViewMixin:
    def _time_unit_from_series(self, time_values):
        """Détermine une unité de temps lisible selon l'échelle du run."""
        arr = np.asarray(time_values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            span_s = 0.0
        else:
            span_s = float(np.nanmax(arr) - np.nanmin(arr))
        if span_s <= 0:
            return ("ms", 1e3)
        candidates = [("s", 1.0), ("ms", 1e3), ("µs", 1e6), ("ns", 1e9)]
        best = candidates[0]
        best_score = float("inf")
        for symbol, scale in candidates:
            span_in_unit = span_s * scale
            score = abs(np.log10(max(span_in_unit, 1e-12)) - 1.5)
            if score < best_score:
                best_score = score
                best = (symbol, scale)
        return best

    def _set_time_display_unit(self, time_values):
        symbol, scale = self._time_unit_from_series(time_values)
        self._time_unit_symbol = symbol
        self._time_unit_scale = float(scale)
        self._dpdt_unit_symbol = f"GPa/{symbol}"
        self._dpdt_scale_from_seconds = 1.0 / self._time_unit_scale
        self._update_time_axis_labels()

    def _update_time_axis_labels(self):
        time_symbol = getattr(self, "_time_unit_symbol", "s")
        dpdt_symbol = getattr(self, "_dpdt_unit_symbol", "GPa/s")
        if hasattr(self, "pg_sigma"):
            self.pg_sigma.setLabel("bottom", f"Time ({time_symbol})")
            axis = self.pg_sigma.getAxis("bottom")
            if axis is not None:
                axis.setScale(self._time_unit_scale)
        if hasattr(self, "pg_P"):
            # Label X uniquement sur le plot du dessous (sigma)
            self.pg_P.setLabel("bottom", "")
            axis = self.pg_P.getAxis("bottom")
            if axis is not None:
                axis.setScale(self._time_unit_scale)
        if hasattr(self, "pg_dPdt"):
            axis = self.pg_dPdt.getAxis("bottom")
            if axis is not None:
                axis.setScale(self._time_unit_scale)
            self.pg_dPdt.setLabel("bottom", "")
            self.pg_dPdt.setLabel("left", f"dP/dt ({dpdt_symbol}), T (K)")

    def _add_ddac_plot(self, *, row: int, col: int, x_label: Optional[str] = None, y_label: Optional[str] = None, show_grid: bool = True, **kwargs):
        plot_item = self.pg_ddac.addPlot(row=row, col=col, **kwargs)
        self._stylize_plot(plot_item, x_label=x_label, y_label=y_label, show_grid=show_grid)
        return plot_item

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
        self._set_movie_controls_enabled(False)


        self.fps_play_spinbox = QSpinBox()
        self.fps_play_spinbox.setRange(1, 1000)
        self.fps_play_spinbox.setValue(100)
        movie_layout.addWidget(QLabel("fps:"))
        movie_layout.addWidget(self.fps_play_spinbox)

        # ================== WIDGET PyQtGraph ==================
        self.pg_ddac = pg.GraphicsLayoutWidget()

        # Col 0 : P, dP/dt/T, sigma
        self.pg_P = self._add_ddac_plot(row=0, col=0, y_label='P (GPa)')

        self.pg_dPdt = self._add_ddac_plot(row=1, col=0, y_label='dP/dt (GPa/ms), T (K)')

        self.pg_sigma = self._add_ddac_plot(row=2, col=0, x_label='Time (s)', y_label='sigma (nm)')


        self.pg_dPdt.setXLink(self.pg_P)

        self.pg_sigma.setXLink(self.pg_P)

        
        # Col 1 : IMAGE + Δλ
        self.pg_movie = self._add_ddac_plot(row=0, col=1, show_grid=False)
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


        self.pg_dlambda = self._add_ddac_plot(row=2, col=1, x_label='Spectrum index', y_label='Δλ12 (nm)')

        # ================== COURBES PERSISTANTES ==================
        self.curves_P = []        # une courbe par jauge
        self.curves_dPdt = []
        self.curves_sigma = []
        self.curves_T = []
        self.curve_piezo_list = []
        self.curve_corr_list = []
        self.curves_dlambda = []

        # Lignes de repère : temps sélectionné synchronisé sur tous les graphes
        # et base horizontale de pression.
        self.line_t_P = self._create_marker_line(angle=90)
        self.line_t_dPdt = self._create_marker_line(angle=90)
        self.line_t_sigma = self._create_marker_line(angle=90)
        self.line_t_dlambda = self._create_marker_line(angle=90)

        self.line_t_spec_P = self._create_marker_line(angle=90)
        self.line_t_spec_dPdt = self._create_marker_line(angle=90)
        self.line_t_spec_sigma = self._create_marker_line(angle=90)
        self.line_t_spec_dlambda = self._create_marker_line(angle=90)

        self.line_p0 = self._create_marker_line(angle=0, pos=0)
        self.pg_P.addItem(self.line_p0)
        self.pg_P.addItem(self.line_t_P)
        self.pg_P.addItem(self.line_t_spec_P)
        self.pg_dPdt.addItem(self.line_t_dPdt)
        self.pg_dPdt.addItem(self.line_t_spec_dPdt)
        self.pg_sigma.addItem(self.line_t_sigma)
        self.pg_sigma.addItem(self.line_t_spec_sigma)
        self.pg_dlambda.addItem(self.line_t_dlambda)
        self.pg_dlambda.addItem(self.line_t_spec_dlambda)

        self.line_nspec = self._create_marker_line(angle=90)
        self.pg_dlambda.addItem(self.line_nspec)
        self.curve_P_extrap = self.pg_P.plot(
            [],
            [],
            pen=pg.mkPen((80, 255, 120), width=2, style=Qt.DashLine),
            name="P extrapolée",
        )
        self.curve_P_extrap.setZValue(12_500)
        self._ddac_pressure_ref_time = None
        self._ddac_pressure_ref_values = None
        self._ddac_zone_summary_text = ""
        self._time_unit_symbol = "s"
        self._time_unit_scale = 1.0
        self._dpdt_unit_symbol = "GPa/s"
        self._dpdt_scale_from_seconds = 1.0
        self._update_time_axis_labels()

        self.c_m_base = make_c_m(self.current_theme)
        self.color=[]

        # ================== MARQUEURS DE CLIC (SCATTER CROIX) ==================
        # Un scatter par graphe pour montrer la position exacte du clic

        self.scatter_P = self._create_scatter_marker(symbol='+')
        self.pg_P.addItem(self.scatter_P)

        self.scatter_dPdt = self._create_scatter_marker(symbol='+')
        self.pg_dPdt.addItem(self.scatter_dPdt)

        self.scatter_sigma = self._create_scatter_marker(symbol='+')
        self.pg_sigma.addItem(self.scatter_sigma)

        self.scatter_dlambda = self._create_scatter_marker(symbol='+')
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
        self.selected_spec_time = None
        self.selected_frame_time = None
        self._last_ddac_click_time = None
        self._last_ddac_click_target = None

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

        self.dpdt_range_entry = QSpinBox()
        self.dpdt_range_entry.setRange(2, 100)
        self.dpdt_range_entry.setValue(3)
        controls_layout.addWidget(self.dpdt_range_entry)
        self.phase_patterns_path = os.path.join(os.path.dirname(__file__), "H2O.txt")
        if not os.path.exists(self.phase_patterns_path):
            self.phase_patterns_path = os.path.join(os.path.dirname(__file__), "phase_patterns.txt")
        self.phase_patterns = self._load_phase_patterns(self.phase_patterns_path)
        self.phase_regions = []
        self._phase_sync_in_progress = False
        self.selected_phase_index = None
        self.selected_phase_template = None

        """Crée 2 régions (fit & camera) sur tous les graphes temporels (pas sur le film)."""

        # --- états / garde-fous ---
        self._block_fit = False
        self._block_cam = False

        self._fit_visible = True
        self._cam_visible = True
        self._ddac_multi_zone_visible = False
        self._ddac_multi_zone_syncing = False
        self._ddac_multi_zone_range = None

        self._spec_time = None   # np.ndarray : temps des spectres (Time)
        self._cam_time = None    # np.ndarray : temps caméra (t_cam)

        # zone caméra (utilisée dans play_movie via _get_movie_bounds)
        self.zone_movie = [None, None]

        # plots concernés (temps)
        self._time_plots = (self.pg_P, self.pg_dPdt, self.pg_sigma, self.pg_dlambda)

        # --- création des régions ---
        self.fit_regions = []
        self.cam_regions = []

        for plot in self._time_plots:
            r_fit = pg.LinearRegionItem(values=(0.0, 1.0), movable=True)
            r_fit.setBrush(pg.mkBrush(255, 165, 0, 45))  # orange léger
            r_fit.setZValue(10_000)
            plot.addItem(r_fit)
            r_fit.sigRegionChanged.connect(partial(self._on_fit_region_changed, r_fit))
            self.fit_regions.append(r_fit)

            r_cam = pg.LinearRegionItem(values=(0.0, 1.0), movable=True)
            r_cam.setBrush(pg.mkBrush(80, 180, 255, 35))  # bleu léger
            r_cam.setZValue(9_000)
            plot.addItem(r_cam)
            r_cam.sigRegionChanged.connect(partial(self._on_cam_region_changed, r_cam))
            self.cam_regions.append(r_cam)

        self.zone_multi_P = pg.LinearRegionItem(values=(0.0, 1.0), movable=True)
        self.zone_multi_P.setBrush(pg.mkBrush(80, 255, 120, 45))
        self.zone_multi_P.setZValue(11_000)
        self.zone_multi_P.setVisible(False)
        self.pg_P.addItem(self.zone_multi_P)

        self.zone_multi_dPdt = pg.LinearRegionItem(values=(0.0, 1.0), movable=True)
        self.zone_multi_dPdt.setBrush(pg.mkBrush(80, 255, 120, 45))
        self.zone_multi_dPdt.setZValue(11_000)
        self.zone_multi_dPdt.setVisible(False)
        self.pg_dPdt.addItem(self.zone_multi_dPdt)

        self.zone_multi_diff_int = pg.LinearRegionItem(values=(0.0, 1.0), movable=True)
        self.zone_multi_diff_int.setBrush(pg.mkBrush(80, 255, 120, 45))
        self.zone_multi_diff_int.setZValue(11_000)
        self.zone_multi_diff_int.setVisible(False)
        self.pg_sigma.addItem(self.zone_multi_diff_int)
        self._connect_ddac_multi_zone_signals()

        layout_graphique.addLayout(controls_layout)
        layout_graphique.addLayout(self._build_phase_controls())

        group_graphique.setLayout(layout_graphique)

        self.grid_layout.addWidget(group_graphique, 0, 3, 3, 1)

        # Timer Qt pour le mode "lecture"
        self.timerMovie = QTimer(self)
        self.timerMovie.timeout.connect(self.play_movie)
    
    

        # Un clic sur n'importe quel graphe temporel aligne les repères, déclenche
        # la sélection spectre/film et met à jour les marqueurs de mesure.
        self.pg_P.scene().sigMouseClicked.connect(self._on_ddac_click)
        self.pg_dPdt.scene().sigMouseClicked.connect(self._on_ddac_click)
        self.pg_sigma.scene().sigMouseClicked.connect(self._on_ddac_click)
        self.pg_dlambda.scene().sigMouseClicked.connect(self._on_ddac_click)



        # ================== FACTEURS DE LIGNES (2, 2, 1) ==================
        self._ddac_row_factors = (4, 1, 1)
        # exemple : colonne temps / colonne image de même largeur
        self._ddac_col_factors = (1, 1)

        self._update_graphicslayout_sizes(
            self.pg_ddac,
            row_factors=self._ddac_row_factors,
            col_factors=self._ddac_col_factors,
        )

    def _build_phase_controls(self):
        layout = QVBoxLayout()
        box = QGroupBox("DAC Element")
        box_layout = QVBoxLayout()

        top_line = QHBoxLayout()
        self.phase_visible_box = QCheckBox("Afficher phases")
        self.phase_visible_box.setChecked(False)
        self.phase_visible_box.toggled.connect(self._on_phase_visibility_toggled)
        top_line.addWidget(self.phase_visible_box)

        self.phase_element_selector = QComboBox()
        self.phase_element_selector.setEditable(False)
        self.phase_element_selector.addItems(sorted(self.phase_patterns.keys()))
        if self.phase_element_selector.count() == 0:
            self.phase_element_selector.addItem("H2O")
        self.phase_element_selector.currentTextChanged.connect(self._on_phase_element_changed)
        top_line.addWidget(QLabel("Élément"))
        top_line.addWidget(self.phase_element_selector)
        box_layout.addLayout(top_line)

        template_line = QHBoxLayout()
        self.phase_template_selector = QComboBox()
        self.phase_template_selector.setEditable(False)
        self.phase_template_selector.currentTextChanged.connect(self._on_phase_template_changed)
        template_line.addWidget(self.phase_template_selector)
        self.phase_file_button = QPushButton("Charger patterns...")
        self.phase_file_button.clicked.connect(self._choose_phase_pattern_file)
        template_line.addWidget(self.phase_file_button)
        box_layout.addLayout(template_line)

        btns = QHBoxLayout()
        self.phase_auto_btn = QPushButton("Auto (pression)")
        self.phase_auto_btn.clicked.connect(self._auto_build_phase_track)
        btns.addWidget(self.phase_auto_btn)

        self.phase_add_btn = QPushButton("Ajouter zone")
        self.phase_add_btn.clicked.connect(self._add_phase_zone)
        btns.addWidget(self.phase_add_btn)

        self.phase_remove_btn = QPushButton("Supprimer zone")
        self.phase_remove_btn.clicked.connect(self._remove_phase_zone)
        btns.addWidget(self.phase_remove_btn)
        box_layout.addLayout(btns)

        self.phase_list = QListWidget()
        self.phase_list.itemSelectionChanged.connect(self._on_phase_selection_changed)
        box_layout.addWidget(self.phase_list)

        box.setLayout(box_layout)
        self.phase_group_box = box
        layout.addWidget(box)
        self._refresh_phase_templates()
        self.phase_list.setMaximumHeight(100)

        self.phase_group_box.setMaximumHeight(200)
        return layout

    def _update_phase_section_height(self):
        if not hasattr(self, "ddac_group_box") or not hasattr(self, "phase_group_box"):
            return
        total_h = self.ddac_group_box.height()
        if total_h <= 0:
            total_h = self.ddac_group_box.sizeHint().height()
        max_h = max(120, int(total_h * 0.20))
        self.phase_group_box.setMaximumHeight(max_h)
        self.phase_group_box.setMinimumHeight(min(120, max_h))

    def _load_phase_patterns(self, path=None):
        patterns = {}
        if path is None:
            path = getattr(self, "phase_patterns_path", os.path.join(os.path.dirname(__file__), "phase_patterns.txt"))
        if not os.path.exists(path):
            return patterns
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                for part in raw.split("\\n"):
                    line = part.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" not in line or "[" not in line or "]" not in line:
                        continue
                    name_part, range_part = line.split(":", 1)
                    tokens = name_part.strip().split()
                    if len(tokens) < 2:
                        continue
                    element = tokens[0]
                    phase_name = " ".join(tokens[1:])
                    bounds = range_part[range_part.index("[") + 1: range_part.index("]")]
                    hi, lo = [float(x.strip()) for x in bounds.split(",")]
                    patterns.setdefault(element, []).append(
                        {"phase": phase_name, "p_high": hi, "p_low": lo}
                    )
        return patterns

    def _phase_color(self, idx):
        palette = [
            (66, 135, 245, 50),
            (52, 168, 83, 50),
            (251, 188, 5, 55),
            (234, 67, 53, 50),
            (171, 71, 188, 50),
            (0, 172, 193, 50),
        ]
        return palette[idx % len(palette)]

    def _phase_color_for_name(self, phase_name: str, element: Optional[str] = None):
        if element is None:
            element = self.phase_element_selector.currentText().strip() if hasattr(self, "phase_element_selector") else "H2O"
        ordered_names = self._phase_names_for_element(element) if hasattr(self, "_phase_names_for_element") else []
        if phase_name in ordered_names:
            idx = ordered_names.index(phase_name)
        else:
            idx = abs(hash(phase_name)) % 6
        return self._phase_color(idx)

        
        
     
    def _recolor_all_runs(self):
        self.c_m_base = make_c_m(self.current_theme)  
        for run_id, state in self.runs.items():
            c = self._get_run_color(state)
            # item liste
            if state.list_item is not None:
                state.list_item.setBackground(QColor(c))
                state.list_item.setForeground(QColor("#000000" if self.current_theme == "light" else "#ffffff"))

            # courbes "run-colored"
            if state.piezo_curve is not None:
                state.piezo_curve.setPen(pg.mkPen(c))

            if state.corr_curve is not None:
                state.corr_curve.setPen(pg.mkPen(c))

            # symbolPen dépendant du run (les symbolBrush restent gauge-colored chez toi)
            for curve in state.curves_P:
                curve.setPen(pg.mkPen(c, width=2))
            for curve in state.curves_dPdt:
                curve.setPen(pg.mkPen(c, width=2))
            for curve in state.curves_sigma:
                curve.setPen(pg.mkPen(c, width=2))
            for curve in state.curves_dlambda:
                curve.setPen(pg.mkPen(c, width=2))

    def attach_spectrum_time(self, time_array):
        """À appeler quand tu as Time (temps des spectres). Sert à borner & synchroniser FitRegion."""
        t = np.asarray(time_array, dtype=float)
        if t.ndim != 1 or len(t) < 2:
            self._spec_time = None
            return

        self._spec_time = t
        tmin, tmax = float(t[0]), float(t[-1])

        for r in self.fit_regions:
            r.setBounds((tmin, tmax))

        # initialiser une région cohérente avec les spinbox
        self._apply_fit_from_spin()


    def _get_run_color(self, state: RunViewState) -> str:
        # self.c_m_base = palette courante (liste de hex)
        if not getattr(self, "c_m_base", None):
            self.c_m_base = make_c_m(self.current_theme)  # ta fonction Tab10 light/dark
        return self.c_m_base[state.color_idx % len(self.c_m_base)]

    def attach_camera_time(self, time_array):
        """À appeler quand tu as t_cam (temps caméra corrélé). Sert à borner & synchroniser CamRegion."""
        t = np.asarray(time_array, dtype=float)
        if t.ndim != 1 or len(t) < 2:
            self._cam_time = None
            return

        self._cam_time = t
        tmin, tmax = float(t[0]), float(t[-1])

        for r in self.cam_regions:
            r.setBounds((tmin, tmax))

        # si zone_movie pas définie, on met tout
        if self.zone_movie[0] is None or self.zone_movie[1] is None:
            self.zone_movie = [tmin, tmax]

        self._apply_cam_from_zone_movie()

    def _on_fit_spin_changed(self, *_):
        if self._block_fit:
            return
        self._apply_fit_from_spin()
        self._update_ddac_multi_zone_range()

    def _effective_dpdt_window(self, size: int) -> int:
        window = int(self.dpdt_range_entry.value()) if hasattr(self, "dpdt_range_entry") else 3
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1
        if window > size:
            window = size if size % 2 == 1 else max(3, size - 1)
        return max(3, window)

    def _on_dpdt_window_changed(self, *_):
        state = self._get_state_for_run()
        if state is not None and self.RUN is not None:
            self._update_curves_for_run(state)
        self._update_ddac_zone_annotations()

    def _apply_fit_from_spin(self):
        """index_start/index_stop (indices) -> FitRegion (temps)."""
        if self._spec_time is None:
            return
        if not (hasattr(self, "index_start_entry") and hasattr(self, "index_stop_entry")):
            return

        i0 = int(self.index_start_entry.value())
        i1 = int(self.index_stop_entry.value())
        if i0 > i1:
            i0, i1 = i1, i0

        i0 = int(np.clip(i0, 0, len(self._spec_time) - 1))
        i1 = int(np.clip(i1, 0, len(self._spec_time) - 1))

        t0 = float(self._spec_time[i0])
        t1 = float(self._spec_time[i1])

        self._block_fit = True
        for r in self.fit_regions:
            r.setRegion((t0, t1))
        self._block_fit = False
        self._update_ddac_multi_zone_range()


        # -------------------------
        #  FIT : région -> spinbox
        # -------------------------

    def _on_fit_region_changed(self, src_region, *_):
        """FitRegion (temps) -> index_start/index_stop (indices)."""
        if self._block_fit or self._spec_time is None:
            return

        t0, t1 = src_region.getRegion()
        if t0 > t1:
            t0, t1 = t1, t0

        i0 = int(np.argmin(np.abs(self._spec_time - t0)))
        i1 = int(np.argmin(np.abs(self._spec_time - t1)))
        if i0 > i1:
            i0, i1 = i1, i0

        self._block_fit = True

        # sync des autres régions fit
        t0s, t1s = float(self._spec_time[i0]), float(self._spec_time[i1])
        for r in self.fit_regions:
            if r is not src_region:
                r.setRegion((t0s, t1s))

        # update spinbox
        if hasattr(self, "index_start_entry") and hasattr(self, "index_stop_entry"):
            self.index_start_entry.blockSignals(True)
            self.index_stop_entry.blockSignals(True)
            self.index_start_entry.setValue(i0)
            self.index_stop_entry.setValue(i1)
            self.index_start_entry.blockSignals(False)
            self.index_stop_entry.blockSignals(False)

        self._block_fit = False
        self._update_ddac_multi_zone_range()

    def _compute_ddac_multi_zone_range(self):
        if not hasattr(self, "index_start_entry") or not hasattr(self, "index_stop_entry"):
            return None

        start_index = int(self.index_start_entry.value())
        stop_index = int(self.index_stop_entry.value())
        if start_index > stop_index:
            start_index, stop_index = stop_index, start_index

        time_values = self._spec_time
        if time_values is None or len(time_values) == 0:
            start_time = float(start_index)
            stop_time = float(stop_index)
            if start_time == stop_time:
                stop_time = start_time + 1.0
            return (start_time, stop_time)

        start_index = int(np.clip(start_index, 0, len(time_values) - 1))
        stop_index = int(np.clip(stop_index, 0, len(time_values) - 1))
        start_time = float(time_values[start_index])

        if stop_index + 1 < len(time_values):
            stop_time = float(time_values[stop_index + 1])
        else:
            stop_time = float(time_values[stop_index])
            if len(time_values) >= 2:
                stop_time += float(time_values[-1] - time_values[-2])
            else:
                stop_time += 1.0

        if start_time > stop_time:
            start_time, stop_time = stop_time, start_time
        return (start_time, stop_time)

    def _update_ddac_multi_zone_range(self) -> None:
        if self._ddac_multi_zone_syncing:
            return

        zone_range = self._compute_ddac_multi_zone_range()
        self._ddac_multi_zone_range = zone_range
        if zone_range is None:
            return
        for attr in ("zone_multi_P", "zone_multi_dPdt", "zone_multi_diff_int"):
            zone = getattr(self, attr, None)
            if zone is not None:
                zone.setRegion(zone_range)
        self._update_ddac_zone_annotations()

    def _apply_ddac_multi_zone_visibility(self) -> None:
        visible = bool(self._ddac_multi_zone_visible)
        for attr in ("zone_multi_P", "zone_multi_dPdt", "zone_multi_diff_int"):
            zone = getattr(self, attr, None)
            if zone is not None:
                zone.setVisible(visible)
        if not visible and hasattr(self, "curve_P_extrap"):
            self._update_curve_safe(self.curve_P_extrap, [], [])
            self._ddac_zone_summary_text = ""

    def set_ddac_multi_zone_visibility(self, checked: bool) -> None:
        checked = bool(checked)
        self._ddac_multi_zone_visible = checked
        self._update_ddac_multi_zone_range()
        self._apply_ddac_multi_zone_visibility()
        self._update_ddac_zone_annotations()

        btn = getattr(self, "btn_zone_dpdt", None)
        if btn is not None and bool(btn.isChecked()) != checked:
            btn.blockSignals(True)
            btn.setChecked(checked)
            btn.blockSignals(False)

    def _connect_ddac_multi_zone_signals(self) -> None:
        for attr in ("zone_multi_P", "zone_multi_dPdt", "zone_multi_diff_int"):
            zone = getattr(self, attr, None)
            if zone is None:
                continue
            zone.sigRegionChangeFinished.connect(self._on_ddac_multi_zone_changed)

    def _on_ddac_multi_zone_changed(self) -> None:
        if self._ddac_multi_zone_syncing:
            return
        sender = self.sender()
        region = sender.getRegion() if sender is not None and hasattr(sender, "getRegion") else None
        if region is None:
            return

        start, stop = sorted(map(float, region))
        self._ddac_multi_zone_range = (start, stop)
        self._ddac_multi_zone_syncing = True
        try:
            for attr in ("zone_multi_P", "zone_multi_dPdt", "zone_multi_diff_int"):
                zone = getattr(self, attr, None)
                if zone is None or zone is sender:
                    continue
                zone.setRegion((start, stop))

            start_index, stop_index = self._indices_from_ddac_range(start, stop)
            self._set_batch_indices_from_zone(start_index, stop_index)
            self._update_ddac_zone_annotations()
        finally:
            self._ddac_multi_zone_syncing = False

    def _indices_from_ddac_range(self, start: float, stop: float):
        time_values = self._spec_time
        if time_values is None or len(time_values) == 0:
            start_index = int(round(start))
            stop_index = int(round(stop))
            return (min(start_index, stop_index), max(start_index, stop_index))

        time_array = np.asarray(time_values, dtype=float)
        start_index = int(np.nanargmin(np.abs(time_array - start)))
        stop_index = int(np.nanargmin(np.abs(time_array - stop)))
        return (min(start_index, stop_index), max(start_index, stop_index))

    def _set_batch_indices_from_zone(self, start_index: int, stop_index: int) -> None:
        if start_index > stop_index:
            start_index, stop_index = stop_index, start_index
        if hasattr(self, "index_start_entry"):
            self.index_start_entry.blockSignals(True)
            self.index_start_entry.setValue(start_index)
            self.index_start_entry.blockSignals(False)
        if hasattr(self, "index_stop_entry"):
            self.index_stop_entry.blockSignals(True)
            self.index_stop_entry.setValue(stop_index)
            self.index_stop_entry.blockSignals(False)

    def _get_reference_pressure_series(self):
        t = np.asarray(getattr(self, "_ddac_pressure_ref_time", []), dtype=float)
        p = np.asarray(getattr(self, "_ddac_pressure_ref_values", []), dtype=float)
        if t.size < 2 or p.size != t.size:
            state = self._get_state_for_run()
            if state is None or not state.curves_P or state.curves_P[0] is None:
                return None, None
            x_data, y_data = state.curves_P[0].getData()
            if x_data is None or y_data is None:
                return None, None
            t = np.asarray(x_data, dtype=float)
            p = np.asarray(y_data, dtype=float)
        mask = np.isfinite(t) & np.isfinite(p)
        t = t[mask]
        p = p[mask]
        if t.size < 2:
            return None, None
        order = np.argsort(t)
        return t[order], p[order]

    def _compute_ddac_zone_pressure_stats(self):
        zone = self._ddac_multi_zone_range
        if not zone:
            return None
        t, p = self._get_reference_pressure_series()
        if t is None or p is None:
            return None
        t0, t1 = sorted(map(float, zone))
        if t1 <= t0:
            return None
        p0 = float(np.interp(t0, t, p))
        p1 = float(np.interp(t1, t, p))
        dt_s = float(t1 - t0)
        dt_unit = dt_s * self._time_unit_scale
        dp = float(p1 - p0)
        dpdt_int = float(dp / dt_unit) if dt_unit != 0 else np.nan
        return {"t0": t0, "t1": t1, "p0": p0, "p1": p1, "dt_unit": dt_unit, "dp": dp, "dpdt_int": dpdt_int}

    def _build_extrapolated_pressure_curve(self, t0: float, t1: float, p0: float):
        t_ref, p_ref = self._get_reference_pressure_series()
        if t_ref is None or p_ref is None or t_ref.size < 2:
            return np.array([t0, t1], dtype=float), np.array([p0, p0], dtype=float)

        if p_ref.size >= 5:
            win = self._effective_dpdt_window(p_ref.size)
            if win < p_ref.size:
                p_ref = CL.savgol_filter(p_ref, win, 1)
        t_ref_unit = t_ref * self._time_unit_scale
        dpdt_ref = np.gradient(p_ref, t_ref_unit, edge_order=1)
        t_dense = np.linspace(t0, t1, 200)
        y_dense = np.interp(t_dense, t_ref, dpdt_ref)
        t_dense_unit = t_dense * self._time_unit_scale
        cumulative = np.zeros_like(t_dense_unit)
        if t_dense_unit.size > 1:
            cumulative[1:] = np.cumsum(0.5 * (y_dense[1:] + y_dense[:-1]) * np.diff(t_dense_unit))
        p_extrap = p0 + cumulative
        return t_dense, p_extrap

    def _update_ddac_zone_annotations(self) -> None:
        visible = bool(self._ddac_multi_zone_visible)
        stats = self._compute_ddac_zone_pressure_stats() if visible else None
        if stats is None:
            if hasattr(self, "curve_P_extrap"):
                self._update_curve_safe(self.curve_P_extrap, [], [])
            self._ddac_zone_summary_text = ""
            state = self._get_state_for_run()
            current_t = state.t_cam[self.current_index] if state is not None and state.t_cam else 0.0
            self.f_text_CEDd_print(current_t)
            return

        t0, t1 = stats["t0"], stats["t1"]
        p0 = stats["p0"]
        t_extrap, p_extrap = self._build_extrapolated_pressure_curve(t0, t1, p0)
        self._update_curve_safe(self.curve_P_extrap, t_extrap, p_extrap)

        p1_extrap = float(p_extrap[-1]) if p_extrap.size else stats["p1"]
        dp_extrap = p1_extrap - p0
        dpdt_extrap = float(dp_extrap / stats["dt_unit"]) if stats["dt_unit"] != 0 else np.nan

        self._ddac_zone_summary_text = (
            f"P start: {p0:.3f} P end: {p1_extrap:.3f} ΔP: {dp_extrap:.3f} (all GPa)\n"
            f"Δt: {stats['dt_unit']:.2f} {self._time_unit_symbol}  dP/dt int: {dpdt_extrap:.4f} {self._dpdt_unit_symbol}"
        )
        state = self._get_state_for_run()
        current_t = state.t_cam[self.current_index] if state is not None and state.t_cam else 0.0
        self.f_text_CEDd_print(current_t)

    def _apply_cam_from_zone_movie(self):
        """zone_movie (t0,t1) -> CamRegion (temps)."""
        if self._cam_time is None:
            return
        if self.zone_movie[0] is None or self.zone_movie[1] is None:
            return

        t0, t1 = float(self.zone_movie[0]), float(self.zone_movie[1])
        if t0 > t1:
            t0, t1 = t1, t0

        # clip dans bornes caméra
        t0 = float(np.clip(t0, float(self._cam_time[0]), float(self._cam_time[-1])))
        t1 = float(np.clip(t1, float(self._cam_time[0]), float(self._cam_time[-1])))

        self._block_cam = True
        for r in self.cam_regions:
            r.setRegion((t0, t1))
        self._block_cam = False

    def _on_cam_region_changed(self, src_region, *_):
        """CamRegion (temps) -> zone_movie + sync autres CamRegion."""
        if self._block_cam or self._cam_time is None:
            return
        
        t0, t1 = src_region.getRegion()
        if t0 > t1:
            t0, t1 = t1, t0

        # bornage
        t0 = float(np.clip(t0, float(self._cam_time[0]), float(self._cam_time[-1])))
        t1 = float(np.clip(t1, float(self._cam_time[0]), float(self._cam_time[-1])))

        self._block_cam = True

        # sync autres cam regions
        for r in self.cam_regions:
            if r is not src_region:
                r.setRegion((t0, t1))

        # update zone_movie (pour play_movie / _get_movie_bounds)
        self.zone_movie = [t0, t1]

        self._block_cam = False

    def toggle_fit_region(self):
        self._fit_visible = not self._fit_visible
        for r in self.fit_regions:
            r.setVisible(self._fit_visible)

    def toggle_cam_region(self):
        self._cam_visible = not self._cam_visible
        for r in self.cam_regions:
            r.setVisible(self._cam_visible)

    def _get_state_for_run(self, run_id: Optional[str] = None) -> Optional[RunViewState]:
        if run_id is None:
            run_id = self.current_run_id
        return self.runs.get(run_id)

    def _save_current_run(self):
        """Sauvegarde le RUN courant dans son état RunViewState."""

        state = self._get_state_for_run()
        if state is not None and self.RUN is not None:
            # Copie profonde indispensable ici : on doit pouvoir revenir à l'état
            # précédent d'un RUN lorsque l'utilisateur change d'onglet.
            state.ced = copy.deepcopy(self.RUN)
            if getattr(state, "cap", None) is not None:
                try:
                    state.cap.release()
                except Exception:
                    pass
                state.cap = None

    def _finalize_run_selection(self, state: RunViewState, name_select: str):
        self.CLEAR_ALL()
        """Replace l'ancienne logique basée sur index_select par l'état RunViewState."""
        self.current_run_id = self._get_run_id(state.ced)
        # On travaille sur une copie pour préserver l'instantané stocké dans le RunViewState
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

        if state.cap is None and getattr(state.ced, "folder_Movie", None):
            cap, _fps, _nb = self.Read_Movie(state.ced)
            state.cap = cap

        if state.index_cam is not None and len(state.index_cam)>0:
            self.current_index = len(state.index_cam) // 2
            self.slider.setMaximum(max(0, len(state.index_cam) - 1))
            self.slider.setValue(self.current_index)
            self.attach_camera_time(time_array=state.t_cam)
        print("READY bool",bool(state.index_cam))
        self._set_movie_controls_enabled(bool(state.index_cam))


        if state.time is not None and len(state.time)>0:
            x_min, x_max = min(state.time), max(state.time)
            self.pg_P.setXRange(x_min, x_max, padding=0.01)
            self.pg_dPdt.setXRange(x_min, x_max, padding=0.01)
            self.pg_sigma.setXRange(x_min, x_max, padding=0.01)
            x_dlambda = self._dlambda_x_values(state)
            if x_dlambda:
                self.pg_dlambda.setXRange(min(x_dlambda), max(x_dlambda), padding=0.01)
            self.attach_spectrum_time(time_array=state.time)

            self._apply_fit_from_spin()
            self._update_ddac_zone_annotations()


        self._update_movie_frame()

        self.label_CED.setText( f"CEDd {name_select} fps :{text_fps}")

        
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
        self._reload_phase_ui_from_state()

    def _sync_phase_state_to_run(self, state: Optional[RunViewState] = None):
        state = state or self._get_state_for_run()
        if state is None or self.RUN is None:
            return
        state.active_phase_element = self.phase_element_selector.currentText().strip() or "H2O"
        setattr(self.RUN, "phase_tracks", copy.deepcopy(state.phase_tracks))
        setattr(self.RUN, "active_phase_element", state.active_phase_element)

    def _reload_phase_ui_from_state(self):
        state = self._get_state_for_run()
        if state is None:
            self.phase_list.clear()
            self._clear_phase_regions()
            return

        state.phase_tracks = copy.deepcopy(getattr(self.RUN, "phase_tracks", getattr(state, "phase_tracks", {})))
        state.active_phase_element = getattr(self.RUN, "active_phase_element", state.active_phase_element or "H2O")

        if state.active_phase_element:
            self.phase_element_selector.blockSignals(True)
            if self.phase_element_selector.findText(state.active_phase_element) < 0:
                self.phase_element_selector.addItem(state.active_phase_element)
            self.phase_element_selector.setCurrentText(state.active_phase_element)
            self.phase_element_selector.blockSignals(False)
        self._refresh_phase_templates()
        self._refresh_phase_list()
        self._render_phase_regions()

    def _refresh_phase_list(self):
        self.phase_list.blockSignals(True)
        self.phase_list.clear()
        state = self._get_state_for_run()
        if state is None:
            self.phase_list.blockSignals(False)
            return
        element = self.phase_element_selector.currentText().strip() or "H2O"
        zones = state.phase_tracks.get(element, [])
        for zone in zones:
            item = QListWidgetItem(f"{zone['name']} [{zone['start']:.3f}, {zone['end']:.3f}] s")
            r, g, b, a = self._phase_color_for_name(zone["name"], element)
            item.setBackground(QColor(r, g, b, min(180, a + 80)))
            item.setForeground(QColor("#ffffff" if self.current_theme == "dark" else "#000000"))
            self.phase_list.addItem(item)
        if len(zones) == 0:
            for tpl in self._phase_names_for_element(element):
                item = QListWidgetItem(f"Template: {tpl} [à définir]")
                r, g, b, a = self._phase_color_for_name(tpl, element)
                item.setBackground(QColor(r, g, b, min(140, a + 60)))
                item.setForeground(QColor("#ffffff" if self.current_theme == "dark" else "#000000"))
                self.phase_list.addItem(item)
        self.phase_list.blockSignals(False)

    def _phase_names_for_element(self, element):
        return [row["phase"] for row in self.phase_patterns.get(element, [])]

    def _refresh_phase_templates(self):
        element = self.phase_element_selector.currentText().strip() or "H2O"
        names = self._phase_names_for_element(element)
        self.phase_template_selector.blockSignals(True)
        self.phase_template_selector.clear()
        self.phase_template_selector.addItems(names)
        self.phase_template_selector.blockSignals(False)
        self.selected_phase_template = self.phase_template_selector.currentText().strip() if names else None
        self._update_add_phase_button_label()

    def _update_add_phase_button_label(self):
        phase_name = (self.selected_phase_template or "").strip()
        if phase_name:
            self.phase_add_btn.setText(f"Ajouter zone: {phase_name}")
        else:
            self.phase_add_btn.setText("Ajouter zone")

    def _on_phase_template_changed(self, text):
        self.selected_phase_template = text.strip() or None
        self._update_add_phase_button_label()
        self._apply_selected_phase_template_to_zone()

    def _apply_selected_phase_template_to_zone(self):
        """Applique le template choisi à la zone actuellement sélectionnée."""
        state = self._get_state_for_run()
        if state is None:
            return
        phase_name = (self.selected_phase_template or "").strip()
        if not phase_name:
            return
        element = self.phase_element_selector.currentText().strip() or "H2O"
        zones = state.phase_tracks.get(element, [])
        idx = self.phase_list.currentRow()
        if idx < 0:
            idx = self.selected_phase_index if self.selected_phase_index is not None else -1
        if idx < 0 or idx >= len(zones):
            return
        if zones[idx].get("name") == phase_name:
            return
        zones[idx]["name"] = phase_name
        self._refresh_phase_list()
        self.phase_list.setCurrentRow(idx)
        self._render_phase_regions()
        self._sync_phase_state_to_run(state)

    def _choose_phase_pattern_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier de patterns de phases",
            os.path.dirname(getattr(self, "phase_patterns_path", "")) or ".",
            "Text files (*.txt);;All files (*)",
        )
        if not filename:
            return
        loaded = self._load_phase_patterns(filename)
        if not loaded:
            self.text_box_msg.setText("Aucun pattern valide trouvé dans le fichier sélectionné.")
            return
        self.phase_patterns_path = filename
        self.phase_patterns = loaded
        current_element = self.phase_element_selector.currentText().strip() or "H2O"
        self.phase_element_selector.blockSignals(True)
        self.phase_element_selector.clear()
        self.phase_element_selector.addItems(sorted(self.phase_patterns.keys()))
        if self.phase_element_selector.findText(current_element) >= 0:
            self.phase_element_selector.setCurrentText(current_element)
        self.phase_element_selector.blockSignals(False)
        self._refresh_phase_templates()
        self._refresh_phase_list()
        self.text_box_msg.setText(f"Patterns chargés: {os.path.basename(filename)}")

    def _clear_phase_regions(self):
        for reg in getattr(self, "phase_regions", []):
            try:
                self.pg_P.removeItem(reg)
            except Exception:
                pass
        self.phase_regions = []

    def _render_phase_regions(self):
        self._clear_phase_regions()
        if not self.phase_visible_box.isChecked():
            return
        state = self._get_state_for_run()
        if state is None:
            return
        element = self.phase_element_selector.currentText().strip() or "H2O"
        zones = state.phase_tracks.get(element, [])
        for idx, zone in enumerate(zones):
            reg = pg.LinearRegionItem(values=(zone["start"], zone["end"]), movable=True)
            reg.setBrush(pg.mkBrush(*self._phase_color_for_name(zone["name"], element)))
            self._set_region_pen(reg, pg.mkPen(255, 255, 255, 110, width=1))
            reg.setZValue(8_500 + idx)
            reg.sigRegionChangeFinished.connect(partial(self._on_phase_region_changed, idx=idx))
            self.pg_P.addItem(reg)
            self.phase_regions.append(reg)
        self._apply_phase_selection_visuals()

    def _on_phase_visibility_toggled(self, _checked):
        self._refresh_phase_list()
        self._render_phase_regions()

    def _on_phase_element_changed(self, text):
        state = self._get_state_for_run()
        if state is None:
            return
        if text and self.phase_element_selector.findText(text) < 0:
            self.phase_element_selector.addItem(text)
        state.active_phase_element = text or "H2O"
        self._refresh_phase_templates()
        self._refresh_phase_list()
        self._render_phase_regions()
        self._sync_phase_state_to_run(state)

    def _on_phase_selection_changed(self):
        self.selected_phase_index = self.phase_list.currentRow()
        item = self.phase_list.currentItem()
        if item is not None:
            txt = item.text()
            if txt.startswith("Template: "):
                self.selected_phase_template = txt.replace("Template: ", "").split("[", 1)[0].strip()
                if self.phase_template_selector.findText(self.selected_phase_template) >= 0:
                    self.phase_template_selector.setCurrentText(self.selected_phase_template)
                self._update_add_phase_button_label()
        self._apply_phase_selection_visuals()

    def _set_region_pen(self, region, pen):
        for line in getattr(region, "lines", []):
            line.setPen(pen)

    def _apply_phase_selection_visuals(self):
        for idx, reg in enumerate(getattr(self, "phase_regions", [])):
            if idx == self.selected_phase_index:
                self._set_region_pen(reg, pg.mkPen(255, 255, 255, 220, width=2))
                reg.setZValue(100)
            else:
                self._set_region_pen(reg, pg.mkPen(255, 255, 255, 110, width=1))

    def _select_phase_at_time(self, t_value: float):
        state = self._get_state_for_run()
        if state is None:
            return
        element = self.phase_element_selector.currentText().strip() or "H2O"
        zones = state.phase_tracks.get(element, [])
        for idx, zone in enumerate(zones):
            if zone["start"] <= t_value <= zone["end"]:
                self.phase_list.setCurrentRow(idx)
                self.selected_phase_index = idx
                self._apply_phase_selection_visuals()
                return

    def _next_available_phase_name(self, element, zones):
        names = self._phase_names_for_element(element)
        if not names:
            return f"Phase {len(zones) + 1}"
        used = {z["name"] for z in zones}
        if self.selected_phase_template in names and self.selected_phase_template not in used:
            return self.selected_phase_template
        for name in names:
            if name not in used:
                return name
        return names[-1]

    def _ensure_time_pressure_for_phases(self, state: RunViewState) -> bool:
        """Garantit la disponibilité de time + pression pour la génération auto des phases."""
        if len(state.time) > 0 and len(state.pressure_series) > 0:
            return True
        if self.RUN is None:
            return False
        try:
            (
                l_P,
                _l_sigma_P,
                _l_lambda,
                _l_fwhm,
                _l_spe,
                _l_T,
                _l_sigma_T,
                Time,
                _spectre_number,
                _time_amp,
                _amp,
                _Gauges_RUN,
            ) = self.Read_RUN(self.RUN)
        except Exception as exc:
            self.text_box_msg.setText(f"Erreur lecture RUN pour phases auto: {exc}")
            return False

        state.time = Time if Time is not None else []
        state.pressure_series = l_P if l_P is not None else []
        return len(state.time) > 0 and len(state.pressure_series) > 0

    def _auto_build_phase_track(self):
        state = self._get_state_for_run()
        if state is None:
            self.text_box_msg.setText("Impossible: aucun run sélectionné.")
            return
        if not self._ensure_time_pressure_for_phases(state):
            self.text_box_msg.setText("Impossible: pas de données temps/pression pour générer des phases.")
            return
        element = self.phase_element_selector.currentText().strip() or "H2O"
        patterns = self.phase_patterns.get(element, [])
        if not patterns:
            self.text_box_msg.setText(f"Aucun pattern chargé pour {element}.")
            return

        time = np.asarray(state.time, dtype=float)
        pressure = np.asarray(state.pressure_series[0], dtype=float)
        n = min(len(time), len(pressure))
        if n < 2:
            return
        time = time[:n]
        pressure = pressure[:n]
        pressure_for_phase = pressure
        smooth_window = 5
        if n >= smooth_window:
            kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
            pressure_for_phase = np.convolve(pressure, kernel, mode="same")

        def phase_for_p(p):
            for row in patterns:
                pmax = max(row["p_high"], row["p_low"])
                pmin = min(row["p_high"], row["p_low"])
                if pmin <= p <= pmax:
                    return row["phase"]
            return None
        
        raw_phase_labels = [phase_for_p(p) for p in pressure_for_phase]
        min_points_per_zone = 4
        if min_points_per_zone > 1:
            labels = list(raw_phase_labels)
            changed = True
            while changed:
                changed = False
                i = 0
                while i < n:
                    j = i + 1
                    while j < n and labels[j] == labels[i]:
                        j += 1
                    run_length = j - i
                    if 0 < run_length < min_points_per_zone:
                        left_label = labels[i - 1] if i > 0 else None
                        right_label = labels[j] if j < n else None
                        replacement = left_label if left_label is not None else right_label
                        if replacement is not None and replacement != labels[i]:
                            for k in range(i, j):
                                labels[k] = replacement
                            changed = True
                    i = j
            phase_labels = labels
        else:
            phase_labels = raw_phase_labels


        zones = []
        current_name = phase_labels[0]
        start = time[0]
        for i in range(1, n):
            name = phase_labels[i]
            if name != current_name:
                if current_name is not None:
                    zones.append({"name": current_name, "start": float(start), "end": float(time[i])})
                start = time[i]
                current_name = name
        if current_name is not None:
            zones.append({"name": current_name, "start": float(start), "end": float(time[-1])})

        state.phase_tracks[element] = zones
        state.active_phase_element = element
        self._refresh_phase_list()
        self._render_phase_regions()
        self._sync_phase_state_to_run(state)

    def _add_phase_zone(self):
        state = self._get_state_for_run()
        if state is None or len(state.time) == 0:
            return
        element = self.phase_element_selector.currentText().strip() or "H2O"
        zones = state.phase_tracks.setdefault(element, [])
        t0, t1 = float(min(state.time)), float(max(state.time))
        zones.sort(key=lambda z: z["start"])
        anchor_time = self.selected_spec_time
        if anchor_time is None:
            anchor_time = self.x_clic if getattr(self, "x_clic", None) is not None else t0
        anchor_time = float(min(max(anchor_time, t0), t1))

        prev_zone = None
        next_zone = None
        for zone in zones:
            z_start = float(zone["start"])
            z_end = float(zone["end"])
            if z_end <= anchor_time:
                prev_zone = zone
            elif z_start >= anchor_time:
                next_zone = zone
                break

        lower_bound = float(prev_zone["end"]) if prev_zone is not None else t0
        upper_bound = float(next_zone["start"]) if next_zone is not None else t1

        if lower_bound >= upper_bound:
            # Recours: on évite la zone pleine plage et on propose une petite fenêtre locale.
            center = anchor_time
            dt = max((t1 - t0) * 0.01, 1e-6)
            lower_bound = max(t0, center - dt)
            upper_bound = min(t1, center + dt)
            if lower_bound >= upper_bound:
                lower_bound, upper_bound = t0, t1

        phase_name = self._next_available_phase_name(element, zones)
        self.selected_phase_template = phase_name
        new_zone = {"name": phase_name, "start": lower_bound, "end": upper_bound}
        zones.append(new_zone)
        zones.sort(key=lambda z: z["start"])
        self._refresh_phase_list()
        selected_idx = zones.index(new_zone)
        self.phase_list.setCurrentRow(selected_idx)
        self._render_phase_regions()
        self._sync_phase_state_to_run(state)

    def _remove_phase_zone(self):
        state = self._get_state_for_run()
        if state is None:
            return
        element = self.phase_element_selector.currentText().strip() or "H2O"
        zones = state.phase_tracks.get(element, [])
        idx = self.phase_list.currentRow()
        if idx < 0 or idx >= len(zones):
            return
        del zones[idx]
        self._refresh_phase_list()
        self._render_phase_regions()
        self._sync_phase_state_to_run(state)

    def _on_phase_region_changed(self, region, idx):
        if self._phase_sync_in_progress:
            return
        state = self._get_state_for_run()
        if state is None:
            return
        element = self.phase_element_selector.currentText().strip() or "H2O"
        zones = state.phase_tracks.get(element, [])
        if idx >= len(zones):
            return
        self.selected_phase_index = idx
        self.phase_list.setCurrentRow(idx)
        start, end = region.getRegion()
        zones[idx]["start"] = float(min(start, end))
        zones[idx]["end"] = float(max(start, end))
        zones.sort(key=lambda z: z["start"])
        self._phase_sync_in_progress = True
        try:
            self._refresh_phase_list()
            self._render_phase_regions()
        finally:
            self._phase_sync_in_progress = False
        self._sync_phase_state_to_run(state)

    def _ensure_curve_at(self, curve_list, index, plot_widget, **plot_kwargs):
        """Garantit la présence d'une courbe à l'index donné et la retourne."""

        while len(curve_list) <= index:
            curve_list.append(None)

        curve = curve_list[index]
        if curve is None:
            curve = plot_widget.plot(**plot_kwargs)
            curve_list[index] = curve
        return curve

    def _curve_is_empty(self, curve: Optional[pg.PlotDataItem]) -> bool:
        if curve is None:
            return True
        x_old, y_old = curve.getData()
        if x_old is None or y_old is None:
            return True
        return len(x_old) == 0 and len(y_old) == 0

    def _update_curve_safe(self, curve: Optional[pg.PlotDataItem], x_data, y_data) -> bool:
        """Met à jour une courbe uniquement si le contenu change.

        Les données sont converties via ``np.asarray(..., copy=False)`` pour éviter
        les copies inutiles. Si les deux axes sont vides et que la courbe l'est
        déjà, aucun appel ``setData`` n'est déclenché.
        """

        if curve is None:
            return False

        x_array = np.asarray(x_data) if x_data is not None else np.array([])
        y_array = np.asarray(y_data) if y_data is not None else np.array([])

        if x_array.size == 0 and y_array.size == 0 and self._curve_is_empty(curve):
            return False

        x_old, y_old = curve.getData()
        if (
            x_old is not None
            and y_old is not None
            and len(x_old) == len(x_array)
            and len(y_old) == len(y_array)
            and np.array_equal(x_old, x_array)
            and np.array_equal(y_old, y_array)
        ):
            return False

        curve.setData(x_array, y_array)
        return True

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
        x_dlambda = np.asarray(self._dlambda_x_values(state), dtype=float)
        if x_dlambda.size >= 2:
            self._apply_viewbox_limits(vb_dlambda, x_dlambda, state.curves_dlambda)

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
        if hasattr(self, "_mark_summary_dirty") and hasattr(self, "_flush_summary_dirty"):
            self._mark_summary_dirty(self.index_spec)
            self._flush_summary_dirty()
        elif hasattr(self, "_corr_summary_for_specs"):
            self._corr_summary_for_specs(self.index_spec)
        else:
            self.RUN.Corr_Summary(num_spec=self.index_spec, All=False)
        state = self._get_state_for_run()
        if state is None:
            self.text_box_msg.setText("Aucun état RunViewState pour rafraîchir ce CEDd")
            return

        run_snapshot = copy.deepcopy(self.RUN)
        request_id = int(getattr(self, "_refresh_request_counter", 0)) + 1
        self._refresh_request_counter = request_id
        self._active_refresh_request_id = request_id
        run_id = self.current_run_id

        self._submit_background_task(
            lambda: {
                "request_id": request_id,
                "run_id": run_id,
                "run_snapshot": run_snapshot,
                "refresh_data": self.Read_RUN(run_snapshot),
            },
            result_slot=self._on_refresh_data_ready,
            description="Rafraîchissement courbes dDAC…",
        )

    @pyqtSlot(object)
    def _on_refresh_data_ready(self, payload):
        if not payload:
            return
        request_id = payload.get("request_id")
        if request_id != getattr(self, "_active_refresh_request_id", None):
            return

        run_id = payload.get("run_id")
        state = self.runs.get(run_id)
        if state is None:
            return

        run_snapshot = payload.get("run_snapshot")
        refresh_data = payload.get("refresh_data")
        if run_snapshot is None or refresh_data is None:
            return

        state.ced = run_snapshot
        if run_id == self.current_run_id:
            self.RUN = run_snapshot
        self._update_curves_for_run_data(state, refresh_data, run_source=run_snapshot)

    def _update_curves_for_run(self, state: RunViewState):
        """Met à jour les courbes PyQtGraph à partir des données d'un `RunViewState`.

        Les listes temporelles et spectrales calculées par `Read_RUN` sont
        réaffectées sur les courbes déjà créées (P, dP/dt, T, Δλ, piézo,
        corrélation). Les attributs `state.time`, `state.spectre_number` et les
        tracés sont actualisés sans recréer d'objets graphiques.
        """

        refresh_data = self.Read_RUN(self.RUN)
        self._update_curves_for_run_data(state, refresh_data, run_source=self.RUN)

    def _update_curves_for_run_data(self, state: RunViewState, refresh_data, run_source):
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
        ) = refresh_data

        state.time = Time
        state.spectre_number = spectre_number
        state.pressure_series = l_P
        self._set_time_display_unit(Time)

        for i, G in enumerate(Gauges_RUN):
            l_p_filtre = CL.savgol_filter(l_P[i], self.dpdt_range_entry.value(), 1) if len(l_P[i]) > 0 else np.array([])
            if len(l_p_filtre) > 4 and len(Time) > 4:
                dps = [
                    (l_p_filtre[x + 1] - l_p_filtre[x - 1]) / (Time[x + 1] - Time[x - 1]) * self._dpdt_scale_from_seconds
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
            self._update_curve_safe(curve_P, Time, l_P[i])

            curve_dPdt = self._ensure_curve_at(state.curves_dPdt, i, self.pg_dPdt, **curve_kwargs)
            self._update_curve_safe(curve_dPdt, time_dps, dps)

            curve_sigma = self._ensure_curve_at(state.curves_sigma, i, self.pg_sigma, **curve_kwargs)
            self._update_curve_safe(curve_sigma, Time, l_fwhm[i])
        
        if len(l_P) > 0 and len(Time) == len(l_P[0]):
            self._ddac_pressure_ref_time = np.asarray(Time, dtype=float)
            self._ddac_pressure_ref_values = np.asarray(l_P[0], dtype=float)

        has_T = "RuSmT" in [x.name_spe for x in run_source.Gauges_init]
        if has_T:
            curve_T = self._ensure_curve_at(
                state.curves_T,
                0,
                self.pg_dPdt,
                pen=pg.mkPen('darkred'),
                symbol='t',
                symbolBrush='darkred',
                symbolSize=10,
            )
            self._update_curve_safe(curve_T, Time, l_T[-1] if l_T else [])
        elif state.curves_T:
            self._update_curve_safe(state.curves_T[0], [], [])

        for i, spe in enumerate(l_spe):
            x_dlambda = self._dlambda_x_values(state, spe)
            curve = self._ensure_curve_at(
                state.curves_dlambda,
                i,
                self.pg_dlambda,
                pen=None,
                symbol='+',
                symbolPen=pg.mkPen(self._get_run_color(state)),
            )
            self._update_curve_safe(curve, x_dlambda, spe[:len(x_dlambda)] if hasattr(spe, "__len__") else spe)
        for extra_index in range(len(l_spe), len(state.curves_dlambda)):
            self._update_curve_safe(state.curves_dlambda[extra_index], [], [])

        if run_source.data_Oscillo is not None:
            state.piezo_curve = self._get_or_create_curve(state.piezo_curve, self.pg_P, pen=pg.mkPen(self._get_run_color(state)))
            self._update_curve_safe(state.piezo_curve, time_amp, amp)
        elif state.piezo_curve is not None:
            self._update_curve_safe(state.piezo_curve, [], [])

        if state.corr_curve is not None:
            if self.chk_show_corr.isChecked() and state.correlations:
                self._update_curve_safe(state.corr_curve, state.t_cam, state.correlations)
            else:
                self._update_curve_safe(state.corr_curve, [], [])
        self._update_ddac_zone_annotations()
        self._refresh_ddac_limits(state)
        if self.phase_visible_box.isChecked():
            self._render_phase_regions()


    def _compute_movie_metrics(self, cap, nb_frames, fps, t0_movie, time_axis, with_corr=False, nb_c=5):
        t_cam = []
        index_cam = []
        correlations = []
        gray_l = []

        if cap is None or nb_frames is None or fps is None or fps == 0:
            return {"t_cam": t_cam, "index_cam": index_cam, "correlations": correlations}

        for i in range(nb_frames):
            t_c = i / fps + t0_movie
            if time_axis[0] < t_c < time_axis[-1]:
                index_cam.append(i)
                t_cam.append(t_c)
                if with_corr:
                    correlation = 0
                    gray_curr = self.read_frame(cap, i, unit="gray")
                    for gray in gray_l:
                        correlation += cv2.matchTemplate(gray, gray_curr, cv2.TM_CCOEFF_NORMED)[0][0] - 1
                    correlations.append(correlation)
                    if len(gray_l) > nb_c:
                        del gray_l[0]
                    gray_l.append(gray_curr)

        return {"t_cam": t_cam, "index_cam": index_cam, "correlations": correlations}

    @pyqtSlot(object)
    def _on_movie_metrics_ready(self, payload):
        if not payload:
            return

        run_id = payload.get("run_id")
        state = self.runs.get(run_id)
        if state is None:
            return

        state.t_cam = payload.get("t_cam", [])
        state.index_cam = payload.get("index_cam", [])
        state.correlations = payload.get("correlations", [])
        c = payload.get("color", "w")

        corr_curve = self._get_or_create_curve(state.corr_curve, self.pg_dPdt, pen=pg.mkPen(c))
        state.corr_curve = corr_curve

        if self.chk_show_corr.isChecked() and state.correlations:
            state.correlations = list(np.array(state.correlations) / max(abs(np.array(state.correlations))))
            self._update_curve_safe(corr_curve, state.t_cam, state.correlations)
        else:
            self._update_curve_safe(corr_curve, [], [])

        self.current_index = len(state.t_cam) // 2 if state.t_cam else 0
        if state.index_cam:
            self.Num_im = state.index_cam[self.current_index]
            t = state.t_cam[self.current_index]
            Frame = self.read_frame(state.cap, self.Num_im)
            if Frame is not None:
                self.img_item.setImage(np.array(Frame), autoLevels=True)

        ready = bool(state.index_cam) and bool(state.t_cam) #and (state.cap is not None)
        self._set_movie_controls_enabled(ready)
        if ready:
            self.slider.setRange(0, len(state.index_cam)-1)
            self.slider.setValue(self.current_index)
        self._update_movie_frame()

    @pyqtSlot(object)
    def _on_cedd_loaded(self, payload):
        """Réception du chargement CEDd effectué en arrière-plan."""
        if not payload:
            return
        request_id = payload.get("request_id")
        active_request_id = getattr(self, "_active_cedd_load_request_id", None)
        if request_id is None or request_id != active_request_id:
            # Résultat obsolète (utilisateur a déjà demandé un autre chargement).
            return
        objet_run = payload.get("objet_run")
        index_file = payload.get("index_file")
        if objet_run is None or index_file is None:
            return

        run_id = self._get_run_id(objet_run)
        self.file_index_map[index_file] = run_id
        self.PRINT_CEDd(item=None, objet_run=objet_run)

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
                request_id = int(getattr(self, "_cedd_load_request_counter", 0)) + 1
                self._cedd_load_request_counter = request_id
                self._active_cedd_load_request_id = request_id
                self._submit_background_task(
                    lambda: {
                        "objet_run": CL.LOAD_CEDd(chemin_fichier),
                        "index_file": index_file,
                        "request_id": request_id,
                    },
                    result_slot=self._on_cedd_loaded,
                    description="Chargement CEDd…",
                )
                return
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

        # ---- choisir un index stable pour ce run
        # option simple : index = nombre de runs déjà chargés (stable tant que tu ne “réordonne” pas)
        color_idx = len(self.runs)

        state = RunViewState(ced=self.RUN, color_idx=color_idx)
        state.phase_tracks = copy.deepcopy(getattr(self.RUN, "phase_tracks", {}))
        state.active_phase_element = getattr(self.RUN, "active_phase_element", "H2O")
        c = self._get_run_color(state)


        name_select = name_select or CL.os.path.basename(self.RUN.CEDd_path)
        item_run = QListWidgetItem(name_select)
        item_run.setBackground(QColor(c))
        item_run.setForeground(QColor("#000000" if self.current_theme == "light" else "#ffffff"))
        item_run.setData(Qt.UserRole, run_id)
        self.liste_objets_widget.addItem(item_run)

        # Lecture des données CEDd
        l_P, l_sigma_P, l_lambda, l_fwhm, l_spe, l_T, l_sigma_T, Time, spectre_number, time_amp, amp, Gauges_RUN = self.Read_RUN(self.RUN)
        if len(l_P) > 0 and len(Time) == len(l_P[0]):
            self._ddac_pressure_ref_time = np.asarray(Time, dtype=float)
            self._ddac_pressure_ref_values = np.asarray(l_P[0], dtype=float)

        # Axes temps
        self.pg_P.setXRange(min(Time), max(Time), padding=0.01)
        self.pg_dPdt.setXRange(min(Time), max(Time), padding=0.01)
        self.pg_sigma.setXRange(min(Time), max(Time), padding=0.01)
        if getattr(self.RUN, "list_nspec", None) is not None and len(self.RUN.list_nspec) > 0:
            self.pg_dlambda.setXRange(min(self.RUN.list_nspec), max(self.RUN.list_nspec), padding=0.01)

        state.time = Time
        self._set_time_display_unit(Time)
        state.spectre_number = self.RUN.list_nspec
        state.list_item = item_run
        state.t_cam=self.RUN.time_movie
        if state.t_cam and len(state.t_cam) > 1:
            self.attach_camera_time(state.t_cam)

        # Courbes P / dPdt / sigma / T

        for i, G in enumerate(Gauges_RUN):
            if len(l_P[i]) >= 10:
                l_p_filtre = CL.savgol_filter(l_P[i], self.dpdt_range_entry.value(), 1)
            else:
                l_p_filtre =l_P[i]
            dps = [
                (l_p_filtre[x + 1] - l_p_filtre[x - 1]) / (Time[x + 1] - Time[x - 1]) * self._dpdt_scale_from_seconds
                for x in range(2, len(l_p_filtre) - 2)
            ]
            state.curves_P.append(
                self.pg_P.plot(Time, l_P[i], pen=pg.mkPen(c, width=2), symbol='d', symbolPen=pg.mkPen(G.color_print[0], width=2), symbolBrush=c, symbolSize=10)
            )   
            state.curves_dPdt.append(
                self.pg_dPdt.plot(Time[2:-2], dps, pen=pg.mkPen(c, width=2), symbol='d', symbolPen=pg.mkPen(G.color_print[0], width=2), symbolBrush=c, symbolSize=10)
            )
            state.curves_sigma.append(
                self.pg_sigma.plot(Time, l_fwhm[i], pen=pg.mkPen(c, width=2), symbol='d', symbolPen=pg.mkPen(G.color_print[0], width=2), symbolBrush=c, symbolSize=10)
            )
        if "RuSmT" in [x.name_spe for x in self.RUN.Gauges_init]:
            state.curves_T.append(
                self.pg_dPdt.plot(Time, l_T[-1], pen=pg.mkPen(c, width=2), symbol='d', symbolPen=pg.mkPen("darkred", width=2),symbolBrush=c, symbolSize=10)
            )
        else:
            state.curves_T.append(self.pg_dPdt.plot([], []))

        # Piezo
        if self.RUN.data_Oscillo is not None:
            state.piezo_curve = self.pg_P.plot(time_amp, amp, pen=pg.mkPen(c, width=3))
        else:
            state.piezo_curve = self.pg_P.plot([], [])

        for spe in l_spe:
            if spe is not []:
                x_dlambda = self._dlambda_x_values(state, spe)
                state.curves_dlambda.append(
                    self.pg_dlambda.plot(
                        x_dlambda,
                        spe[:len(x_dlambda)] if hasattr(spe, "__len__") else spe,
                        pen=pg.mkPen(c, width=3),
                        symbol='h',
                        symbolPen=pg.mkPen(c, width=3),
                        symbolSize=10,
                    )
                )

        # Film
        if self.RUN.folder_Movie is not None:
            t0_movie = getattr(self.RUN, 't0_movie', 0)
            cap, fps, nb_frames = self.Read_Movie(self.RUN)
            state.cap = cap
            state.t_cam = []
            state.index_cam = []
            state.correlations = []

            self._submit_background_task(
                lambda: {
                    **self._compute_movie_metrics(
                        cap=cap,
                        nb_frames=nb_frames,
                        fps=fps,
                        t0_movie=t0_movie,
                        time_axis=Time,
                        with_corr=self.chk_show_corr.isChecked(),
                    ),
                    "run_id": run_id,
                    "color": c,
                },
                result_slot=self._on_movie_metrics_ready,
                description="Calcul corrélation/trajectoire…",
            )

        else:
            self.Num_im = 0
            t = Time[0]
            state.corr_curve = self.pg_dPdt.plot([], [])

        # Enregistre l'état et finalise l'affichage
        self.runs[run_id] = state
        self._finalize_run_selection(state, name_select)
        self._refresh_ddac_limits(state)

    def _set_movie_controls_enabled(self, enabled: bool):
        self.slider.setEnabled(enabled)
        self.previous_button.setEnabled(enabled)
        self.play_stop_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
   

    def f_text_CEDd_print(self, t):
        """Met à jour le texte d'information dDAC dans le TextItem PyQtGraph."""
        t_spec = self.selected_spec_time if self.selected_spec_time is not None else 0.0

        t_spec_disp = t_spec * self._time_unit_scale
        t_frame_disp = t * self._time_unit_scale
        dt_disp = (t - t_spec) * self._time_unit_scale

        txt = (
            f"t_spec={t_spec_disp:.3f}{self._time_unit_symbol} n°Spec={self.x5:.2f}\n"
            f"t_frame={t_frame_disp:.3f}{self._time_unit_symbol} n°Frame={self.Num_im}\n"
            f"P={self.y1:.2f}GPa  T or dP/dt={self.y3:.2f} K or {self._dpdt_unit_symbol} t_clic={self.x_clic*self._time_unit_scale:.3f}{self._time_unit_symbol}"
            )
        if self._ddac_zone_summary_text:
            txt = f"{txt}\n- - - - - - - - -\n{self._ddac_zone_summary_text}"
        self.pg_text_label.setText(txt)

    def f_CEDd_update_print(self):
        """Met à jour texte + sélection de spectre en fonction de x_clic / t."""
        state = self._get_state_for_run()
        if self.RUN is None or state is None:
            return

        # temps courant pour le film (si t_cam dispo)
        if state.t_cam:
            t = state.t_cam[self.current_index]
        else:
            t = 0.0

        # spectre le plus proche en temps
        idx_spec = self._nearest_spectrum_index(state, self.x_clic)
        if idx_spec is None:
            self.f_text_CEDd_print(t)
            return
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

    def _nearest_index(self, values, target):
        """Retourne l'index du point temporel le plus proche de `target`."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return None
        if arr.size == 1:
            return 0

        diffs = np.diff(arr)
        if np.all(diffs >= 0):
            idx = int(np.searchsorted(arr, target, side="left"))
            if idx <= 0:
                return 0
            if idx >= arr.size:
                return arr.size - 1
            left = idx - 1
            return left if abs(target - arr[left]) <= abs(arr[idx] - target) else idx

        return int(np.argmin(np.abs(arr - target)))

    def _nearest_spectrum_index(self, state: RunViewState, target_time: float):
        """Index du spectre le plus proche basé uniquement sur `state.time`."""
        if state.time is None or state.spectre_number is None:
            return None
        bound = min(len(state.time), len(state.spectre_number))
        if bound <= 0:
            return None
        idx = self._nearest_index(state.time[:bound], target_time)
        if idx is None:
            return None
        return min(max(0, idx), bound - 1)

    def _nearest_movie_index(self, state: RunViewState, target_time: float):
        """Index movie le plus proche basé uniquement sur `state.t_cam`."""
        if not state.t_cam or not state.index_cam:
            return None
        bound = min(len(state.t_cam), len(state.index_cam))
        if bound <= 0:
            return None
        idx = self._nearest_index(state.t_cam[:bound], target_time)
        if idx is None:
            return None
        return min(max(0, idx), bound - 1)
    
    def _dlambda_x_values(self, state: Optional[RunViewState], y_values=None):
        """Axe X de Δλ: toujours basé sur les numéros de spectre."""
        if state is None or not state.spectre_number:
            if y_values is None:
                return []
            return list(range(len(y_values)))
        x_vals = list(state.spectre_number)
        if y_values is not None:
            n = min(len(x_vals), len(y_values))
            return x_vals[:n]
        return x_vals

    def _nearest_spectrum_number(self, state: Optional[RunViewState], target_time: float):
        idx = self._nearest_spectrum_index(state, target_time) if state is not None else None
        if idx is None or not state.spectre_number:
            return None
        return state.spectre_number[idx]

    def _time_from_spectrum_number(self, state: Optional[RunViewState], spectrum_number: float):
        if state is None or not state.spectre_number or not state.time:
            return None
        bound = min(len(state.spectre_number), len(state.time))
        if bound <= 0:
            return None
        spec = np.asarray(state.spectre_number[:bound], dtype=float)
        idx = self._nearest_index(spec, float(spectrum_number))
        if idx is None:
            return None
        return float(state.time[idx])
    

    def _get_ddac_target_from_pos(self, pos):
        """Détermine le ViewBox et le type de courbe visé par un clic PyQtGraph."""

        if self.pg_P.sceneBoundingRect().contains(pos):
            return self.pg_P.getViewBox(), "P"
        if self.pg_dPdt.sceneBoundingRect().contains(pos):
            return self.pg_dPdt.getViewBox(), "dPdt"
        if self.pg_sigma.sceneBoundingRect().contains(pos):
            return self.pg_sigma.getViewBox(), "sigma"
        if self.pg_dlambda.sceneBoundingRect().contains(pos):
            return self.pg_dlambda.getViewBox(), "dlambda"

        return None, None

    def _update_ddac_markers(self, which: str, x: float, y: float):
        """Positionne lignes/scatters en fonction de la courbe cliquée.

        Le clic définit un temps unique (lignes verticales) partagé entre
        P/dPdt/σ/Δλ, déclenche la mesure dP/dt (couple Pstart/Pend) et prépare
        la sélection de spectre/vidéo associée au temps choisi.
        """

        if which != "P" and getattr(self, "is_selecting_dp_range", False):
            # Séquence interrompue : on réinitialise pour éviter des états incohérents
            self._reset_dp_selection()
            if hasattr(self, "_report_warning"):
                self._report_warning("Sélection Pstart/Pend réinitialisée après un clic hors P.")

        state = self._get_state_for_run()
        x_time = x
        if which == "dlambda":
            x_time_from_spec = self._time_from_spectrum_number(state, x)
            if x_time_from_spec is not None:
                x_time = x_time_from_spec

        self.x_clic, self.y_clic = x_time, y
        if self.spectrum_select_box.isChecked():
            self.selected_spec_time = x_time
            self.line_t_spec_P.setPos(x_time)
            self.line_t_spec_sigma.setPos(x_time)
            self.line_t_spec_dPdt.setPos(x_time)
            nspec_from_time = self._nearest_spectrum_number(state, x_time)
            if nspec_from_time is not None:
                self.line_t_spec_dlambda.setPos(nspec_from_time)


        if which == "P":
            self.x1, self.y1 = x, y
            if hasattr(self, "scatter_P"):
                self._update_curve_safe(self.scatter_P, [x], [y])
            if not self.is_selecting_dp_range:
                self.Pstart, self.tstart, self.is_selecting_dp_range = self.y1, self.x1, True
            else:
                self.Pend, self.tend, self.is_selecting_dp_range = self.y1, self.x1, False
        elif which == "dPdt":
            self.x3, self.y3 = x, y
            if hasattr(self, "scatter_dPdt"):
                self._update_curve_safe(self.scatter_dPdt, [x], [y])
        elif which == "sigma":
            self.x5, self.y5 = x, y
            if hasattr(self, "scatter_sigma"):
                self._update_curve_safe(self.scatter_sigma, [x], [y])
        elif which == "dlambda":
            self.x6, self.y6 = x, y
            self.line_nspec.setPos(x)
            if hasattr(self, "scatter_dlambda"):
                self._update_curve_safe(self.scatter_dlambda, [x], [y])
        


    def _reset_dp_selection(self):
        """Réinitialise les variables de mesure dP/dt en cas d'usage inattendu."""

        self.is_selecting_dp_range = False
        self.Pstart = None
        self.Pend = None
        self.tstart = None
        self.tend = None

    def _sync_movie_after_click(self, state: Optional[RunViewState]):
        """Synchronise slider/frames après un clic si une vidéo est disponible."""

        if (
            not self.movie_select_box.isChecked()
            or self.RUN is None
            or state is None
            or not state.t_cam
        ):
            return

        idx = self._nearest_movie_index(state, self.x_clic)
        if idx is None:
            return
        self.current_index = idx
        self.Num_im=state.index_cam[self.current_index]
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)
        self._update_movie_frame()

    def _on_ddac_click(self, mouse_event):
        """Gère les clics sur les graphes temporels dDAC et synchronise l'état UI.

        Le clic est transformé des coordonnées scène vers le ViewBox ciblé (P,
        dP/dt, σ ou Δλ) pour extraire (t, valeur). Les lignes verticales, les
        marqueurs scatter et les textes sont mis à jour, puis la sélection de
        spectre/film est recalculée en fonction du temps cliqué.
        """
        self.setFocus()
        pos = mouse_event.scenePos()
        vb, which = self._get_ddac_target_from_pos(pos)
        if vb is None or which is None:
            return

        mouse_point = vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        if (
            self._last_ddac_click_target == which
            and self._last_ddac_click_time is not None
            and abs(self._last_ddac_click_time - x) <= 1e-12
        ):
            return
        self._last_ddac_click_target = which
        self._last_ddac_click_time = x

        self._update_ddac_markers(which, x, y)
        if which == "P":
            self._select_phase_at_time(x)

        state = self._get_state_for_run()
        self._sync_movie_after_click(state)
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
        if state is None or len(state.index_cam) == 0:
            return

        idx_list = state.index_cam
        if self.current_index < 0 or self.current_index >= len(idx_list):
            return

        frame_idx = idx_list[self.current_index]
        t = state.t_cam[self.current_index]
        self.selected_frame_time = t

        frame = self.read_frame(state.cap, frame_idx)
        if frame is None:
            return

        # frame : np.array 2D ou 3D
        self.img_item.setImage(frame, autoLevels=True)  # ou sans .T suivant orientation

        # Lignes verticales
        self.line_t_P.setPos(t)
        self.line_t_dPdt.setPos(t)
        self.line_t_sigma.setPos(t)
        nspec = self._nearest_spectrum_number(state, t)
        if nspec is not None:
            self.line_t_dlambda.setPos(nspec)
            self.line_nspec.setPos(nspec)

        # Texte (inclut aussi le buffer de zone dP/dt si actif)
        self.Num_im = frame_idx
        self.f_text_CEDd_print(t)

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
        if self.zone_movie[0] is None or self.zone_movie[1] is None or not self._cam_visible:
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
