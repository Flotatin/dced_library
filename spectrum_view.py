import copy
import random
from typing import Any, Callable, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QTableWidgetItem, QVBoxLayout, QWidget

from Bibli_python import CL_FD_Update as CL


class SciAxis(pg.AxisItem):
    """Axe qui affiche les ticks en notation scientifique (1.23e+04)."""

    def tickStrings(self, values, scale, spacing):
        # values est la liste de positions de ticks en coordonnées "données"
        # On ne touche PAS aux données, on ne fait que formatter l'affichage.
        return [f"{v:.1e}" for v in values]


class SpectrumViewMixin:
    def _add_spec_plot(self, *, row: int, col: int, x_label: Optional[str] = None, y_label: Optional[str] = None, show_grid: bool = True, **kwargs):
        plot_item = self.pg_spec.addPlot(row=row, col=col, **kwargs)
        self._stylize_plot(plot_item, x_label=x_label, y_label=y_label, show_grid=show_grid)
        return plot_item

    def _setup_spectrum_box(self):
        self._create_spectrum_plots()
        self._create_persistent_curves()
        self._create_selection_items()
        self._init_spectrum_state()
        self._build_spectrum_controls()
        self._connect_spectrum_events()
        self._configure_spectrum_layout()

    def _create_spectrum_plots(self) -> None:
        self.SpectraBox = QGroupBox("Spectrum")
        SpectraBoxFirstLayout = QVBoxLayout()

        self._spectra_layout = SpectraBoxFirstLayout

        # Conteneur principal : colonne gauche (plots) + colonne droite (infos jauge + zoom)
        self.spectrum_container = QWidget()
        container_layout = QVBoxLayout(self.spectrum_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(6)

        # ================== WIDGET PyQtGraph (colonne gauche) ==================
        self.pg_spec = pg.GraphicsLayoutWidget()

        # Plot principal (au centre)
        y_axis_sci = SciAxis(orientation='left')
        self.pg_spectrum = self._add_spec_plot(
            row=0,
            col=0,
            axisItems={'left': y_axis_sci},
            x_label='X (U.A)',
            y_label='Y (U.A)',
        )
        self.pg_spectrum.setTitle("Spectrum main")
        self.pg_spectrum.hideAxis('bottom')

        # Résidus / dY (en haut)
        self.pg_dy = self._add_spec_plot(row=1, col=0)
        self.pg_dy.setTitle("Residuals / dY")
        self.pg_dy.setXLink(self.pg_spectrum)
        container_layout.addWidget(self.pg_spec, stretch=3)

        # ================== Colonne droite : AddBox + Zoom ==================
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        self._spectrum_right_layout = right_layout

        zoom_index = 0
        if hasattr(self, "AddBox"):
            right_layout.addWidget(self.AddBox)
            zoom_index = 1

        self.pg_zoom = pg.PlotWidget()
        self._stylize_plot(self.pg_zoom, show_grid=False)
        self.pg_zoom.hideAxis('bottom')
        self.pg_zoom.hideAxis('left')
        right_layout.addWidget(self.pg_zoom)
        right_layout.setStretch(zoom_index, 1)

        container_layout.addLayout(right_layout, stretch=2)

        # ================== FFT (en dessous du conteneur principal) ==================
        self.pg_fft = pg.PlotWidget()
        self._stylize_plot(self.pg_fft, x_label='f', y_label='|F|')

        self._spectra_layout.addWidget(self.spectrum_container)
        #self._spectra_layout.addWidget(self.pg_fft)

    def _create_persistent_curves(self) -> None:
        # ================== COURBES PERSISTANTES ==================
        # Spectre corrigé
        self.curve_spec_data = self.pg_spectrum.plot()  # Spectrum.y_corr

        # Fit total (somme des pics)
        self.curve_spec_fit = self.pg_spectrum.plot()

        # Filtres, baseline etc.
        self.curve_baseline_brut = self.pg_spectrum.plot()
        self.curve_baseline_blfit = self.pg_spectrum.plot()

        # dY
        self.curve_dy = self.pg_dy.plot()
        self.line_dy_zero = self.pg_dy.addLine(y=0)

        # Zoom : zone + pic sélectionné + spectre brut / corrigé
        self.curve_zoom_data = self.pg_zoom.plot()
        self.curve_zoom_pic = self.pg_zoom.plot(pen=None, fillLevel=0)
        self.curve_zoom_data_brut = self.pg_zoom.plot(pen='y')

        # Sur le spectre principal : pic sélectionné
        self.curve_spec_pic_select = self.pg_spectrum.plot(pen=None, fillLevel=0)

    def _create_selection_items(self) -> None:
        # Sélections verticales/horizontales
        self.vline_spec = self._create_marker_line(angle=90)
        self.hline_spec = self._create_marker_line(angle=0)
        self.pg_spectrum.addItem(self.vline_spec)
        self.pg_spectrum.addItem(self.hline_spec)

        self.vline_dy = self._create_marker_line(angle=90)
        self.pg_dy.addItem(self.vline_dy)


        self.vline_zoom = self._create_marker_line(angle=90)
        self.pg_zoom.addItem(self.vline_zoom)

        # Pour le zoom : on met une petite croix
        self.cross_zoom = self._create_scatter_marker(symbol='+')
        self.pg_zoom.addItem(self.cross_zoom)

        # Zones d'exclusion (fit window)
        self.zoom_exclusion_left = pg.LinearRegionItem(values=(0, 0.5), movable=False, brush=pg.mkBrush(255, 0, 0, 40))
        self.zoom_exclusion_right = pg.LinearRegionItem(values=(0.5, 1), movable=False, brush=pg.mkBrush(255, 0, 0, 40))
        self.pg_zoom.addItem(self.zoom_exclusion_left)
        self.pg_zoom.addItem(self.zoom_exclusion_right)
        self.zoom_exclusion_left.setZValue(-10)
        self.zoom_exclusion_right.setZValue(-10)
        self.zoom_exclusion_left.setVisible(False)
        self.zoom_exclusion_right.setVisible(False)

    def _init_spectrum_state(self) -> None:
        """Initialise l'état interne utilisé par les vues spectrales."""

        self._spectrum_limits_initialized = False  # une seule initialisation XRange
        self._zoom_limits_initialized = False

        self.select_clic_box = None

        # index pour les jauges / pics
        self.index_jauge: int = -1
        """Index de la jauge sélectionnée dans le RUN courant (-1 si aucune)."""
        self.index_pic_select: int = -1
        """Index du pic sélectionné dans le spectre visible (-1 si aucun)."""
        self.index_spec: int = 0
        """Numéro de spectre actuellement chargé dans l'onglet Spectrum."""

        self.lines = []
        self.X0 = 0
        self.Y0 = 0
        self.Zone_fit = []
        self.X_s = []
        self.X_e = []

        self.y_fit_start = None
        self.model_pic_fit = None

        self.bit_fit_T: bool = False
        """Indique si un fit de température est actif (zones verrouillées)."""
        self.bit_print_fit_T: bool = False
        """Autorise l'affichage immédiat des résultats de fit température."""
        self.bit_modif_jauge: bool = False
        """Bloque les mises à jour quand une jauge est en cours de modification."""
        self.is_loading_gauge: bool = False
        """Vrai pendant le chargement d'une jauge pour éviter les callbacks."""
        self.bit_filtre: bool = False
        """Indique qu'un filtre de spectre est actif pour le tracé courant."""

        self._refreshing_pic_table: bool = False

    def _build_spectrum_controls(self) -> None:
        # ================== INTÉGRATION UI ==================
        layout_check = QHBoxLayout()
        self.select_clic_box = QCheckBox("Select clic pic (q)", self)
        self.select_clic_box.setChecked(True)
        layout_check.addWidget(self.select_clic_box)

        self.zone_spectrum_box = QCheckBox("Zone Fit Spectrum (Z)", self)
        self.zone_spectrum_box.setChecked(True)
        layout_check.addWidget(self.zone_spectrum_box)

        self.Gauge_init_box = QCheckBox("Gauge_init (0)", self)
        self.Gauge_init_box.setChecked(False)
        layout_check.addWidget(self.Gauge_init_box)

        self.vslmfit = QCheckBox("vslmfit", self)
        self.vslmfit.setChecked(False)
        layout_check.addWidget(self.vslmfit)

        self._spectra_layout.addLayout(layout_check)
        self.SpectraBox.setLayout(self._spectra_layout)
        self.grid_layout.addWidget(self.SpectraBox, 0, 2, 3, 1)

    def _connect_spectrum_events(self) -> None:
        # ================== EVENTS PyQtGraph ==================
        # clic sur le spectre principal
        self.pg_spectrum.scene().sigMouseClicked.connect(self._on_pg_spectrum_click)
        self.pg_zoom.scene().sigMouseClicked.connect(self._on_pg_spectrum_click)
        self.pg_dy.scene().sigMouseClicked.connect(self._on_pg_spectrum_click)
        # tu peux aussi connecter sigMouseMoved si tu veux un "hover" au lieu de clic

    def _configure_spectrum_layout(self) -> None:
        # ================== FACTEURS LIGNES / COLONNES ==================
        # 3 lignes empilées (Residuals / Spectrum / Baseline preview)
        self._spec_row_factors = (5, 1)
        # 1 colonne dans le GraphicsLayoutWidget
        self._spec_col_factors = (1,)

        # premier ajustement immédiat
        self._update_graphicslayout_sizes(
            self.pg_spec,
            row_factors=self._spec_row_factors,
            col_factors=self._spec_col_factors,
        )

    def _is_print_baseline_enabled(self) -> bool:
        cb = getattr(self, "baseline_preview_checkbox", None)
        if cb is None:
            return False
        return bool(cb.isChecked())

    def _on_print_baseline_toggled(self, state: int):
        # state = Qt.Checked / Qt.Unchecked
        self._refresh_spectrum_view()

    # -----------------------------
    # Helpers de rafraîchissement
    # -----------------------------
    def _curve_is_empty(self, curve: pg.PlotDataItem) -> bool:
        x_old, y_old = curve.getData()
        if x_old is None or y_old is None:
            return True
        return len(x_old) == 0 and len(y_old) == 0

    def _update_curve_safe(self, curve: pg.PlotDataItem, x_data, y_data) -> bool:
        """Pousse les nouvelles données sur une courbe uniquement si elles changent.

        Les tableaux sont convertis en vues NumPy (pas de copie) pour limiter le
        coût mémoire. En cas d'entrée vide, rien n'est envoyé si la courbe est déjà
        vide. La fonction retourne ``True`` si un ``setData`` a effectivement été
        déclenché.
        """

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

    def _run_spectrum_task(
        self,
        task: Callable[[], Any],
        description: str,
        result_slot: Callable[[Any], None] | None = None,
    ) -> None:
        """Exécute une tâche de traitement spectrale dans le pool Qt si possible."""

        if not hasattr(self, "_submit_background_task"):
            result = task()
            if result_slot is not None:
                result_slot(result)
            return

        self._submit_background_task(
            task,
            result_slot=result_slot,
            description=description,
        )

    def _update_fit_window(self, indexX = None) -> None:
        """Update the excluded zoom regions according to the current fit window."""

        left_region = getattr(self, "zoom_exclusion_left", None)
        right_region = getattr(self, "zoom_exclusion_right", None)

        if left_region is None or right_region is None:
            return

        def _hide_regions() -> None:
            left_region.setVisible(False)
            right_region.setVisible(False)

        gauge_index = getattr(self, "index_jauge", None)

        if (
            self.Spectrum is None
            or self.index_pic_select is None
            or self.index_pic_select < 0
            or gauge_index is None
            or gauge_index < 0
        ):
            _hide_regions()
            return


        try:
            params = self.Param0[gauge_index][self.index_pic_select]
        except (IndexError, TypeError, AttributeError):
            _hide_regions()
            return

        if not params:
            _hide_regions()
            return

        n_sigma_widget = getattr(self, "sigma_pic_fit_entry", None)
        if n_sigma_widget is None:
            _hide_regions()
            return

        center = float(params[0])
        sigma = float(params[2])
        n_sigma = float(n_sigma_widget.value())

        if not np.isfinite(center) or not np.isfinite(sigma) or not np.isfinite(n_sigma):
            _hide_regions()
            return

        spectrum_x = getattr(self.Spectrum, "wnb", None)
        if spectrum_x is None:
            _hide_regions()
            return

        x_data = np.asarray(spectrum_x, dtype=float)
        if x_data.size == 0:
            _hide_regions()
            return

        window_left = center - n_sigma * sigma
        window_right = center + n_sigma * sigma
        fit_left = float(min(window_left, window_right))
        fit_right = float(max(window_left, window_right))

        if indexX is None:
            mask = (x_data >= fit_left) & (x_data <= fit_right)
            index_array = np.nonzero(mask)[0]
        else:
            index_array = np.asarray(indexX)

        if index_array.size == 0:
            _hide_regions()
            return

        x_min = float(np.min(x_data))
        x_max = float(np.max(x_data))

        left_region_end = min(max(fit_left, x_min), x_max)
        right_region_start = max(min(fit_right, x_max), x_min)

        if left_region_end > x_min:
            left_region.setRegion((x_min, left_region_end))
            left_region.setVisible(True)
        else:
            left_region.setVisible(False)

        if right_region_start < x_max:
            right_region.setRegion((right_region_start, x_max))
            right_region.setVisible(True)
        else:
            right_region.setVisible(False)

    def _on_pg_spectrum_click(self, mouse_event):
        """Gère le clic gauche sur les vues spectrales et met à jour les marqueurs.

        Le point cliqué est d'abord converti des coordonnées scène vers le ViewBox
        actif (spectre principal ou zoom) afin de récupérer (x, y). Ces valeurs
        déplacent les lignes infinies, la croix de zoom et peuvent déclencher la
        sélection d'un pic lorsque la case « clic = select pic » est cochée.
        """
        if mouse_event.button() != Qt.LeftButton:
            return

        pos = mouse_event.scenePos()

        # --- déterminer sur quel plot on a cliqué ---
        clicked_on_spec = self.pg_spectrum.sceneBoundingRect().contains(pos)
        clicked_on_zoom = self.pg_zoom.sceneBoundingRect().contains(pos)
        clicked_on_dy = self.pg_dy.sceneBoundingRect().contains(pos)


        if not (clicked_on_spec or clicked_on_zoom or clicked_on_dy):
            return

        if clicked_on_zoom:
            vb = self.pg_zoom.getViewBox()
        elif clicked_on_spec:
            vb = self.pg_spectrum.getViewBox()
        elif clicked_on_dy:
            vb = self.pg_dy.getViewBox()


        mouse_point = vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        # Conversion scène -> axes du plot : x = position spectrale, y = intensité
        self.X0, self.Y0 = x, y
        self.vline_spec.setPos(x)
        self.vline_dy.setPos(x)
        self.vline_zoom.setPos(x)
        if clicked_on_spec or clicked_on_zoom:
            self.hline_spec.setPos(y)
        self._update_curve_safe(self.cross_zoom, [x], [y])

        # Si tu veux garder la logique "clic = sélectionner/placer un pic"
        # → seulement quand on clique dans le spectre principal
        if self.select_clic_box.isChecked() and clicked_on_spec:
            try:
                self._select_nearest_pic_from_x(x)
            except Exception as e:
                print("Error in _select_nearest_pic_from_x:", e)

    def _select_nearest_pic_from_x(self, x):
        if self.Spectrum is None or self.Param0 is None:
            return

        centers = []
        indices = []

        for i, Ljp in enumerate(self.Param0):
            for j, Jp in enumerate(Ljp):
                try:
                    centers.append(float(Jp[0]))
                    indices.append((i, j))
                except Exception:
                    pass

        if not centers:
            return

        centers = np.array(centers)
        k = np.argmin(np.abs(centers - x))
        best_i, best_j = indices[k]
        best_dx = abs(centers[k] - x)
        # tolérance
        wnb = getattr(self.Spectrum, "wnb", None)
        if wnb is not None and len(wnb) > 1:
            max_dx = (wnb[-1] - wnb[0]) / 10.0
            if best_dx > max_dx:
                return

        # changement jauge
        if best_i != self.index_jauge:

            self.index_jauge = best_i

            if 0 <= best_i < len(self.list_name_gauges):
                gauge_name = self.list_name_gauges[best_i]
                if gauge_name in self.liste_type_Gauge:
                    self.Gauge_type_selector.setCurrentIndex(
                        self.liste_type_Gauge.index(gauge_name)
                    )

            self.LOAD_Gauge()
        if best_j != self.index_pic_select:
            # sélection pic
            self.index_pic_select = best_j
            self.bit_bypass = True
            try:
                self.select_pic()
            finally:
                self.bit_bypass = False
    
    def _refresh_spectrum_view(self):
        """Rafraîchit l'ensemble des vues spectrales PyQtGraph à partir du spectre courant.

        Les courbes de spectre, fit global, dérivée dY, baseline et FFT sont mises
        à jour sur leurs plots respectifs (pg_spectrum, pg_dy, pg_baseline_preview, pg_fft).
        La fonction ajuste aussi les limites de ViewBox et met à jour l'état local
        (drapeau d'initialisation des limites) sans modifier la logique métier du
        calcul du spectre.
        """
        S = self.Spectrum
        if S is None:
            # on vide tout
            self._update_curve_safe(self.curve_spec_data, [], [])
            self._update_curve_safe(self.curve_spec_fit, [], [])
            self._update_curve_safe(self.curve_dy, [], [])
            self._update_curve_safe(self.curve_baseline_brut, [], [])
            self._update_curve_safe(self.curve_baseline_blfit, [], [])
            self._update_curve_safe(self.curve_zoom_data, [], [])
            self._update_curve_safe(self.curve_zoom_data_brut, [], [])
            self._update_curve_safe(self.curve_zoom_pic, [], [])
            self._update_curve_safe(self.curve_spec_pic_select, [], [])
            self._update_curve_safe(self.curve_spec_brut, [], [])
            self._update_curve_safe(self.curve_spec_blfit, [], [])
            return

        # --- On mémorise les X/Y du spectre principal pour les autres graphes ---
        x_spec = None
        y_spec = None

        # 1) Spectre corrigé
        if hasattr(S, "x_corr") and hasattr(S, "y_corr") and S.x_corr is not None and S.y_corr is not None:
            x = np.asarray(S.x_corr)
            y = np.asarray(S.y_corr)
            self._update_curve_safe(self.curve_spec_data, x, y)

            x_spec = x
            y_spec = y

            vb_spec = self.pg_spectrum.getViewBox()

            # Contraindre les bornes de pan/zoom au spectre
            self._set_viewbox_limits_from_data(vb_spec, x, y, padding=0.02)

            # Centrer la vue sur les données seulement une fois
            if not self._spectrum_limits_initialized:
                try:
                    vb_spec.setXRange(float(x[0]), float(x[-1]), padding=0.02)
                except Exception as e:
                    print("XRange error:", e)
                self._spectrum_limits_initialized = True
        else:
            self._update_curve_safe(self.curve_spec_data, [], [])
            # si pas de x_corr, on peut prendre wnb comme X de référence
            if hasattr(S, "wnb") and S.wnb is not None and hasattr(S, "spec") and S.spec is not None:
                x_spec = np.asarray(S.wnb)
                y_spec = np.asarray(S.spec)

        # 2) Fit total (self.y_fit_start)
        if self.y_fit_start is not None and hasattr(S, "wnb"):
            self._update_curve_safe(self.curve_spec_fit, S.wnb, self.y_fit_start)
        else:
            self._update_curve_safe(self.curve_spec_fit, [], [])
        
        # 2 bis) Mise à jour du fond avec *tous* les pics
        self._update_gauge_peaks_background()

        # 3) dY : X = ceux du spectre, Y = dY
        if hasattr(S, "dY") and S.dY is not None:
            x_dy = S.wnb[S.indexX] if S.indexX is not None else S.wnb
            self._update_curve_safe(self.curve_dy, x_dy, S.dY)
        else:
            self._update_curve_safe(self.curve_dy, [], [])

        # 4) ---- Overlay baseline/brut sur le spectre principal (si activé) ----
        if self._is_print_baseline_enabled():
            if hasattr(S, "wnb") and S.wnb is not None and hasattr(S, "spec") and S.spec is not None:
                self._update_curve_safe(self.curve_spec_brut, S.wnb, S.spec)
            else:
                self._update_curve_safe(self.curve_spec_brut, [], [])

            if hasattr(S, "wnb") and S.wnb is not None and hasattr(S, "blfit") and S.blfit is not None:
                if getattr(S, "indexX", None) is not None:
                    self._update_curve_safe(self.curve_spec_blfit, S.wnb[S.indexX], S.blfit[S.indexX])
                else:
                    self._update_curve_safe(self.curve_spec_blfit, S.wnb, S.blfit)
            else:
                self._update_curve_safe(self.curve_spec_blfit, [], [])
        else:
            self._update_curve_safe(self.curve_spec_brut, [], [])
            self._update_curve_safe(self.curve_spec_blfit, [], [])


       
        # 6) Limites du graphe dY :
        # X : mêmes que le spectre (x_spec)
        # Y : dY
        if x_spec is not None and hasattr(S, "dY") and S.dY is not None:
            vb_dy = self.pg_dy.getViewBox()
            # même astuce : x_data = x_spec, y_data = dY
            self._set_viewbox_limits_from_data(vb_dy, x_spec, S.dY, padding=0.02)
            x_spec_array = np.asarray(x_spec)
            if x_spec_array.size > 0:
                self.pg_dy.setLimits(xMin=float(x_spec_array[0]), xMax=float(x_spec_array[-1]))

    def _set_viewbox_limits_from_data(self, vb, x_data, y_data=None, padding=0.02):
        """Contraint les limites d'un ViewBox en fonction des données visibles.

        Le padding ajoute une marge relative autour des bornes min/max détectées
        pour éviter de coller les courbes aux bords. Aucun calcul métier n'est
        modifié : seules les limites de navigation (pan/zoom) sont recalculées.
        """

        if x_data is None:
            return

        x = np.asarray(x_data)
        if x.size == 0:
            return

        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))

        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            return

        span_x = x_max - x_min
        x0 = x_min - span_x * padding
        x1 = x_max + span_x * padding

        if y_data is not None:
            y = np.asarray(y_data)
            if y.size > 0:
                y_min = float(np.nanmin(y))
                y_max = float(np.nanmax(y))
                if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
                    span_y = y_max - y_min
                    y0 = y_min - span_y * padding
                    y1 = y_max + span_y * padding
                else:
                    y0 = y_min
                    y1 = y_max
            else:
                y0 = None
                y1 = None
        else:
            y0 = y1 = None

        if y0 is not None and y1 is not None:
            vb.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1)
        else:
            vb.setLimits(xMin=x0, xMax=x1)

    def _update_gauge_peaks_background(self):
        """Dessine en fond du spectre la contribution de chaque pic de jauge.

        Les zones remplies utilisent les couleurs définies sur chaque jauge
        (G.color_print) et sont forcées derrière les courbes principales afin de
        faciliter la lecture. Aucune donnée n'est modifiée, seul l'habillage
        graphique du `pg_spectrum` est mis à jour.
        """

        if hasattr(self, "gauge_peak_items"):
            for it in self.gauge_peak_items:
                try:
                    self.pg_spectrum.removeItem(it)
                except Exception:
                    pass
        else:
            self.gauge_peak_items = []

        S = self.Spectrum
        if S is None:
            return

        if not hasattr(S, "wnb") or S.wnb is None:
            return
        x = np.asarray(S.wnb, dtype=float)
        if x.size == 0:
            return

        if not self.list_y_fit_start or not S.Gauges:
            return

        new_items = []

        for j, G in enumerate(S.Gauges):
            colors_for_peaks = None
            if hasattr(G, "color_print") and isinstance(G.color_print, (list, tuple)) and len(G.color_print) > 1:
                colors_for_peaks = G.color_print[1]

            if j >= len(self.list_y_fit_start):
                continue
            list_peaks_j = self.list_y_fit_start[j]
            if not list_peaks_j:
                continue

            for i, y_pic in enumerate(list_peaks_j):
                if y_pic is None:
                    continue

                y_pic = np.asarray(y_pic, dtype=float)
                if y_pic.size != x.size:
                    continue

                if colors_for_peaks and i < len(colors_for_peaks):
                    col = colors_for_peaks[i]
                    brush = pg.mkBrush(col)
                    c = brush.color()
                    c.setAlpha(80)
                    brush = c
                else:
                    brush = (200, 200, 50, 60)

                item = self.pg_spectrum.plot(
                    x,
                    y_pic,
                    pen=None,
                    fillLevel=0,
                    brush=brush,
                )
                item.setZValue(-5)
                new_items.append(item)

        self.gauge_peak_items = new_items

    def Baseline_spectrum(self):
        param = [float(self.param_filtre_1_entry.text()), float(self.param_filtre_2_entry.text())]
        if self.filtre_type_selector.currentText() == "svg":
            param[0], param[1] = int(param[0]), int(param[1])

        def _process_baseline():
            self.Spectrum.Data_treatement(
                deg_baseline=int(self.deg_baseline_entry.value()),
                type_filtre=self.filtre_type_selector.currentText(),
                param_f=param,
                print_data=False,  # plus besoin de passer ax=ax_baseline, ax2=ax_fft
            )
            return True

        self._run_spectrum_task(
            _process_baseline,
            description="Calcul baseline/FFT…",
            result_slot=lambda _=None: self._on_baseline_ready(),
        )

    def _on_baseline_ready(self):
        """Callback UI exécuté après le recalcul de baseline en tâche de fond."""

        self._recompute_y_fit_start()   # somme des pics
        self._refresh_spectrum_view()

    def Auto_pic_fit(self):
        """Auto-pic fit : boucle sur tous les pics de toutes les jauges, sans Matplotlib."""
        save_jauge = self.index_jauge
        save_pic = self.index_pic_select

        for _ in range(self.spinbox_cycle.value()):
            indices = [
                (i, j)
                for i in range(len(self.list_y_fit_start))
                for j in range(len(self.list_y_fit_start[i]))
            ]
            sample_indices = random.sample(indices, len(indices))

            for Indx in sample_indices:
                self.index_jauge, self.index_pic_select = Indx
                out, X_pic, _, bit = self.f_pic_fit()
                if bit:
                    y_plot = out.best_fit
                else:
                    p = out.make_params()
                    y_plot = X_pic.model.eval(p, x=self.Spectrum.wnb)

                self.list_y_fit_start[self.index_jauge][self.index_pic_select] = y_plot
                self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select] = X_pic
                new_name = (
                    self.Nom_pic[self.index_jauge][self.index_pic_select]
                    + "   X0:"
                    + str(self.Param0[self.index_jauge][self.index_pic_select][0])
                    + "   Y0:"
                    + str(self.Param0[self.index_jauge][self.index_pic_select][1])
                    + "   sigma:"
                    + str(self.Param0[self.index_jauge][self.index_pic_select][2])
                    + "   Coef:"
                    + str(self.Param0[self.index_jauge][self.index_pic_select][3])
                    + " ; Modele:"
                    + str(self.Param0[self.index_jauge][self.index_pic_select][4])
                )
                self.list_text_pic[self.index_jauge][self.index_pic_select] = str(new_name)

        self.index_jauge = save_jauge
        self.index_pic_select = save_pic
        self.LOAD_Gauge()
        self.Print_fit_start()
    
    def Update_var(self,name=None):
        self.list_name_gauges.append(name)
        self.Nom_pic.append([])
        self.Spec_fit.append(None)
        self.Param0.append([])
        self.Param_FIT.append([])
        self.J.append(0) #Compteur du nombre d epic selectioné
        self.X0=0 #cordonnée X du clique
        self.Y0=0 #cordonnée Y du clique [None)
        self.X_s.append(None) #X de départ de donné selectioné
        self.X_e.append(None) #X de fin de ddonné selectionéinter_entryself.
        self.z1.append(None) #afficahge de la zone enlver au debut
        self.z2.append(None) #afficahge de la zone enlver a la fin
        self.Zone_fit.append(None)
        self.bit_plot_fit.append(False) # True les fit son affiché sinon False
        self.plot_fit.append(None)
        self.plot_pic.append([]) #plot des fit
        self.list_text_pic.append([]) # texte qui donne les different pic enregistré
        self.bit_fit.append(True)
        self.plot_pic_fit.append([])
        self.list_y_fit_start.append([])

    def Auto_pic(self): # - - - AUTO PIC- - -#
        #ALLER VORIE DANS FD_DRX POUR REFAIRE
        if self.index_jauge==-1:
            return print("Auto_pic bug resolve : if self.index_jauge==-1")
        if self.Spectrum.Gauges[self.index_jauge].lamb_fit is not None:
            x0=self.Spectrum.Gauges[self.index_jauge].lamb_fit
        else:
            x0=self.Spectrum.Gauges[self.index_jauge].lamb0
        for i,p in enumerate(self.Spectrum.Gauges[self.index_jauge].pics):
            self.X0=x0+self.Spectrum.Gauges[self.index_jauge].deltaP0i[i][0]
            m=np.argmin(np.abs(self.Spectrum.x_corr -self.X0 ))
            self.Y0=round(self.Spectrum.y_corr[m],3)
            self.Nom_pic[self.index_jauge].insert(i,self.list_name_gauges[self.index_jauge] +'_p'+str(i+1))
            self.Param0[self.index_jauge].insert(i,i)
            self.list_text_pic[self.index_jauge].insert(i,i)
            self.list_y_fit_start[self.index_jauge].insert(i,i)
            self.plot_pic_fit[self.index_jauge].insert(i,"pic_fit")
            self.listbox_pic.insertItem(self.J[self.index_jauge]-1,str(i))
            self.J[self.index_jauge]+=1
            self.index_pic_select=i
            self.Param0[self.index_jauge][self.index_pic_select]=[self.X0,self.Y0,float(p.sigma[0]),np.array([float(x[0]) for x in p.coef_spe]),str(p.model_fit) ]
            self.bit_bypass=True
            self.Replace_pic()
            self.bit_bypass=False

        self.Spectrum.bit_fit=False
        self.Spectrum.Gauges[self.index_jauge].bit_fit =False
        self.select_pic()

    def Click_Confirme(self): # Fonction qui confirme le choix du pic et qui passe au suivant
        if  "DRX" in (self.Spectrum.Gauges[self.index_jauge].spe or self.Spectrum.Gauges[self.index_jauge].name_spe): #a travailler !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            n=1
            while n <= self.J[self.index_jauge] and n < self.Spectrum.Gauges[self.index_jauge].nb_pic :
                n+=1
            self.Nom_pic[self.index_jauge].append(self.list_name_gauges[self.index_jauge] +'_'+self.Spectrum.Gauges[self.index_jauge].Element_ref.name_dhkl[i]+'_')
        else:
            self.Nom_pic[self.index_jauge].append(self.list_name_gauges[self.index_jauge] +'_p'+str(self.J[self.index_jauge])+'_')

        self.Param0[self.index_jauge].append([self.X0,self.Y0,float(self.spinbox_sigma.value()),np.array([float(spin.value()) for spin in self.coef_dynamic_spinbox]),str(self.model_pic_fit)])
        new_name= str(self.Nom_pic[self.index_jauge][-1]) + "   X0:"+str(self.Param0[self.index_jauge][-1][0])+"   Y0:"+ str(self.Param0[self.index_jauge][-1][1]) + "   sigma:" + str(self.Param0[self.index_jauge][-1][2]) + "   Coef:" + str(self.Param0[self.index_jauge][-1][3]) +" ; Modele:" + str(self.Param0[self.index_jauge][-1][4])
        self.J[self.index_jauge]+=1
        self.text_box_msg.setText('Parametre initiale pic'+str(self.J[self.index_jauge])+': \n VIDE')
        self.list_text_pic[self.index_jauge].append(str(new_name))
        self.listbox_pic.insertItem(self.J[self.index_jauge]-1,new_name)

        X_pic=CL.Pics(name=self.Nom_pic[self.index_jauge][-1],ctr=self.Param0[self.index_jauge][-1][0],ampH=self.Param0[self.index_jauge][-1][1],coef_spe=self.Param0[self.index_jauge][-1][3],sigma=self.Param0[self.index_jauge][-1][2],model_fit=self.Param0[self.index_jauge][-1][4])
        params=X_pic.model.make_params()
        y_plot=X_pic.model.eval(params,x=self.Spectrum.wnb)#+self.Spectrum.blfit

        self.list_y_fit_start[self.index_jauge].append(y_plot)

        self.Print_fit_start()

        self.plot_pic_fit[self.index_jauge].append(
            self.pg_spectrum.plot(
                self.Spectrum.wnb, y_plot,
                pen=None,
                fillLevel=float(np.nanmin(y_plot)),
                brush=pg.mkBrush(self.Spectrum.Gauges[self.index_jauge].color_print[0])
            )
        )
        self.Spectrum.Gauges[self.index_jauge].pics.append(X_pic)
        self._refresh_pic_table()

    def _recompute_y_fit_start(self):
        """Recale self.y_fit_start à partir de list_y_fit_start."""
        self.y_fit_start = None
        if not any(self.list_y_fit_start):
            return

        for i, l in enumerate(self.list_y_fit_start):
            for j, y in enumerate(l):
                if self.y_fit_start is None:
                    self.y_fit_start = y.copy()
                else:
                    self.y_fit_start = self.y_fit_start + y

    def _refresh_pic_table(self):
        table = getattr(self, "pic_table", None)
        if table is None:
            return

        if not hasattr(self, "Param0") or not hasattr(self, "Nom_pic"):
            table.setRowCount(0)
            return

        gauge_index = getattr(self, "index_jauge", -1)
        if gauge_index is None or gauge_index < 0:
            table.setRowCount(0)
            return

        params_list = self.Param0[gauge_index] if gauge_index < len(self.Param0) else []
        names_list = self.Nom_pic[gauge_index] if gauge_index < len(self.Nom_pic) else []
        if params_list is None:
            params_list = []
        if names_list is None:
            names_list = []

        self._refreshing_pic_table = True
        try:
            table.setRowCount(len(params_list))

            def _fmt_number(value):
                if value is None:
                    return ""
                if isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value).ravel()
                    return ", ".join(f"{v:.4g}" for v in arr)
                try:
                    return f"{float(value):.4g}"
                except Exception:
                    return str(value)

            for row, params in enumerate(params_list):
                name_val = names_list[row] if row < len(names_list) else ""
                coef_val = params[3] if len(params) > 3 else None
                values = [
                    name_val,
                    params[0] if len(params) > 0 else "",
                    params[1] if len(params) > 1 else "",
                    params[2] if len(params) > 2 else "",
                    coef_val,
                    params[4] if len(params) > 4 else "",
                ]

                for col, val in enumerate(values):
                    item = QTableWidgetItem(_fmt_number(val))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table.setItem(row, col, item)

            self._sync_pic_table_selection(self.index_pic_select)
        finally:
            self._refreshing_pic_table = False

    def _sync_pic_table_selection(self, row: Optional[int]):
        table = getattr(self, "pic_table", None)
        if table is None:
            return

        self._refreshing_pic_table = True
        try:
            if row is None or row < 0 or row >= table.rowCount():
                table.clearSelection()
                return
            table.selectRow(int(row))
        finally:
            self._refreshing_pic_table = False

    def _on_pic_table_selection_changed(self):
        if getattr(self, "_refreshing_pic_table", False):
            return

        table = getattr(self, "pic_table", None)
        if table is None:
            return

        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            return

        self.index_pic_select = selected_rows[0].row()
        self.select_pic()

        if self.Spectrum is not None:
            if self.Spectrum.indexX is not None:
                self.Spectrum.dY = self.Spectrum.y_corr[self.Spectrum.indexX] - self.y_fit_start[self.Spectrum.indexX]
            else:
                self.Spectrum.dY = self.Spectrum.y_corr - self.y_fit_start

    def Print_fit_start(self):
        # simple wrapper UI
        if self.fit_start_box.isChecked():
            self._recompute_y_fit_start()
        else:
            #self.y_fit_start = None
            if self.Spectrum is not None:
                self.Spectrum.dY = self.Spectrum.y_corr
        self._refresh_spectrum_view()

    def Undo_pic(self):
        if self.J[self.index_jauge] >0:
            del(self.Nom_pic[self.index_jauge][-1])
            del(self.Param0[self.index_jauge][-1])
            del(self.Spectrum.Gauges[self.index_jauge].pics[-1])
            self.text_box_msg.setText('Parametre initiale pic'+str(self.J[self.index_jauge])+': \n VIDE')
            del(self.list_text_pic[self.index_jauge][-1])
            self.J[self.index_jauge]-=1
            self.listbox_pic.takeItem(self.J[self.index_jauge])
            try:
                self.pg_spectrum.removeItem(self.plot_pic_fit[self.index_jauge][-1])
            except Exception:
                pass
            del(self.plot_pic_fit[self.index_jauge][-1])
            self.text_box_msg.setText('UNDO PIC')
            del(self.list_y_fit_start[self.index_jauge][-1])
            self.index_pic_select = self.J[self.index_jauge] - 1
            self.Print_fit_start()
            self._refresh_pic_table()

    def Undo_pic_select(self):
        if self.index_pic_select is not None:
            name=self.listbox_pic.item(self.index_pic_select).text()
            #motif = r'_p(\d+)_'(continued)
            #matches = re.findall(motif, name)
            #self.index_pic_select=int(matches[0])
            if self.J[self.index_jauge] >0:
                self.text_box_msg.setText('PIC'+ self.Nom_pic[self.index_jauge][self.index_pic_select] +' DELETED')
                del(self.Nom_pic[self.index_jauge][self.index_pic_select])
                del(self.Param0[self.index_jauge][self.index_pic_select])
                self.text_box_msg.setText('PIC'+str(self.J[self.index_jauge])+': \n VIDE')
                del(self.list_text_pic[self.index_jauge][self.index_pic_select])
                self.J[self.index_jauge]-=1
                self.listbox_pic.takeItem(self.index_pic_select)
                try:
                    self.pg_spectrum.removeItem(self.plot_pic_fit[self.index_jauge][self.index_pic_select])
                except Exception:
                    pass
                del(self.plot_pic_fit[self.index_jauge][self.index_pic_select])
                del(self.list_y_fit_start[self.index_jauge][self.index_pic_select])
                del(self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select])
                self.index_pic_select = min(self.index_pic_select, self.J[self.index_jauge] - 1)
                self.Print_fit_start()
                self._refresh_pic_table()
    
    def select_pic(self):
        if not self.bit_bypass:
            if hasattr(self, "pic_table") and self.pic_table is not None:
                selected_rows = self.pic_table.selectionModel().selectedRows()
                self.index_pic_select = selected_rows[0].row() if selected_rows else -1
            elif hasattr(self, "listbox_pic"):
                self.index_pic_select = self.listbox_pic.currentRow()
            if self.index_pic_select is None or self.index_pic_select < 0:
                return

        if (
            self.index_jauge is None
            or self.index_jauge < 0
            or self.index_jauge >= len(self.Param0)
            or self.index_pic_select is None
            or self.index_pic_select < 0
            or self.index_pic_select >= len(self.Param0[self.index_jauge])
        ):
            return

        # Mise à jour X0/Y0
        self.X0 = self.Param0[self.index_jauge][self.index_pic_select][0]
        self.Y0 = self.Param0[self.index_jauge][self.index_pic_select][1]

        # sigma + modèle
        self.spinbox_sigma.setValue(
            self.Param0[self.index_jauge][self.index_pic_select][2]
        )
        self.model_pic_fit = self.Param0[self.index_jauge][self.index_pic_select][4]
        for i, model in enumerate(self.liste_type_model_pic):
            if model == self.model_pic_fit:
                self.model_pic_type_selector.setCurrentIndex(i)
                break

        # Coeffs spé
        self.bit_bypass = True
        self.f_model_pic_type()
        self.bit_bypass = False
        for i, spin in enumerate(self.coef_dynamic_spinbox):
            spin.setValue(self.Param0[self.index_jauge][self.index_pic_select][3][i])

        self.text_box_msg.setText(
            "PIC SELECT " + self.Nom_pic[self.index_jauge][self.index_pic_select]
        )

        # ------------- Données de zoom & pic sélectionné -------------
        S = self.Spectrum

        y_pic = None

        if (
            self.index_jauge >= 0
            and self.index_pic_select >= 0
            and self.list_y_fit_start
            and len(self.list_y_fit_start[self.index_jauge]) > self.index_pic_select
        ):
            y_pic = self.list_y_fit_start[self.index_jauge][self.index_pic_select]
            self._update_curve_safe(self.curve_zoom_pic, S.wnb, y_pic)
            self._update_curve_safe(self.curve_spec_pic_select, S.wnb, y_pic)
            self._update_curve_safe(self.curve_zoom_data, S.wnb, S.spec - (self.y_fit_start - y_pic) - S.blfit)
            self._update_curve_safe(self.curve_zoom_data_brut, S.wnb, S.spec- S.blfit)
        else:
            self._update_curve_safe(self.curve_zoom_pic, [], [])
            self._update_curve_safe(self.curve_spec_pic_select, [], [])
            if hasattr(S, "wnb") and hasattr(S, "spec"):
                self._update_curve_safe(self.curve_zoom_data, S.wnb, S.spec)
                self._update_curve_safe(self.curve_zoom_data_brut, S.wnb, S.spec)
            else:
                self._update_curve_safe(self.curve_zoom_data, [], [])
                self._update_curve_safe(self.curve_zoom_data_brut, [], [])

        self._sync_pic_table_selection(self.index_pic_select)

        # ------------- Limites du ZOOM : basées sur le pic sélectionné -------------
        if hasattr(S, "wnb") and S.wnb is not None:
            xz = np.asarray(S.wnb)

            if y_pic is not None:
                y_zoom = np.asarray(y_pic, dtype=float)

                # Option : ne zoomer que sur la zone où le pic est significatif
                # par exemple là où y_pic > 10% du max
                mask = y_zoom > (0.1 * np.nanmax(y_zoom))
                if np.any(mask):
                    xz_zoom = xz[mask]
                    yz_zoom = y_zoom[mask]
                else:
                    # fallback : tout le spectre
                    xz_zoom = xz
                    yz_zoom = y_zoom
            else:
                # Pas de pic -> zoom sur le brut
                if hasattr(S, "spec") and S.spec is not None:
                    xz_zoom = xz
                    yz_zoom = np.asarray(S.spec, dtype=float)
                else:
                    xz_zoom = xz
                    yz_zoom = xz  # fallback bidon, mais ne devrait pas arriver

            vb_zoom = self.pg_zoom.getViewBox()

            # On ne garde PLUS le "une seule fois" : on veut que le zoom s'adapte
            # à chaque sélection de pic, donc pas de _zoom_limits_initialized ici.
            self._set_viewbox_limits_from_data(vb_zoom, xz_zoom, yz_zoom, padding=0.5)

            try:
                vb_zoom.setXRange(float(np.nanmin(xz_zoom)), float(np.nanmax(xz_zoom)), padding=0.1)
            except Exception as e:
                print("Zoom XRange error:", e)

    def _build_full_y_plot(self, y_partial, indexX):
        total_len = getattr(self.Spectrum, "n", None)
        if total_len is None:
            x_data = getattr(self.Spectrum, "wnb", None)
            if x_data is not None:
                total_len = len(x_data)

        if total_len is None:
            return np.asarray(y_partial) if y_partial is not None else np.array([])

        if y_partial is None:
            return np.zeros(total_len)

        y_partial = np.asarray(y_partial, dtype=float)
        if len(y_partial) == total_len:
            return y_partial

        y_full = np.zeros(total_len)
        if indexX is not None and len(indexX) == len(y_partial):
            y_full[indexX] = y_partial
        else:
            # Fallback: copy what we can.
            y_full[: min(total_len, len(y_partial))] = y_partial[: min(total_len, len(y_partial))]
        return y_full

    def _clear_selected_peak_overlay(self):
        """Clear the overlay curves highlighting the selected peak."""

        try:
            self._update_curve_safe(self.curve_spec_pic_select, [], [])
            self._update_curve_safe(self.curve_zoom_pic, [], [])
        except Exception:
            pass
    
    def Replace_pic(self):
        if (
            self.Spectrum is None
            or self.index_jauge is None
            or self.index_pic_select is None
            or self.index_jauge < 0
            or self.index_pic_select < 0
        ):
            return

        if (
            self.index_jauge >= len(self.Param0)
            or self.index_jauge >= len(self.Nom_pic)
            or self.index_pic_select >= len(self.Param0[self.index_jauge])
            or self.index_pic_select >= len(self.Nom_pic[self.index_jauge])
            or not hasattr(self.Spectrum, "Gauges")
            or self.index_jauge >= len(self.Spectrum.Gauges)
            or self.index_pic_select >= len(self.Spectrum.Gauges[self.index_jauge].pics)
        ):
            return

        if not self.bit_bypass:
            self.Param0[self.index_jauge][self.index_pic_select] = [
                self.X0,
                self.Y0,
                float(self.spinbox_sigma.value()),
                np.array([float(spin.value()) for spin in self.coef_dynamic_spinbox]),
                str(self.model_pic_fit),
            ]
        else:
            self.Param0[self.index_jauge][self.index_pic_select][0] = self.X0
            self.Param0[self.index_jauge][self.index_pic_select][1] = self.Y0

        new_name = (
            self.Nom_pic[self.index_jauge][self.index_pic_select]
            + "   X0:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][0])
            + "   Y0:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][1])
            + "   sigma:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][2])
            + "   Coef:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][3])
            + " ; Modele:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][4])
        )
        self.list_text_pic[self.index_jauge][self.index_pic_select] = str(new_name)
        self.listbox_pic.takeItem(self.index_pic_select)
        self.listbox_pic.insertItem(self.index_pic_select, str(new_name))
        self.text_box_msg.setText(
            "PIC " + self.Nom_pic[self.index_jauge][self.index_pic_select] + " REPLACE"
        )

        X_pic = CL.Pics(
            name=self.Nom_pic[self.index_jauge][self.index_pic_select],
            ctr=self.Param0[self.index_jauge][self.index_pic_select][0],
            ampH=self.Param0[self.index_jauge][self.index_pic_select][1],
            coef_spe=self.Param0[self.index_jauge][self.index_pic_select][3],
            sigma=self.Param0[self.index_jauge][self.index_pic_select][2],
            model_fit=self.Param0[self.index_jauge][self.index_pic_select][4],
        )
        x_eval = getattr(self.Spectrum, "wnb", None)
        params = X_pic.model.make_params()
        y_plot = X_pic.model.eval(params, x=x_eval) if x_eval is not None else np.array([])

        y_plot_full = self._build_full_y_plot(y_plot, None)

        self.list_y_fit_start[self.index_jauge][self.index_pic_select] = y_plot_full
        self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select] = X_pic

        self.Print_fit_start()

        x_data = getattr(self.Spectrum, "wnb", np.array([]))
        self._update_curve_safe(self.curve_zoom_pic, x_data, y_plot_full)

        self._update_curve_safe(self.curve_spec_pic_select, x_data, y_plot_full)
        if hasattr(self.Spectrum, "spec") and hasattr(self.Spectrum, "blfit"):
            self._update_curve_safe(
                self.curve_zoom_data,
                x_data,
                self.Spectrum.spec - (self.y_fit_start - y_plot_full) - self.Spectrum.blfit,
            )
            self._update_curve_safe(self.curve_zoom_data_brut, x_data, self.Spectrum.spec- self.Spectrum.blfit)
        self._refresh_pic_table()
        self._update_fit_window()
        # Option : ne zoomer que sur la zone où le pic est significatif
        # par exemple là où y_pic > 10% du max
        mask = y_plot_full > (0.1 * np.nanmax(y_plot_full))
        if np.any(mask):
            xz_zoom = x_data[mask]
            yz_zoom = y_plot_full[mask]
        else:
            # fallback : tout le spectre
            xz_zoom = x_data
            yz_zoom = y_plot_full
        vb_zoom = self.pg_zoom.getViewBox()

        # On ne garde PLUS le "une seule fois" : on veut que le zoom s'adapte
        # à chaque sélection de pic, donc pas de _zoom_limits_initialized ici.
        self._set_viewbox_limits_from_data(vb_zoom, xz_zoom, yz_zoom, padding=0.5)

        try:
            vb_zoom.setXRange(float(np.nanmin(xz_zoom)), float(np.nanmax(xz_zoom)), padding=0.1)
        except Exception as e:
            print("Zoom XRange error:", e)

    def Replace_pic_fit(self):
        if (
            self.Spectrum is None
            or self.index_jauge is None
            or self.index_pic_select is None
            or self.index_jauge < 0
            or self.index_pic_select < 0
        ):
            return

        if (
            self.index_jauge >= len(self.Param0)
            or self.index_jauge >= len(self.Nom_pic)
            or self.index_pic_select >= len(self.Param0[self.index_jauge])
            or self.index_pic_select >= len(self.Nom_pic[self.index_jauge])
            or not hasattr(self.Spectrum, "Gauges")
            or self.index_jauge >= len(self.Spectrum.Gauges)
            or self.index_pic_select >= len(self.Spectrum.Gauges[self.index_jauge].pics)
        ):
            return

        out, X_pic, indexX, bit = self.f_pic_fit()
        if not bit:
            self.text_box_msg.setText(
                "PIC FIT " + self.Nom_pic[self.index_jauge][self.index_pic_select] + " PARAM ERROR"
            )
            return

        new_name = (
            self.Nom_pic[self.index_jauge][self.index_pic_select]
            + "   X0:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][0])
            + "   Y0:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][1])
            + "   sigma:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][2])
            + "   Coef:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][3])
            + " ; Modele:"
            + str(self.Param0[self.index_jauge][self.index_pic_select][4])
        )
        self.list_text_pic[self.index_jauge][self.index_pic_select] = str(new_name)
        self.listbox_pic.takeItem(self.index_pic_select)
        self.listbox_pic.insertItem(self.index_pic_select, str(new_name))
        self.text_box_msg.setText(
            "PIC FIT " + self.Nom_pic[self.index_jauge][self.index_pic_select] + " REPLACE"
        )

        if bit:
            y_plot = out.best_fit
            p = out.params
        else:
            p = out.make_params()
            x_eval = getattr(self.Spectrum, "wnb", None)
            y_plot = X_pic.model.eval(p, x=x_eval) if x_eval is not None else np.array([])

        y_plot_full = self._build_full_y_plot(y_plot, indexX)

        self.list_y_fit_start[self.index_jauge][self.index_pic_select] = y_plot_full
        self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select] = X_pic

        self.Print_fit_start()

        x_data = getattr(self.Spectrum, "wnb", np.array([]))
        self._update_curve_safe(self.curve_zoom_pic, x_data, y_plot_full)

        self._update_curve_safe(self.curve_spec_pic_select, x_data, y_plot_full)
        if hasattr(self.Spectrum, "spec") and hasattr(self.Spectrum, "blfit"):
            self._update_curve_safe(
                self.curve_zoom_data,
                x_data,
                self.Spectrum.spec - (self.y_fit_start - y_plot_full) - self.Spectrum.blfit,
            )
            self._update_curve_safe(self.curve_zoom_data_brut, x_data, self.Spectrum.spec - self.Spectrum.blfit)
        self._refresh_pic_table()
        self._update_fit_window()
        # Option : ne zoomer que sur la zone où le pic est significatif
        # par exemple là où y_pic > 10% du max
        mask = y_plot_full > (0.1 * np.nanmax(y_plot_full))
        if np.any(mask):
            xz_zoom = x_data[mask]
            yz_zoom = y_plot_full[mask]
        else:
            # fallback : tout le spectre
            xz_zoom = x_data
            yz_zoom = y_plot_full
        vb_zoom = self.pg_zoom.getViewBox()

        # On ne garde PLUS le "une seule fois" : on veut que le zoom s'adapte
        # à chaque sélection de pic, donc pas de _zoom_limits_initialized ici.
        self._set_viewbox_limits_from_data(vb_zoom, xz_zoom, yz_zoom, padding=0.5)

        try:
            vb_zoom.setXRange(float(np.nanmin(xz_zoom)), float(np.nanmax(xz_zoom)), padding=0.1)
        except Exception as e:
            print("Zoom XRange error:", e)

    def LOAD_Spectrum(self, item=None, Spectrum=None):
        """Charge un spectre dans self.Spectrum et reconstruit toute la structure de fit (sans Matplotlib)."""
        save_index_spec = -1
        old_spectrum = self.Spectrum
        nb_j_old = len(old_spectrum.Gauges) if old_spectrum is not None else 0

        self._clear_selected_peak_overlay()
        bypass = False  # valeur par défaut

        # =========================
        # 1) Sélection de Spectrum
        # =========================
        if Spectrum is None:
            # On vient de l'UI (spinner / listbox)
            if not self.bit_bypass and old_spectrum is not None:
                # On sauvegarde le spectre courant dans le RUN (si possible)
                try:
                    self.RUN.Spectra[self.index_spec] = old_spectrum
                except Exception:
                    self.text_box_msg.setText("ERROR Load Spec")
                    return

                save_index_spec = self.spinbox_spec_index.value()

            # Choix de l'index de spectre cible
            if save_index_spec >= 0:
                target_index = save_index_spec
            else:
                target_index = self.index_spec

            # Si on change réellement de spectre
            if target_index != self.index_spec:
                if not self.bit_bypass:
                    self.index_spec = target_index

            # Nouveau spectre de référence
            try:
                new_spec = self.RUN.Spectra[self.index_spec]
            except (AttributeError, IndexError, TypeError):
                self.text_box_msg.setText("ERROR: invalid spectrum index")
                return

            new_has_fit = getattr(new_spec, "bit_fit", False)
            old_has_fit = getattr(old_spectrum, "bit_fit", False) if old_spectrum is not None else False

            if new_has_fit:
                # On prend directement le spectre du RUN (déjà fitté)
                self.Spectrum = new_spec
            elif old_has_fit:
                # On garde les paramètres de fit de l'ancien, mais on remplace jauges/modèle
                save_J = copy.deepcopy(new_spec.Gauges)
                save_M = copy.deepcopy(new_spec.model)
                self.Spectrum = copy.deepcopy(old_spectrum)
                self.Spectrum.Gauges = save_J
                self.Spectrum.model = save_M
                bypass = True
            else:
                # Pas de fit ni dans l'ancien, ni dans le nouveau : simple copie
                self.Spectrum = copy.deepcopy(new_spec)

        else:
            # Chargement direct d'un objet Spectre
            self.Spectrum = Spectrum
            bypass = True
            self.bit_bypass = False

        S = self.Spectrum
        if S is None:
            self.text_box_msg.setText("ERROR: no Spectrum loaded")
            return

        # =========================================
        # 2) Réinitialise toute la structure de fit
        # =========================================
        has_global_fit = getattr(S, "bit_fit", False)
        has_gauge_fit = any(getattr(G, "bit_fit", False) for G in getattr(S, "Gauges", []))

        if has_global_fit or bypass or self.bit_bypass or has_gauge_fit:
            nb_j = len(S.Gauges) if S.Gauges is not None else 0
            self.Zone_fit = [None for _ in range(nb_j)]
            self.X_s = [None for _ in range(nb_j)]
            self.X_e = [None for _ in range(nb_j)]
            self.bit_fit = [False for _ in range(nb_j)]

            # Gestion des zones de fit : uniquement logique (plus de remplissage Matplotlib)
            if getattr(S, "indexX", None) is not None:
                for i in range(nb_j):
                    self.Zone_fit[i] = S.Gauges[i].indexX
                # La jauge 0 garde aussi la zone globale
                self.Zone_fit[0] = S.indexX

            # Reconstruction du modèle global si besoin
            if S.model is None and getattr(S, "Gauges", None):
                S.model = None
                for j, g in enumerate(S.Gauges):
                    g.Update_model()
                    if j == 0:
                        S.model = g.model
                    else:
                        S.model += g.model

            # (re)construction des listes de pics
            self.Nom_pic = [[] for _ in range(nb_j)]
            self.list_text_pic = [[] for _ in range(nb_j)]
            self.Param0 = [[] for _ in range(nb_j)]
            self.J = [0 for _ in range(nb_j)]
            self.list_y_fit_start = [[] for _ in range(nb_j)]
            self.list_name_gauges = []

            if getattr(S, "Gauges", None):
                for i, Jg in enumerate(S.Gauges):
                    self.list_name_gauges.append(Jg.name)
                    for j, p in enumerate(Jg.pics):
                        self.Nom_pic[i].append(p.name)
                        new_P0, param = p.Out_model()
                        # new_P0 = [X0, Y0, sigma, coef]
                        self.Param0[i].append(new_P0 + [p.model_fit])
                        new_name = (
                            p.name
                            + "   X0:" + str(self.Param0[i][-1][0])
                            + "   Y0:" + str(self.Param0[i][-1][1])
                            + "   sigma:" + str(self.Param0[i][-1][2])
                            + "   Coef:" + str(self.Param0[i][-1][3])
                            + " ; Modele:" + str(self.Param0[i][-1][4])
                        )
                        self.J[i] += 1
                        self.list_text_pic[i].append(new_name)
                        y_plot = p.model.eval(param, x=S.wnb)
                        self.list_y_fit_start[i].append(y_plot)

            # Mise à jour des infos de filtre et baseline
            if hasattr(self, "filtre_type_selector") and getattr(S, "type_filtre", None) is not None:
                index = self.filtre_type_selector.findText(S.type_filtre)
                if index != -1:
                    self.filtre_type_selector.setCurrentIndex(index)

            if getattr(S, "param_f", None) is not None and len(S.param_f) >= 2:
                self.param_filtre_1_entry.setText(str(S.param_f[0]))
                self.param_filtre_2_entry.setText(str(S.param_f[1]))

            if hasattr(S, "deg_baseline"):
                self.deg_baseline_entry.setValue(S.deg_baseline)

            self._spectrum_limits_initialized = False
            self._zoom_limits_initialized = False

            # -> Data_treatement + y_fit_start + refresh graph
            self.Baseline_spectrum()

            # Mise à jour UI jauge/pics
            self.listbox_pic.clear()
            if getattr(S, "Gauges", None):
                self.index_jauge = 0
                if self.list_name_gauges:
                    try:
                        idx_gauge = self.liste_type_Gauge.index(self.list_name_gauges[self.index_jauge])
                        self.Gauge_type_selector.setCurrentIndex(idx_gauge)
                    except ValueError:
                        # nom inconnu → on n'impose pas l'index du combo
                        pass
                self.LOAD_Gauge()
            else:
                self.index_jauge = -1
    
    def f_index_gauge(self,spec):
        l_name = [ga.name for ga in spec.Gauges]
        try:
            index = l_name.index(self.Gauge_type_selector.currentText())
            print(index)
        except Exception:
            index=None
            self.Gauge_type_selector.setCurrentIndex(l_name.index(self.Gauge_type_selector.currentText()))
        return index

    def LOAD_Gauge(self):
        """Charge la jauge `self.index_jauge` dans l'UI (sans Matplotlib)."""
        self.is_loading_gauge = False
        self.bit_modif_jauge = True

        self._clear_selected_peak_overlay()

        G = self.Spectrum.Gauges[self.index_jauge]

        if G.name == "Ruby":
            self.spinbox_T.setEnabled(True)
        else:
            self.spinbox_T.setEnabled(False)
            self.spinbox_T.setValue(293)
            self.deltalambdaT = 0

        self.lamb0_entry.setText(str(G.lamb0))
        self.name_spe_entry.setText(str(G.name_spe))
        self.spinbox_P.setValue(G.P)

        self.f_dell_lines()
        self.f_p_move(G, value=G.P)

        self.listbox_pic.clear()
        for name in self.list_text_pic[self.index_jauge]:
            self.listbox_pic.addItem(name)

        self.Gauge_type_selector.setCurrentIndex(
            self.liste_type_Gauge.index(G.name)
        )
        self.name_gauge.setText("In")
        self.name_gauge.setStyleSheet("background-color: green;")
        self._refresh_pic_table()

    def on_spec_index_changed(self, value: int):
        """Appelé quand on change l'index de spectre via la spinbox."""
        if self.RUN is None:
            return
        if not hasattr(self.RUN, "Spectra") or not self.RUN.Spectra:
            return
        if value < 0 or value >= len(self.RUN.Spectra):
            return

        self.index_spec = value
        self.Spectrum = self.RUN.Spectra[self.index_spec]

        self.bit_bypass = True
        try:
            # si LOAD_Spectrum accepte un argument Spectrum, utilise-le
            self.LOAD_Spectrum(Spectrum=self.Spectrum)
        except TypeError:
            # sinon, si ta LOAD_Spectrum lit self.index_spec directement
            self.LOAD_Spectrum()
        self.bit_bypass = False

        self.Update_Print()
