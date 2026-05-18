# BANC CED Dynamic 

import copy
import io
import logging
import os
import sys
import traceback
from typing import Optional
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QGridLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QWidget,
)

from ui_layout import UiLayoutMixin
from spectrum_view import SpectrumViewMixin
from ddac_view import DdacViewMixin, RunViewState
from ui_services import (
    BackgroundTaskMixin,
    RunStateMixin,
    ThemeMixin,
    CedCreationServiceMixin,
    FileFolderServiceMixin,
    SpectrumWorkflowMixin,
    GaugeWorkflowMixin,
    KeyboardShortcutMixin,
    FitWorkflowMixin,
)

logger = logging.getLogger(__name__)

pg.setConfigOptions(
    antialias=True,          # courbes lissées
    useOpenGL=False,         # tu peux tester True si ta carte suit bien
    #downsample=True,         # évite de tracer chaque point si énorme -> bug 
)

Setup_mode = False

folder_start=r"F:\Aquisition_Banc_CEDd"
folder_CEDd=r"F:\Aquisition_Banc_CEDd\Fichier_CEDd"


from Bibli_python import CL_FD_Update as CL

class ProgressDialog(QDialog):
    """
    Dialogue interactif pour afficher un spectre + bouton Stop.
    Compatible avec ton usage dans le multi-fit.
    """

    def __init__(self, cancel_text="Annuler", value=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Progression du fit")
        self.was_canceled = False

        layout = QVBoxLayout(self)

        # -------- LABEL --------
        self.label = QLabel("Traitement en cours…")
        layout.addWidget(self.label)

        # -------- BARRE DE PROGRESSION --------
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(value)
        layout.addWidget(self.progress)

        # -------- BOUTONS --------
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton(cancel_text)
        self.btn_cancel.clicked.connect(self.cancel)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def cancel(self):
        self.was_canceled = True
        self.close()

    def setValue(self, v):
        self.progress.setValue(v)

    def setLabelText(self, text):
        self.label.setText(text)

    def wasCanceled(self):
        return self.was_canceled

    def get_user_choice(self):
        """
        Compatible avec ton initData :
        - renvoie (True, params, inter) si l'utilisateur continue,
        - renvoie (False, None, None) si stop.
        """
        return (not self.was_canceled, None, None)

class MainWindow(
    QMainWindow,
    UiLayoutMixin,
    SpectrumViewMixin,
    DdacViewMixin,
    ThemeMixin,
    BackgroundTaskMixin,
    RunStateMixin,
    CedCreationServiceMixin,
    FileFolderServiceMixin,
    SpectrumWorkflowMixin,
    GaugeWorkflowMixin,
    KeyboardShortcutMixin,
    FitWorkflowMixin,
):
    def __init__(self, folder_start=None):
        super().__init__()

        self._init_state(folder_start)
        self._build_ui()
        self._connect_main_signals()
        self._initialize_data()

        if Setup_mode is True:
            self._run_setup_mode()
            print("setup mode RUN")

    def _init_state(self, folder_start: Optional[str]) -> None:
        self._init_theme_state()
        self._init_task_state()
        self._init_run_state()

        self.Spectrum = None
        self.liste_chemins_fichiers = []
        self._spectro_cache = {
            "path": None,
            "mtime_ns": None,
            "size": None,
            "dataframe": None,
        }
        self._summary_dirty_indices = set()
        self.dossier_selectionne=folder_CEDd #r"F:\Aquisition_Banc_CEDd\Fichier_CEDd"
        self.Spectrum_save=None
        self.CEDd_save=None
        self.Gauge_select=None
        self.data_Spectro=None
        self.RUN = None
        self.setup_mode = bool(Setup_mode)

        self.viewer = None
        self.is_reduced_column_mode: int = 0
        """Réduit l'affichage des colonnes Spectrum lorsque True."""

        self.liste_objets = []  # liste des CEDd chargés (legacy)
        self.index_select = -1

        self.x1: float = 0.0
        self.y1: float = 0.0
        self.x5: float = 0.0
        self.y3: float = 0.0
        self.x_clic: float = 0.0
        self.y_clic: float = 0.0

        if folder_start is None:
            folder_start = r"E:\Aquisition_Banc_CEDd"
        self.folder_start = folder_start

        # Tables de stylers mises en cache pour éviter de recréer les mêmes
        # structures à chaque changement de thème.
        self._spectrum_pen_mapping = {
            "curve_spec_data": "spectrum_data",
            "curve_spec_fit": "spectrum_fit",
            "curve_spec_brut": "baseline_brut",
            "curve_spec_blfit": "baseline_fit",
            "curve_dy": "dy",
            "line_dy_zero": "zero_line",
            "curve_zoom_data": "zoom_data",
            "curve_zoom_data_brut": "zoom_data_brut",
            "vline_spec": "selection_line",
            "hline_spec": "selection_line",
            "vline_dy": "selection_line",
            "vline_zoom": "selection_line",
        }
        self._spectrum_brush_mapping = {
            "curve_spec_pic_select": "spectrum_pic_brush",
            "curve_zoom_pic": "zoom_pic_brush",
            "cross_zoom": "cross_zoom",
        }
        self._ddac_pen_mapping = {
            "line_t_P": "line_t_frame",
            "line_t_dPdt": "line_t_frame",
            "line_t_sigma": "line_t_frame",
            "line_t_dlambda": "line_t_frame",
            "line_t_spec_P": "line_t_spec",
            "line_t_spec_dPdt": "line_t_spec",
            "line_t_spec_sigma": "line_t_spec",
            "line_t_spec_dlambda": "line_t_spec",
            "line_nspec": "line_t_frame",
            "line_p0": "baseline_time",
            "scatter_P": "scatter",
            "scatter_dPdt": "scatter",
            "scatter_sigma": "scatter",
            "scatter_dlambda": "scatter",
        }

    def _build_ui(self) -> None:
        self._setup_main_window()

        self.grid_layout = QGridLayout()
        central_widget = QWidget()
        central_widget.setLayout(self.grid_layout)
        self.setCentralWidget(central_widget)

        self._setup_file_box()           # (4, 0) -> file_spectro/oscilo/movie
        self._setup_tools_tabs()         # (0, 1)
        self._setup_gauge_info()         # (1, 2)
        self._setup_spectrum_box()       # (0, 2)
        self._setup_ddac_box()           # (0, 3)
        self._setup_file_gestion()       # (2, 0)
        self._setup_python_kernel()      # (2, 2)
        self._setup_text_box_msg()       # (1, 0)
        self._setup_layout_stretch()

        self._apply_theme(self.current_theme)

    def _connect_main_signals(self) -> None:
        self._apply_theme_toggle_state()

    def _initialize_data(self) -> None:
        self.load_latest_file()
        self.f_gauge_select()

    def toggle_python_kernel(self, checked: bool):
        """
        Affiche/cache le kernel Python et réarrange la grille :
        - si caché :
            SpectraBox: (0,2, 2,1)
            AddBox    : (2,2, 1,1)
        - si affiché :
        SpectraBox: (0,2, 1,1)
        AddBox    : (1,2, 1,1)
        promptBox : (2,2, 1,1)
        """
        if checked:
            # bouton enfoncé -> on montre le kernel
            self.python_kernel_button.setText("Hide Python Kernel")
            self.promptBox.show()

            # retirer puis replacer pour être sûr
            self.grid_layout.removeWidget(self.SpectraBox)
            #self.grid_layout.removeWidget(self.AddBox)
            self.grid_layout.removeWidget(self.promptBox)

            self.grid_layout.addWidget(self.SpectraBox, 0, 2, 2, 1)
            #self.grid_layout.addWidget(self.AddBox,     1, 2, 1, 1)
            self.grid_layout.addWidget(self.promptBox,  2, 2, 1, 1)

        else:
            # bouton relâché -> on cache le kernel
            self.python_kernel_button.setText("Show Python Kernel")
            self.promptBox.hide()

            self.grid_layout.removeWidget(self.SpectraBox)
            #self.grid_layout.removeWidget(self.AddBox)
            # promptBox peut rester dans la grille mais caché

            self.grid_layout.addWidget(self.SpectraBox, 0, 2, 3, 1)
            #self.grid_layout.addWidget(self.AddBox,     2, 2, 1, 1)

    # ==================================================================
    # ===============   SETUP MODE POUR DEBUG  =========================
    # ==================================================================
    def _run_setup_mode(self):
        self.CREAT_new_Spectrum()
        self.Gauge_type_selector.setCurrentIndex(0)
        self.f_gauge_select()
        self.is_loading_gauge = True
        self.spinbox_P.setValue(11)
        self.ADD_gauge()
        self.FIT_lmfitVScurvfit()
        self.CREAT_new_CEDd()

    def _update_graphicslayout_sizes(self, glw, row_factors=None, col_factors=None):
        """
        Ajuste dynamiquement hauteurs de lignes et largeurs de colonnes
        d'un GraphicsLayoutWidget en fonction de facteurs.
        - row_factors : tuple/list, ex (2, 2, 1) pour 3 lignes
        - col_factors : tuple/list, ex (3, 7) pour 2 colonnes (30% / 70%)
        """
        layout = glw.ci.layout

        # ----- LIGNES -----
        if row_factors is not None:
            total_rf = float(sum(row_factors))
            h = glw.height()
            if h <= 0:
                h = glw.sizeHint().height()

            for i, f in enumerate(row_factors):
                row_h = int(h * (f / total_rf))
                layout.setRowPreferredHeight(i, row_h)
                layout.setRowMinimumHeight(i, 0)
                # pas de Maximum pour laisser Qt respirer

        # ----- COLONNES -----
        if col_factors is not None:
            total_cf = float(sum(col_factors))
            w = glw.width()
            if w <= 0:
                w = glw.sizeHint().width()

            for j, f in enumerate(col_factors):
                col_w = int(w * (f / total_cf))
                layout.setColumnPreferredWidth(j, col_w)
                layout.setColumnMinimumWidth(j, 0)
                # là aussi, pas de Maximum

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Spectrum box
        if hasattr(self, "pg_spec"):
            self._update_graphicslayout_sizes(
                self.pg_spec,
                row_factors=getattr(self, "_spec_row_factors", None),
                col_factors=getattr(self, "_spec_col_factors", None),
            )

        # dDAC box
        if hasattr(self, "pg_ddac"):
            self._update_graphicslayout_sizes(
                self.pg_ddac,
                row_factors=getattr(self, "_ddac_row_factors", None),
                col_factors=getattr(self, "_ddac_col_factors", None),
            )

    def Print_error(self,error): 
        error_box = QMessageBox(self)
        error_box.setWindowTitle("Warning error")
        
        text='ERROR:'+ str(error)+'  \n check in console variable \n Press X for quit'
        error_box.setText(text)

        x_button = error_box.addButton("Quit (X)", QMessageBox.AcceptRole)
        error_box.setDefaultButton(x_button)

        def on_key_press(event):
            if event.key() == Qt.Key_X:
                x_button.click()

        error_box.keyPressEvent = on_key_press
        
        error_box.exec_()    

    def execute_code(self, code=None):
        if code is None:
            code = self.text_edit.toPlainText()

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        exec_locals = {'self': self, 'CL': CL}

        try:
            exec(code, {}, exec_locals)
            output = stdout_capture.getvalue()
            output += exec_locals.get('result', 'Exécution réussie sans sortie spécifique.')
            self.output_display.setPlainText(output)
        except Exception as e:
            self.output_display.setPlainText(f"Erreur : {e}")
        finally:
            sys.stdout = old_stdout
    
    def toggle_colonne(self):
        if self.is_reduced_column_mode==0:
            self.grid_layout.setColumnStretch(2, 1)
            self.grid_layout.setColumnStretch(3, 8)
            self.is_reduced_column_mode=1
        elif self.is_reduced_column_mode==1:
            self.grid_layout.setColumnStretch(2, 4)
            self.grid_layout.setColumnStretch(3, 6)
            self.is_reduced_column_mode=2
        else:
            self.grid_layout.setColumnStretch(2, 8)
            self.grid_layout.setColumnStretch(3, 1)
            self.is_reduced_column_mode=0

    def try_command(self,item):
        print("à coder")
        #command=item.text()
        #self.execute_code(code=text[1])

    def display_command(self, item):
        previous_command = self.text_edit.toPlainText()

        commande=self.list_Commande_python.item(self.list_Commande.row(item))
        # self.update the label and save the current text as previous_text
        self.text_edit.setText(previous_command + commande.text())

    def code_print(self):
        expr = self.text_edit.toPlainText()
        self.execute_code(code=f"print({expr})")
    
    def code_len(self):
        self.execute_code(code="print(len("+self.text_edit.toPlainText()+'))')

    def code_clear(self):
        self.text_edit.setText("")

#########################################################################################################################################################################################
#? MOVIE 
#########################################################################################################################################################################################
#? COMMANDE FILE
#########################################################################################################################################################################################
    def Update_Print(self):
        state = self._get_state_for_run()
        if state is None:
            return

        show_flags = (
            bool(self.chk_show_dpdt.isChecked()),  # dPdt
            bool(self.chk_show_T.isChecked()),  # T
            bool(self.chk_show_piezo.isChecked()),  # piezo
            bool(self.chk_show_corr.isChecked()),  # corr
            bool(self.chk_show_P.isChecked()),  # P
        )
        visibility_cache = getattr(state, "_print_visibility_cache", None)
        if visibility_cache is None:
            visibility_cache = {}
            setattr(state, "_print_visibility_cache", visibility_cache)

        def _set_visible_if_changed(cache_key, items, visible):
            previous = visibility_cache.get(cache_key)
            item_count = sum(1 for item in items if item is not None)
            current = (visible, item_count)
            if previous == current:
                return False
            for item in items:
                if item is not None:
                    item.setVisible(visible)
            visibility_cache[cache_key] = current
            return True

        changed = False
        show_dPdt, show_T, show_piezo, show_corr, show_P = show_flags
        changed |= _set_visible_if_changed("T", state.curves_T, show_T)
        changed |= _set_visible_if_changed("P", state.curves_P, show_P)
        changed |= _set_visible_if_changed("dPdt", state.curves_dPdt, show_dPdt)
        changed |= _set_visible_if_changed("sigma", state.curves_sigma, show_P)
        changed |= _set_visible_if_changed("piezo", [state.piezo_curve], show_piezo)
        changed |= _set_visible_if_changed("corr", [state.corr_curve], show_corr)

        # Pas de refresh global si aucun état de visibilité n'a changé.
        if not changed:
            return

        # Recalcule la plage Y uniquement si la visibilité a été modifiée.
        y_min = None
        y_max = None

        def _update_min_max(curves, visible):
            nonlocal y_min, y_max
            if not visible:
                return
            for curve in curves:
                if curve is None:
                    continue
                y = curve.getData()[1]
                if y is None or not len(y):
                    continue
                y_data = np.asarray(y, dtype=float)
                finite = y_data[np.isfinite(y_data)]
                if finite.size == 0:
                    continue
                local_min = float(np.min(finite))
                local_max = float(np.max(finite))
                y_min = local_min if y_min is None else min(y_min, local_min)
                y_max = local_max if y_max is None else max(y_max, local_max)

        _update_min_max(state.curves_T, show_T)
        _update_min_max(state.curves_P, show_P)
        _update_min_max(state.curves_dPdt, show_dPdt)
        _update_min_max(state.curves_sigma, show_P)
        _update_min_max([state.piezo_curve], show_piezo)
        _update_min_max([state.corr_curve], show_corr)

        if y_min is not None and y_max is not None and y_min != y_max:
            self.pg_dPdt.setYRange(y_min * 1.01, y_max * 1.01, padding=0)

        # Auto-range seulement sur les graphes impactés par les toggles.
        if show_P != visibility_cache.get("_last_P_autorange"):
            self.pg_P.enableAutoRange(axis='y', enable=True)
            self.pg_sigma.enableAutoRange(axis='y', enable=True)
            visibility_cache["_last_P_autorange"] = show_P

        self.pg_dPdt.enableAutoRange(axis='y', enable=True)
#########################################################################################################################################################################################
#? COMMANDE self.update 
    def f_filtre_select(self):
        col1 = self.filtre_type_selector.model().item(self.filtre_type_selector.currentIndex()).background().color().getRgb()
        self.filtre_type_selector.setStyleSheet("background-color: rgba{};	selection-background-color: gray;".format(col1))
        if self.filtre_type_selector.currentText() == "svg":
            self.param_filtre_1_name.setText("window_length")
            self.param_filtre_2_name.setText("polyorder")
        elif self.filtre_type_selector.currentText() == "fft":
            self.param_filtre_1_name.setText("f_c low")
            self.param_filtre_2_name.setText("f_c high")
        else:
            self.param_filtre_1_name.setText("filtre p1")
            self.param_filtre_2_name.setText("filtre p2")

    def f_model_pic_type(self):
        col1 = self.model_pic_type_selector.model().item(
            self.model_pic_type_selector.currentIndex()
        ).background().color().getRgb()
        self.model_pic_type_selector.setStyleSheet(
            "background-color: rgba{}; selection-background-color: gray;".format(col1)
        )

        # Supprimer les anciens widgets
        while self.coef_dynamic_spinbox:
            w = self.coef_dynamic_spinbox.pop()
            self.ParampicLayout.removeWidget(w)
            w.deleteLater()

        while self.coef_dynamic_label:
            w = self.coef_dynamic_label.pop()
            self.ParampicLayout.removeWidget(w)
            w.deleteLater()

        self.model_pic_fit = self.model_pic_type_selector.currentText()
        pic_exemple = CL.Pics(model_fit=self.model_pic_fit)

        for i, coef in enumerate(pic_exemple.name_coef_spe):
            layh = QHBoxLayout()
            text = coef
            if coef == "fraction":
                text = "\u03B1"
            elif coef == "beta":
                text = "\u03B2"
            elif coef == "sigma_r":
                text = "\u03C3 <sub>right<\sub>"
            elif coef == "expon":
                text = "m"
            elif coef == "skew":
                text = "\u03C5"

            coef_label = QLabel(text + ":", self)
            spinbox_coef = QDoubleSpinBox(self)
            spinbox_coef.valueChanged.connect(self.setFocus)
            spinbox_coef.setRange(-100, 100)
            spinbox_coef.setSingleStep(0.01)
            spinbox_coef.setValue(1 - i * 1.5)

            layh.addWidget(coef_label)
            layh.addWidget(spinbox_coef)
            self.ParampicLayout.addLayout(layh)

            self.coef_dynamic_label.append(coef_label)
            self.coef_dynamic_spinbox.append(spinbox_coef)   
            
    def Read_RUN(self, RUN):
        l_P, l_sigma_P, l_lambda, l_fwhm = [], [], [], []
        l_spe, l_T, l_sigma_T = [], [], []
        Gauges_RUN = RUN.Gauges_init

        Summary = getattr(RUN, "Summary", None)

        # --- Cas Summary vide ou None : on renvoie des tableaux vides mais cohérents
        if Summary is None or Summary.empty:
            # Time / spectre_number basés sur Time_spectrum + list_nspec
            if RUN.Time_spectrum is not None:
                dt_n = len(RUN.Time_spectrum) - len(RUN.list_nspec)
                if dt_n > 0:
                    Time = [RUN.Time_spectrum[i] for i in RUN.list_nspec]
                elif dt_n < 0:
                    dt = np.mean(np.diff(RUN.Time_spectrum))
                    Time = np.array(
                        list(RUN.Time_spectrum)
                        + [RUN.Time_spectrum[-1] + dt * (1 + i) for i in range(abs(dt_n))]
                    )
                else:
                    Time = RUN.Time_spectrum
            else:
                Time = RUN.list_nspec

            spectre_number = RUN.list_nspec

            # Oscillo éventuel
            if RUN.data_Oscillo is not None:
                time_amp = np.array(RUN.data_Oscillo["Time"])
                b = np.array(RUN.data_Oscillo["Channel3"])
                amp = CL.savgol_filter(b, 101, 2)
            else:
                time_amp = np.array([])
                amp = np.array([])

            # Pour chaque jauge, on met des arrays vides
            for G in Gauges_RUN:
                l_P.append(np.array([]))
                l_sigma_P.append(np.array([]))
                l_lambda.append(np.array([]))
                l_fwhm.append(np.array([]))

            return (
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
            )

        # --- Cas Summary non vide : on garde TOUTES les lignes, avec NaN où il faut
        n_rows = len(Summary)

        for G in Gauges_RUN:
            name = G.name
            name_spe = G.name_spe

            # on convertit toute la colonne en float, NaN compris
            col_P = pd.to_numeric(Summary["P_" + name], errors="coerce").to_numpy(dtype=float)
            col_sigma_P = pd.to_numeric(Summary["sigma_P_" + name], errors="coerce").to_numpy(dtype=float)
            col_lambda = pd.to_numeric(Summary["lambda_" + name], errors="coerce").to_numpy(dtype=float)
            col_fwhm = pd.to_numeric(Summary["fwhm_" + name], errors="coerce").to_numpy(dtype=float)

            l_P.append(col_P)
            l_sigma_P.append(col_sigma_P)
            l_lambda.append(col_lambda)
            l_fwhm.append(col_fwhm)

            if "Ru" in name_spe and "Deltap12" in Summary.columns:
                col_spe = pd.to_numeric(Summary["Deltap12"], errors="coerce").to_numpy(dtype=float)
                l_spe.append(col_spe)

            if "T" in name_spe:
                if "T_" + name in Summary.columns:
                    col_T = pd.to_numeric(Summary["T_" + name], errors="coerce").to_numpy(dtype=float)
                    l_T.append(col_T)
                if "sigma_T_" + name in Summary.columns:
                    col_sigma_T = pd.to_numeric(
                        Summary["sigma_T_" + name], errors="coerce"
                    ).to_numpy(dtype=float)
                    l_sigma_T.append(col_sigma_T)

        # --- Construction de Time aligné sur Summary
        if RUN.Time_spectrum is not None:
            # on aligne sur le nombre de lignes de Summary
            ts = np.array(RUN.Time_spectrum)
            if len(ts) >= n_rows:
                Time = ts[:n_rows]
            else:
                # on extrapole si besoin
                if len(ts) > 1:
                    dt = np.mean(np.diff(ts))
                else:
                    dt = 1.0
                extra = ts[-1] + dt * np.arange(1, n_rows - len(ts) + 1)
                Time = np.concatenate([ts, extra])
        else:
            # fallback : on prend l'indice des spectres / Summary
            if "n°Spec" in Summary.columns:
                Time = Summary["n°Spec"].to_numpy(dtype=float)
            else:
                Time = np.arange(n_rows, dtype=float)

        spectre_number = RUN.list_nspec

        # --- Oscillo éventuel
        if RUN.data_Oscillo is not None:
            time_amp = np.array(RUN.data_Oscillo["Time"])
            b = np.array(RUN.data_Oscillo["Channel3"])
            amp = CL.savgol_filter(b, 101, 2)
        else:
            time_amp = np.array([])
            amp = np.array([])

        return (
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
        )

    def Read_Movie(self,RUN):
        # Lecture vidéo avec OpenCV
        import cv2

        cap = cv2.VideoCapture(RUN.folder_Movie)
        fps = cap.get(cv2.CAP_PROP_FPS)
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return cap,fps,nb_frames

    def SELECT_CEDd(self,item):
        self._save_current_run()

        run_id = item.data(Qt.UserRole)
        if run_id is None:
            self.text_box_msg.setText("RUN introuvable (clé manquante)")
            return

        state = self.runs.get(run_id)
        if state is None:
            self.text_box_msg.setText("RUN introuvable")
            return

        # Synchronise les pointeurs courants
        self.current_run_id = run_id
        self.RUN = copy.deepcopy(state.ced)
        self.index_select = self.liste_objets_widget.row(item)

        if hasattr(self.RUN,"fps") and self.RUN.fps is not None:
            try:
                text_fps="Movie :1e"+str(round(np.log10(self.RUN.fps),2))+"fps"
            except Exception as e:
                print("fps log ERROR:",e)
                text_fps="Movie :"+str(round(self.RUN.fps,2))+"fps"
        else:
            text_fps="No Movie"

    
        if state.index_cam:
            self.current_index = len(state.index_cam)//2 if state.index_cam else 0
            self.slider.setMaximum(max(0, len(state.index_cam) - 1))
            self.slider.setValue(self.current_index)
        self._update_movie_frame()
        self.label_CED.setText( f"CEDd {item.text()} fps :{text_fps}")
        if state.time is not None:
            x_min,x_max=min(state.time),max(state.time)
            self.pg_P.setXRange(x_min,x_max,padding=0.01)
            self.pg_dPdt.setXRange(x_min,x_max,padding=0.01)
            self.pg_sigma.setXRange(x_min,x_max,padding=0.01)
            self.pg_dlambda.setXRange(x_min,x_max,padding=0.01)

    
    
#########################################################################################################################################################################################
#? COMMANDE UNLOAD
    def CLEAR_CEDd(self,item=None):
        run_id = None
        if item is not None:
            run_id = item.data(Qt.UserRole)
        if run_id is None:
            run_id = self.current_run_id

        state = self.runs.pop(run_id, None)
        if state is None:
            return

        self._release_cap_for_state(state)

        # Nettoyage des courbes stockées dans l'état
        for curve in state.curves_T:
            try:
                self.pg_dPdt.removeItem(curve)
            except Exception:
                pass
        for curve in state.curves_P:
            try:
                self.pg_P.removeItem(curve)
            except Exception:
                pass
        for curve in state.curves_dPdt:
            try:
                self.pg_dPdt.removeItem(curve)
            except Exception:
                pass
        for curve in state.curves_sigma:
            try:
                self.pg_sigma.removeItem(curve)
            except Exception:
                pass
        for curve in state.curves_dlambda:
            try:
                self.pg_dlambda.removeItem(curve)
            except Exception:
                pass
        if state.piezo_curve is not None:
            try:
                self.pg_P.removeItem(state.piezo_curve)
            except Exception:
                pass
        if state.corr_curve is not None:
            try:
                self.pg_dPdt.removeItem(state.corr_curve)
            except Exception:
                pass

        if state.list_item is not None:
            index = self.liste_objets_widget.row(state.list_item)
            self.liste_objets_widget.takeItem(index)

        if run_id == self.current_run_id:
            self.current_run_id = None
            self.RUN = None

        # Nettoie les mappings fichier -> run_id
        keys_to_delete = [k for k, v in self.file_index_map.items() if v == run_id]
        for k in keys_to_delete:
            del self.file_index_map[k]

        # Supprime le CEDd legacy si présent
        self.liste_objets = [ced for ced in self.liste_objets if self._get_run_id(ced) != run_id]

        # Sélectionne un autre run si disponible
        if self.liste_objets_widget.count() > 0:
            next_item = self.liste_objets_widget.item(0)
            self.SELECT_CEDd(item=next_item)
    
#########################################################################################################################################################################################
#? COMMANDE CLEAR

    def CLEAR_ALL(self, empty=False):
        """
        Réinitialise tout l'état lié au module 'spectrum' (pics, jauges, fit),
        sans bidouiller les axes Matplotlib CEDd.
        """
        # Gestion du spectre courant
        if empty:
            self.Spectrum = None
            self.nb_jauges = 0
            self.list_name_gauges = []
            # Vide les courbes PyQtGraph
            self._refresh_spectrum_view()
            self._refresh_pic_table()
        else:
            if isinstance(self.Spectrum, CL.Spectre):
                self.nb_jauges = len(self.Spectrum.Gauges)
                self.list_name_gauges = [jauge.name for jauge in self.Spectrum.Gauges]
                print("gauges save")
            else:
                print("gauges dell")
                self.nb_jauges = 0
                self.list_name_gauges = []
                self._refresh_pic_table()

        # Variables de fit/pics
        self.plot_fit = []
        self.plot_pic = []
        self.plot_pic_fit = [[] for _ in range(self.nb_jauges)]

        self.list_y_fit_start = [[] for _ in range(self.nb_jauges)]

        self.Nom_pic = [[] for _ in range(self.nb_jauges)]
        self.Spec_fit = [None for _ in range(self.nb_jauges)]
        self.Param0 = [[] for _ in range(self.nb_jauges)]
        self.Param_FIT = [[] for _ in range(self.nb_jauges)]
        self.model_pic_fit = None
        self.y_fit_start = None

        self.J = [0 for _ in range(self.nb_jauges)]          # nombre de pics / jauge
        self.X_s = [None for _ in range(self.nb_jauges)]
        self.X_e = [None for _ in range(self.nb_jauges)]
        self.z1 = [None for _ in range(self.nb_jauges)]
        self.z2 = [None for _ in range(self.nb_jauges)]
        self.Zone_fit = [None for _ in range(self.nb_jauges)]
        self.list_text_pic = [[] for _ in range(self.nb_jauges)]
        self._refresh_pic_table()

        self.X0 = 0
        self.Y0 = 0
        self.selected_file = None

        # Bits de contrôle
        self.bit_print_fit_T = False
        self.bit_fit_T = False
        self.bit_fit = [False for _ in range(self.nb_jauges)]
        self.bit_modif_jauge = False
        self.is_loading_gauge = False
        self.bit_filtre = False
        self.bit_plot_fit = [False for _ in range(self.nb_jauges)]

        # Indice de jauge
        self.index_jauge = -1

    def DEBUG_SPECTRUM(self):
        self.CLEAR_ALL()
        self.bit_bypass=True
        self.LOAD_Spectrum()
        self.bit_bypass=False
#########################################################################################################################################################################################
#? COMMANDE SAVE
    def CREAT_new_CEDd_fit(self):
        self.Spectrum_save = copy.deepcopy(self.Spectrum)
        New_CEDd = self._build_new_cedd(fit=True)

        if New_CEDd.Summary.empty:
            print("Loop STOP")
            return

        print("Save as ", os.path.basename(self.loaded_filename_oscilo), "in folder ", folder_CEDd)
        print(New_CEDd.Summary)
        self._finalize_new_cedd(New_CEDd, folder_CEDd)

    def CREAT_new_CEDd(self):
        """Crée un CEDd sans lancer de fit automatique."""
        self.Spectrum_save = copy.deepcopy(self.Spectrum)
        New_CEDd = self._build_new_cedd(fit=False)
        self._finalize_new_cedd(New_CEDd, folder_CEDd)
        print("Created CEDd (no fit) as ", New_CEDd.CEDd_path)
        self.text_box_msg.setText("CEDd sans fit auto chargé.\nLancer ensuite le multi-fit.")

#########################################################################################################################################################################################
#? COMMANDE PIC
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
            self.J[self.index_jauge]+=1 
            self.index_pic_select=i
            self.Param0[self.index_jauge][self.index_pic_select]=[self.X0,self.Y0,float(p.sigma[0]),np.array([float(x[0]) for x in p.coef_spe]),str(p.model_fit) ]
            self.bit_bypass=True
            self.Replace_pic()
            self.bit_bypass=False
        
        self.Spectrum.bit_fit=False
        self.Spectrum.Gauges[self.index_jauge].bit_fit =False
        self.select_pic()
            
    def Click_Zone(self):
        """Définit / efface la zone de fit pour la jauge courante, sans tracés Matplotlib."""
        if self.index_jauge == -1 or self.Spectrum is None:
            print("Zone bug resolve : no gauge selected")
            return

        j = self.index_jauge
        x_click = float(self.X0)

        # Cas 1 : aucune zone définie -> on pose X_s
        if self.Zone_fit[j] is None and self.X_s[j] is None:
            self.X_s[j] = x_click
            self.text_box_msg.setText("Zone 1")
            return

        # Cas 2 : X_s existe mais pas X_e -> on pose X_e + on calcule Zone_fit
        if self.Zone_fit[j] is None and self.X_e[j] is None:
            if x_click < self.X_s[j]:
                self.text_box_msg.setText("X_end < X_start -> ignored")
                return

            self.X_e[j] = x_click
            self.Zone_fit[j] = np.where(
                (self.Spectrum.wnb >= self.X_s[j]) &
                (self.Spectrum.wnb <= self.X_e[j])
            )[0]

            # On mémorise la zone dans la jauge
            self.Spectrum.Gauges[j].indexX = self.Zone_fit[j]

            # Si c'est la jauge 0 et que la box "zone_spectrum" est cochée,
            # on applique aussi au spectre global
            if j == 0 and self.zone_spectrum_box.isChecked():
                self.Spectrum.indexX = self.Zone_fit[0]
                self.Spectrum.x_corr = self.Spectrum.wnb[self.Zone_fit[0]]
            else:
                # zone_jauge seule, on laisse indexX global comme il est
                pass

            # Re-traitement des données avec la nouvelle zone
            self.Baseline_spectrum()
            self.text_box_msg.setText("Zone 2")
            return

        # Cas 3 : une zone est déjà définie -> on la supprime
        self.X_s[j] = None
        self.X_e[j] = None
        self.Zone_fit[j] = None
        self.Spectrum.Gauges[j].indexX = None

        if j == 0 and self.zone_spectrum_box.isChecked():
            self.Spectrum.indexX = None
            self.Spectrum.x_corr = self.Spectrum.wnb

        self.Baseline_spectrum()
        self.text_box_msg.setText("Zone Clear")

    def f_pic_fit(self):
        n_sigma = int(self.sigma_pic_fit_entry.value())
        ctr = self.Param0[self.index_jauge][self.index_pic_select][0]
        sig = self.Param0[self.index_jauge][self.index_pic_select][2]

        indexX = np.where(
            (self.Spectrum.x_corr > ctr - sig * n_sigma)
            & (self.Spectrum.x_corr < ctr + sig * n_sigma)
        )[0]
        x_sub = np.array(self.Spectrum.x_corr[indexX])

        self.y_fit_start = self.y_fit_start - self.list_y_fit_start[self.index_jauge][self.index_pic_select]


        y_sub = (self.Spectrum.y_corr - self.y_fit_start)[indexX]

        inter = float(self.inter_entry.value())

        X_pic = CL.Pics(
            name=self.Nom_pic[self.index_jauge][self.index_pic_select],
            ctr=ctr,
            ampH=self.Param0[self.index_jauge][self.index_pic_select][1],
            coef_spe=self.Param0[self.index_jauge][self.index_pic_select][3],
            sigma=sig,
            inter=inter,
            Delta_ctr=ctr * inter / 10,
            model_fit=self.Param0[self.index_jauge][self.index_pic_select][4],
        )
        try:
            out = X_pic.model.fit(y_sub, x=x_sub)
            self.Param0[self.index_jauge][self.index_pic_select][:4] = X_pic.Out_model(out)
            return out, X_pic, indexX, True
        except Exception as e:
            print("f_pic_fit ERROR pic not change:", e)
            return X_pic.model, X_pic, indexX, False

    def _build_full_y_plot(self, y_partial, indexX):
        """Return a y-array matching ``self.Spectrum.wnb`` length.

        ``y_partial`` may only cover a subset of x (``indexX``). This helper
        re-injects those values into a zeroed array aligned with the full
        spectrum to keep downstream sums (``list_y_fit_start``) coherent.
        """

        S = self.Spectrum
        if S is None or not hasattr(S, "wnb") or S.wnb is None:
            return np.asarray(y_partial) if y_partial is not None else np.array([])

        total_len = len(S.wnb)
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
            self.curve_spec_pic_select.setData([], [])
            self.curve_zoom_pic.setData([], [])
        except Exception:
            pass
        

    def _CED_multi_fit(self):
        """
        Multi-fit chaîné :
        - on part d'un CED (self.RUN) déjà chargé,
        - index_start / index_stop définis dans des spinbox,
        - pour i > index_start, on copie les Gauges du spectre i-1 vers i
        pour utiliser le fit précédent comme point de départ.
        - Affiche une barre de progression avec bouton Stop.
        """
        if self.RUN is None:
            print("no CED X LOAD")
            self.text_box_msg.setText("Multi-fit : aucun CEDd chargé.")
            return

        # On s'assure que le spectre courant est sauvegardé
        try:
            self.RUN.Spectra[self.index_spec] = self.Spectrum
        except Exception as e:
            print("Error saving current spectrum in RUN:", e)

        index_start = int(self.index_start_entry.value())
        index_stop  = int(self.index_stop_entry.value())

        # clamp...
        index_start = max(0, index_start)
        index_stop  = min(len(self.RUN.Spectra) - 1, index_stop)


        self.index_start_entry.setValue(index_start)
        self.index_stop_entry.setValue(index_stop)
        self.apply_from_spin()


        if index_start > index_stop:
            self.text_box_msg.setText("Multi-fit : index_start > index_stop.")
            print("Index start must be <= index stop.")
            return

        # Vérif : il faut des Gauges + pics sur le premier spectre
        ref_spec = self.RUN.Spectra[index_start]
        if not getattr(ref_spec, "Gauges", None) or len(ref_spec.Gauges) == 0:
            self.text_box_msg.setText(
                "Multi-fit : le spectre de départ n'a pas de jauges/pics.\n"
                "Configurer au moins une jauge + pics sur ce spectre puis relancer."
            )
            print("No Gauges on starting spectrum for multi-fit chain.")
            return

        # Message dans la zone de texte
        self.text_box_msg.setText(
            f"Multi-fit chaîné en cours : spectres {index_start} → {index_stop}."
        )

        # On garde l'index initial pour revenir dessus ensuite
        original_index_spec = self.index_spec

        # ----- DIALOGUE DE PROGRESSION -----
        n_tot = index_stop - index_start + 1
        dlg = ProgressDialog(             # tu peux passer des figs si tu veux
            cancel_text="Stop",
            value=0,
            parent=self
        )
        dlg.setLabelText(f"Multi-fit : spectres {index_start} → {index_stop}")
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()
    
        canceled = False

        self._multi_fit_running = True
        self._multi_fit_fast_mode = bool(getattr(self, "chk_multi_fit_fast", None) and self.chk_multi_fit_fast.isChecked())
        failed_indices = []

        # ----- BOUCLE SUR LES SPECTRES -----
        try:
            for k, i in enumerate(range(index_start, index_stop + 1), start=1):

                
                # Vérifier si l'utilisateur a cliqué sur Stop
                if dlg.wasCanceled():
                    canceled = True
                    print("Multi-fit: canceled by user.")
                    break

                if not self._multi_fit_fast_mode:
                    t = self.RUN.Time_spectrum[i]
                    # Lignes verticales
                    self.line_t_P.setPos(t)
                    self.line_t_dPdt.setPos(t)
                    self.line_t_sigma.setPos(t)
                #print(f"Multi-fit : spectre {i}/{index_stop}")

                # Mise à jour du texte et de la barre de progression
                if not self._multi_fit_fast_mode:
                    dlg.setLabelText(f"Fit du spectre {i} / {index_stop}")
                percent = int(100 * k / n_tot)
                dlg.setValue(percent)
                progress_step = max(1, int((index_stop - index_start) / 10))
                if not self._multi_fit_fast_mode or (k % progress_step == 0):
                    QApplication.processEvents()

                original_curr_spec = copy.deepcopy(self.RUN.Spectra[i])

                # Pour i > index_start : on copie les Gauges du spectre précédent
                if i > index_start:
                    prev_spec = self.RUN.Spectra[i - 1]
                    curr_spec = self.RUN.Spectra[i]

                    # On écrase les Gauges actuelles par une copie profonde du spectre précédent
                    curr_spec.Gauges = copy.deepcopy(prev_spec.Gauges)

                    # On remet le modèle à None pour forcer LOAD_Spectrum à reconstruire
                    curr_spec.model   = None
                    curr_spec.bit_fit = False

                # Chargement du spectre i dans l'UI + reconstruction de Param0
                self.bit_bypass = True      # pour éviter trop de boîtes de dialogue
                try:
                    self.LOAD_Spectrum(Spectrum=self.RUN.Spectra[i])
                except Exception as e:
                    print(f"Error loading spectrum {i}:", e)
                    continue   # on passe au suivant, mais on ne stoppe pas toute la boucle
                
                # Fit automatique avec les paramètres initiaux venant de prev_spec
                try:
                    self.bit_bypass = True
                    self.FIT_lmfitVScurvfit(run_asynchron=False)
    
                except Exception as e:
                    print(f"Error during fit on spectrum {i}:", e)
                    failed_indices.append(i)
                    self.Spectrum = copy.deepcopy(original_curr_spec)
                self.bit_bypass = False

                # Si le fit n'a pas convergé, on garde le spectre original
                # (sans écraser par le dernier fit convergé).
                if not bool(getattr(self.Spectrum, "bit_fit", False)):
                    failed_indices.append(i)
                    self.Spectrum = copy.deepcopy(original_curr_spec)

                # On sauvegarde le spectre (fitté ou original) dans RUN
                self.RUN.Spectra[i] = copy.deepcopy(self.Spectrum)

        finally:
            # On ferme le dialogue quoi qu'il arrive
            dlg.close()

        # ----- APRÈS LA BOUCLE : Corr_Summary ciblé + REFRESH -----
        try:
            self._mark_summary_dirty(range(index_start, index_stop + 1))
            self._flush_summary_dirty()
        except Exception as e:
            print("Error in targeted RUN.Corr_Summary:", e)

        # On revient sur le spectre initial dans l'UI
        self.index_spec = original_index_spec
        self.spinbox_spec_index.blockSignals(True)
        self.spinbox_spec_index.setValue(self.index_spec)
        self.spinbox_spec_index.blockSignals(False)
        self.Spectrum   = self.RUN.Spectra[self.index_spec]
        self.LOAD_Spectrum(Spectrum=self.Spectrum)
        self.REFRESH()

        self._set_movie_controls_enabled(bool(self.RUN.time_movie))

        self._multi_fit_fast_mode = False
        self._multi_fit_running = False
        if canceled:
            self.text_box_msg.setText("Multi-fit chaîné interrompu par l'utilisateur.")
        elif failed_indices:
            failed_unique = sorted(set(failed_indices))
            self.text_box_msg.setText(
                "Multi-fit terminé avec échecs de convergence sur spectres : "
                + ", ".join(str(x) for x in failed_unique)
            )
        else:
            self.text_box_msg.setText("Multi-fit chaîné terminé.")
    def apply_from_spin(self):
        """Synchronise les spinbox start/stop avec la région de fit en mode bypass."""
        previous_bypass = self.bit_bypass
        self.bit_bypass = True
        try:
            if hasattr(self, "_apply_fit_from_spin"):
                self._apply_fit_from_spin()
        finally:
            self.bit_bypass = previous_bypass


    # ============================================================
    # Helpers graphiques / PyQtGraph

    def add_zone(self):
        return print("A CODER")
    
    def dell_zone(self):
        return print("A CODER")
    
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow(folder_start)
    window.show()
    sys.exit(app.exec_())
