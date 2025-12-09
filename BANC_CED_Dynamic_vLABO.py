# BANC CED Dynamic 

import copy
import io
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pynverse import inversefunc
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGridLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from scipy.optimize import curve_fit

from theme_config import STYLE_TEMPLATE, THEMES
from ui_layout import UiLayoutMixin
from spectrum_view import SpectrumViewMixin
from ddac_view import DdacViewMixin

pg.setConfigOptions(
    antialias=True,          # courbes lissées
    useOpenGL=False,         # tu peux tester True si ta carte suit bien
    #downsample=True,         # évite de tracer chaque point si énorme -> bug 
)




Setup_mode = False

folder_start=r"F:\Aquisition_Banc_CEDd"
folder_CEDd=r"F:\Aquisition_Banc_CEDd\Fichier_CEDd"#r"C:\Users\dDAC-LHPS\Desktop\PROG_FLO_dDAC\Aquisition_Banc_CEDd\Fichier_CEDd"

from Bibli_python import CL_FD_Update as CL

from Bibli_python import Oscilloscope_LeCroy_vLABO as Oscilo

def plot_clear(plot):
    try:
        if plot is None:
            return
        if type(plot) is list:
            for p in plot:
                p.remove()
        else:
            plot.remove()
    except Exception as e:
        print(e)

def configurer_axes(ax):
    ax.set_facecolor("#2b2b2b")
    ax.tick_params(colors="#e0e0e0", labelcolor="#e0e0e0")
    ax.tick_params(which="minor", colors="#e0e0e0", labelcolor="#e0e0e0") 
    ax.title.set_color("#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#555555")

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QProgressBar, QWidget
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SciAxis(pg.AxisItem):
    """Axe qui affiche les ticks en notation scientifique (1.23e+04)."""

    def tickStrings(self, values, scale, spacing):
        # values est la liste des positions de ticks en coordonnées "données"
        # On ne touche PAS aux données, on ne fait que formatter l'affichage.
        return [f"{v:.1e}" for v in values]


class ProgressDialog(QDialog):
    """
    Dialogue interactif pour afficher un spectre + bouton Stop.
    Compatible avec ton usage dans le multi-fit.
    """

    def __init__(self, figs=None, cancel_text="Annuler",
                 value=0, Gauges=None, parent=None):
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

        # -------- FIGURES MATPLOTLIB --------
        if figs:
            for fig in figs:
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)

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


class EditableDelegate(QStyledItemDelegate):
    """A delegate that allows for cell editing"""

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        return editor

@dataclass
class RunViewState:
    """État graphique associé à un CEDd (remplace les listes parallèles)."""

    ced: CL.CEDd
    color: str
    time: list = field(default_factory=list)
    spectre_number: list = field(default_factory=list)
    curves_P: list = field(default_factory=list)
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


class MainWindow(QMainWindow, UiLayoutMixin, SpectrumViewMixin, DdacViewMixin):
    def __init__(self, folder_start=None):
        super().__init__()

        self.current_theme = "dark"

        # --- Variables "métier" de base ---
        self.Spectrum = None
        self.valeurs_boutons = [True,True, True,False,True,True,True]
        self.name_boutons= ["dP\dt","T","Piézo","Image Correlation","M2R","use Movie file","print P"]
        self.liste_chemins_fichiers = []
        self.liste_objets = []
        self.dossier_selectionne=folder_CEDd #r"F:\Aquisition_Banc_CEDd\Fichier_CEDd"
        self.Spectrum_save=None
        self.CEDd_save=None
        self.Gauge_select=None
        self.data_Spectro=None
        self.RUN = None

        # États init pour éviter les accès à des attributs non initialisés
        self.bit_dP = 0
        self.Pstart = 0.0
        self.Pend = 0.0
        self.tstart = 0.0
        self.tend = 0.0
        self.x1 = 0.0
        self.y1 = 0.0
        self.x5 = 0.0
        self.y3 = 0.0
        self.x_clic = 0.0
        self.y_clic = 0.0

        # Chemin de base
        if folder_start is None:
            folder_start = r"E:\Aquisition_Banc_CEDd"
        self.folder_start = folder_start

        # --- Config fenêtre ---
        self._setup_main_window()

        # --- Layout central ---
        self.grid_layout = QGridLayout()
        central_widget = QWidget()
        central_widget.setLayout(self.grid_layout)
        self.setCentralWidget(central_widget)

        # --- Construction des différentes sections UI ---
        self._setup_file_box()           # (4, 0) -> file_spectro/oscilo/movie
        self._setup_tools_tabs()         # (0, 1)
        self._setup_spectrum_box()       # (0, 2)
        self.line_dy_zero = self.pg_dy.addLine(y=0)
        self._setup_ddac_box()           # (0, 3)
        self._setup_file_gestion()       # (2, 0)
        self._setup_python_kernel()      # (2, 2)
        self._setup_gauge_info()         # (1, 2)
        self._setup_text_box_msg()       # (1, 0)
        self._setup_layout_stretch()

        self._apply_theme(self.current_theme)

        # Autres attributs divers
        self.viewer = None
        self.bit_c_reduite = True

        # Nouvelle bibliothèque d'états par CEDd (clé stable -> RunViewState)
        self.runs = {}
        self.current_run_id = None

        self.file_index_map = {}          # key = index dans self.liste_fichiers, value = run_id
        self.liste_objets = []  # liste des CEDd chargés (legacy)
        self.index_select = -1

        # Initialisation logique légère
        self.load_latest_file()
        self.f_gauge_select()

        # Setup debug optionnel
        if Setup_mode is True:
            self._run_setup_mode()
            print("setup mode RUN")

    # ==================================================================
    # ===============   UTILITAIRES ÉTAT DE RUN   ======================
    # ==================================================================
    def _get_theme(self, name: Optional[str] = None):
        return THEMES.get(name or self.current_theme, THEMES["dark"])

    def _mk_pen(self, spec):
        if isinstance(spec, dict):
            return pg.mkPen(**spec)
        return pg.mkPen(spec)

    def _build_stylesheet(self, theme):
        try:
            return STYLE_TEMPLATE.safe_substitute(
                window=theme["window"],
                background=theme["background"],
                text=theme["text"],
                accent=theme["accent"],
                accent_hover=theme["accent_hover"],
                accent_pressed=theme["accent_pressed"],
                input_background=theme["input_background"],
                selection=theme["selection"],
                selection_text=theme["selection_text"],
                menu_background=theme["menu_background"],
                button_text=theme["button_text"],
            )
        except Exception as exc:
            print("Failed to build stylesheet:", exc)
            return ""

    def _apply_plot_item_theme(self, plot_item, theme):
        """Applique le thème à un objet PyQtGraph :
        - GraphicsLayoutWidget
        - PlotItem
        - ViewBox
        """
        if plot_item is None:
            return

        # 1) GraphicsLayoutWidget (pg.GraphicsLayoutWidget)
        if isinstance(plot_item, pg.GraphicsLayoutWidget):
            plot_item.setBackground(theme["plot_background"])
            return

        # 2) ViewBox direct (ex: self.pg_text)
        if isinstance(plot_item, pg.ViewBox):
            # ViewBox possède bien setBackgroundColor
            plot_item.setBackgroundColor(theme["plot_background"])
            return

        # 3) PlotItem classique
        # (pg.PlotItem, ce que renvoie addPlot)
        vb = plot_item.getViewBox()
        if vb is not None and hasattr(vb, "setBackgroundColor"):
            vb.setBackgroundColor(theme["plot_background"])

        axis_pen = self._mk_pen(theme["axis_pen"])
        text_pen = self._mk_pen(theme["text"])

        for name in ("bottom", "left", "right", "top"):
            axis = plot_item.getAxis(name)
            if axis is not None:
                axis.setPen(axis_pen)
                axis.setTextPen(text_pen)

        plot_item.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def _apply_theme(self, theme_name: str):
        theme = self._get_theme(theme_name)
        self.current_theme = theme_name if theme_name in THEMES else "dark"

        # --- Bouton toggle light/dark ---
        if hasattr(self, "theme_toggle_button"):
            self.theme_toggle_button.blockSignals(True)
            self.theme_toggle_button.setChecked(self.current_theme == "light")
            self.theme_toggle_button.setText(
                "Light mode" if self.current_theme == "light" else "Dark mode"
            )
            self.theme_toggle_button.blockSignals(False)

        # --- Stylesheet Qt global ---
        self.setStyleSheet(self._build_stylesheet(theme))

        # ================== SPECTRUM BOX ==================
        if hasattr(self, "pg_spec"):
            self.pg_spec.setBackground(theme["plot_background"])

            for plot in (
                self.pg_zoom,
                self.pg_baseline,
                self.pg_fft,
                self.pg_dy,
                self.pg_spectrum,
            ):
                self._apply_plot_item_theme(plot, theme)

            pens = theme["pens"]
            self.curve_spec_data.setPen(self._mk_pen(pens["spectrum_data"]))
            self.curve_spec_fit.setPen(self._mk_pen(pens["spectrum_fit"]))
            self.curve_spec_pic_select.setBrush(pg.mkBrush(pens["spectrum_pic_brush"]))
            self.curve_dy.setPen(self._mk_pen(pens["dy"]))
            self.line_dy_zero.setPen(self._mk_pen(pens["zero_line"]))
            self.curve_baseline_brut.setPen(self._mk_pen(pens["baseline_brut"]))
            self.curve_baseline_blfit.setPen(self._mk_pen(pens["baseline_fit"]))
            self.curve_fft.setPen(self._mk_pen(pens["fft"]))
            self.curve_zoom_data.setPen(self._mk_pen(pens["zoom_data"]))
            self.curve_zoom_data_brut.setPen(self._mk_pen(pens["zoom_data_brut"]))
            self.curve_zoom_pic.setBrush(pg.mkBrush(pens["zoom_pic_brush"]))
            self.vline.setPen(self._mk_pen(pens["selection_line"]))
            self.hline.setPen(self._mk_pen(pens["selection_line"]))
            # cross_zoom est un ScatterPlotItem : on joue sur le brush
            self.cross_zoom.setBrush(pg.mkBrush(pens["cross_zoom"]))
            self.pg_text_label.setColor(pens["text_item"])

        # ================== dDAC BOX ==================
        if hasattr(self, "pg_ddac"):
            self.pg_ddac.setBackground(theme["plot_background"])

            for plot in (
                self.pg_P,
                self.pg_dPdt,
                self.pg_sigma,
                self.pg_movie,
                self.pg_dlambda,
            ):
                self._apply_plot_item_theme(plot, theme)

            pens = theme["pens"]
            line_pen = self._mk_pen(pens["line_t"])
            self.line_t_P.setPen(line_pen)
            self.line_t_dPdt.setPen(line_pen)
            self.line_t_sigma.setPen(line_pen)
            self.line_nspec.setPen(line_pen)
            self.line_p0.setPen(self._mk_pen(pens["baseline_time"]))

            zone_pen = self._mk_pen(pens["zone_movie"])
            for line in self.zone_movie_lines:
                line.setPen(zone_pen)

            scatter_pen = self._mk_pen(pens["scatter"])
            # scatter_* sont des ScatterPlotItem : on met le "pen" (contour)
            self.scatter_P.setPen(scatter_pen)
            self.scatter_dPdt.setPen(scatter_pen)
            self.scatter_sigma.setPen(scatter_pen)
            self.scatter_dlambda.setPen(scatter_pen)

        # ================== TEXT VIEWBOX (dDAC) ==================
        if hasattr(self, "pg_text"):
            # pg_text est un ViewBox, pas un PlotItem : on met juste le fond
            self.pg_text.setBackgroundColor(theme["plot_background"])

        # ================== POLICES DES AXES ==================
        font = QFont("Segoe UI", 11)

        def _set_axes_font(plot_item):
            if plot_item is None:
                return
            for name in ("bottom", "left"):
                axis = plot_item.getAxis(name)
                if axis is not None:
                    axis.setTickFont(font)

        for plot in (
            getattr(self, "pg_zoom", None),
            getattr(self, "pg_baseline", None),
            getattr(self, "pg_fft", None),
            getattr(self, "pg_dy", None),
            getattr(self, "pg_spectrum", None),
            getattr(self, "pg_P", None),
            getattr(self, "pg_dPdt", None),
            getattr(self, "pg_sigma", None),
            getattr(self, "pg_dlambda", None),
        ):
            _set_axes_font(plot)

    def _toggle_theme(self, checked: bool):
        self._apply_theme("light" if checked else "dark")

    def _stylize_plot(self, plot_item, x_label=None, y_label=None, show_grid=True):
        if plot_item is None:
            return
        if x_label:
            plot_item.setLabel('bottom', x_label)
        if y_label:
            plot_item.setLabel('left', y_label)
        if show_grid:
            # utilise alpha de ton thème si tu veux
            plot_item.showGrid(x=True, y=True, alpha=self._get_theme()["grid_alpha"])
        plot_item.setMouseEnabled(x=True, y=True)


    def _get_run_id(self, ced):
        """Retourne une clé stable pour un CEDd donné."""

        if hasattr(ced, "CEDd_path") and ced.CEDd_path:
            return ced.CEDd_path
        return f"memory_{id(ced)}"


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
            self.grid_layout.removeWidget(self.AddBox)
            self.grid_layout.removeWidget(self.promptBox)

            self.grid_layout.addWidget(self.SpectraBox, 0, 2, 1, 1)
            self.grid_layout.addWidget(self.AddBox,     1, 2, 1, 1)
            self.grid_layout.addWidget(self.promptBox,  2, 2, 1, 1)

        else:
            # bouton relâché -> on cache le kernel
            self.python_kernel_button.setText("Show Python Kernel")
            self.promptBox.hide()

            self.grid_layout.removeWidget(self.SpectraBox)
            self.grid_layout.removeWidget(self.AddBox)
            # promptBox peut rester dans la grille mais caché

            self.grid_layout.addWidget(self.SpectraBox, 0, 2, 2, 1)
            self.grid_layout.addWidget(self.AddBox,     2, 2, 1, 1)

    # ==================================================================
    # ===============   SETUP MODE POUR DEBUG  =========================
    # ==================================================================
    def _run_setup_mode(self):
        self.CREAT_new_Spectrum()
        self.Gauge_type_selector.setCurrentIndex(0)
        self.f_gauge_select()
        self.bit_load_jauge = True
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


#? CALVIER COMMANDE CONTROLE

    def keyPressEvent(self, event):# - - - COMMANDE CLAVIER - - -# 
        key = event.key() 
        modifiers = event.modifiers()
        f,param=None,None
        Box,name_f=False,None

        if self.viewer is not None:
            if self.focusWidget() == self.viewer:
                return print("focus in Lecroy")
        if Setup_mode is True:
            print(key)
        
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier: # EXECTUE CODE PYTHON
            self.execute_code()
            return

        elif key == Qt.Key_L and modifiers & Qt.ShiftModifier: # FENETRE LECROY
            self.viewer = Oscilo.OscilloscopeViewer(folder=os.path.join(folder_start,"Aquisition_LECROY_Banc_CEDd"))
            self.viewer.show()

        elif key == Qt.Key_B: # BASE LINE
            f=self.Baseline_spectrum
            name_f="Baseline_spectrum"
    
        elif key == Qt.Key_E and modifiers & Qt.ShiftModifier:  #New CEDd 
            f=self.CREAT_new_CEDd
            name_f="CREAT new file CEDd"
            Box=True

        elif key == Qt.Key_K and modifiers & Qt.ShiftModifier:  #New CEDd 
            self.bit_bypass=True
            f=self.CLEAR_CEDd
            Box=True
            name_f="CLEAR CEDd"
        
        
        elif key == Qt.Key_T and modifiers & Qt.ShiftModifier: #New Spectrum
            f=self.CREAT_new_Spectrum
            name_f="CREAT new Spectrum"
            Box=True

        elif key == Qt.Key_A and modifiers & Qt.ShiftModifier: # RUN FIT TOTAL
            f=self.FIT_lmfitVScurvfit
            Box=True
            name_f="FIT lmfit VS curvfit"
            
        elif key == Qt.Key_C : #CONFIRME PIC
            f=self.Click_Confirme
        
        elif key == Qt.Key_Y and modifiers & Qt.ShiftModifier : # AUTOPIX
            f=self.Auto_pic
            #name_f="Auto pic"
        
        elif key == Qt.Key_P: #RESIZE WINDOW
            f=self.toggle_colonne

        elif key == Qt.Key_Z :
            if modifiers & Qt.ShiftModifier: #ZONE 
                f=self.Click_Zone
            else:
                f=self.f_zone_movie
        elif key == Qt.Key_Return and modifiers & Qt.ShiftModifier : 
            f=self.Click_Clear
            
        elif key == Qt.Key_O and modifiers & Qt.ShiftModifier : #DELL JAUGE
            f=self.Dell_Jauge
        
        elif key == Qt.Key_U and modifiers & Qt.ShiftModifier: #Delle last pic
            f=self.Undo_pic

        elif key == Qt.Key_R :
            if modifiers & Qt.ShiftModifier:
                f=self.Replace_pic_fit
            else:
                f=self.Replace_pic

        elif key == Qt.Key_W :
            f=self.Undo_pic_select

        elif key == Qt.Key_N and modifiers & Qt.ShiftModifier: 
            f=self.REFRESH
            Box=True
            name_f="Refresh data CEDd"
        
        elif key == Qt.Key_S :
            try:
                if modifiers & Qt.ShiftModifier:
                    self.Spectrum.Calcul_study(mini=True)
                else:
                    self.Spectrum.Calcul_study(mini=False)
            except Exception:
                e = traceback.format_exc()
                self.Print_error(e)
        
        elif key == Qt.Key_X and modifiers & Qt.ShiftModifier:
            f=self.SAVE_CEDd
            Box=True
            name_f="Save CEDd"
        elif key == Qt.Key_J and modifiers & Qt.ShiftModifier:
            f=self.ADD_gauge
        
        elif key == Qt.Key_0 :
            f=self.f_lambda0

        elif key == Qt.Key_Q :
            self.select_clic_box.setChecked(not self.select_clic_box.isChecked())

        elif key == Qt.Key_F :
            self.fit_start_box.setChecked(not self.fit_start_box.isChecked())

        elif key == Qt.Key_M :
            self.movie_select_box.setChecked(not self.movie_select_box.isChecked())

        elif key == Qt.Key_H :
            self.spectrum_select_box.setChecked(not self.spectrum_select_box.isChecked())

        if f is not None:
            if Box==True:
                self.Box_loading(fonction=f,name_f=name_f)
            else:
                try:
                    f()
                    if name_f is not None:
                        self.text_box_msg.setText(name_f+"SUCCES idTouche:" + str(key))
                except Exception:
                    e = traceback.format_exc()
                    self.Print_error(e)
            self.bit_bypass=False

    def Box_loading(self,fonction,name_f):
        # Création de la QMessageBox
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowTitle(name_f+"\n En cours")
        self.msg_box.setText("ATTENTION \n Sur un malentendu ça peu planter")
        self.msg_box.setStandardButtons(QMessageBox.NoButton)
        self.msg_box.show()
        try:
            fonction()
            text=name_f+"\n Terminée"
        except Exception:
            e = traceback.format_exc()
            text='ERROR:'+ str(e)+'  \n check in console variable'
            """Actions lorsque la tâche est terminée."""
        self.msg_box.setText(text+"\n Press Entrée for quit")
        ok_button = self.msg_box.addButton(QMessageBox.Ok)
        ok_button.setText("OK (Entrée)")
        ok_button.setShortcut("Return")

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

    def _refresh_spectrum_view(self):
        """Rafraîchit l'ensemble des vues spectrales PyQtGraph à partir du spectre courant.

        Les courbes de spectre, fit global, dérivée dY, baseline et FFT sont mises
        à jour sur leurs plots respectifs (pg_spectrum, pg_dy, pg_baseline, pg_fft).
        La fonction ajuste aussi les limites de ViewBox et met à jour l'état local
        (drapeau d'initialisation des limites) sans modifier la logique métier du
        calcul du spectre.
        """
        S = self.Spectrum
        if S is None:
            # on vide tout
            self.curve_spec_data.setData([], [])
            self.curve_spec_fit.setData([], [])
            self.curve_dy.setData([], [])
            self.curve_baseline_brut.setData([], [])
            self.curve_baseline_blfit.setData([], [])
            self.curve_fft.setData([], [])
            self.curve_zoom_data.setData([], [])
            self.curve_zoom_data_brut.setData([], [])
            self.curve_zoom_pic.setData([], [])
            self.curve_spec_pic_select.setData([], [])
            return

        # --- On mémorise les X/Y du spectre principal pour les autres graphes ---
        x_spec = None
        y_spec = None

        # 1) Spectre corrigé
        if hasattr(S, "x_corr") and hasattr(S, "y_corr") and S.x_corr is not None and S.y_corr is not None:
            x = np.asarray(S.x_corr)
            y = np.asarray(S.y_corr)
            self.curve_spec_data.setData(x, y)

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
            self.curve_spec_data.setData([], [])
            # si pas de x_corr, on peut prendre wnb comme X de référence
            if hasattr(S, "wnb") and S.wnb is not None and hasattr(S, "spec") and S.spec is not None:
                x_spec = np.asarray(S.wnb)
                y_spec = np.asarray(S.spec)

        # 2) Fit total (self.y_fit_start)
        if self.y_fit_start is not None and hasattr(S, "wnb"):
            self.curve_spec_fit.setData(S.wnb, self.y_fit_start)
        else:
            self.curve_spec_fit.setData([], [])
        
        # 2 bis) Mise à jour du fond avec *tous* les pics
        self._update_gauge_peaks_background()

        # 3) dY : X = ceux du spectre, Y = dY
        if hasattr(S, "dY") and S.dY is not None:
            self.curve_dy.setData(S.X if hasattr(S, "X") and S.X is not None else S.wnb, S.dY)
        else:
            self.curve_dy.setData([], [])

        # 4) Baseline : brut + blfit
        if hasattr(S, "wnb") and hasattr(S, "spec") and S.wnb is not None and S.spec is not None:
            self.curve_baseline_brut.setData(S.wnb, S.spec)

            # limites du graphe brut : X et Y du brut
            vb_base = self.pg_baseline.getViewBox()
            self._set_viewbox_limits_from_data(vb_base, S.wnb, S.spec, padding=0.02)
        else:
            self.curve_baseline_brut.setData([], [])

        if hasattr(S, "blfit") and S.blfit is not None:
            if getattr(S, "indexX", None) is not None:
                self.curve_baseline_blfit.setData(S.wnb[S.indexX], S.blfit[S.indexX])
            else:
                self.curve_baseline_blfit.setData(S.wnb, S.blfit)
        else:
            self.curve_baseline_blfit.setData([], [])

        # 5) FFT : même X que le spectre, mais limites Y = amplitude FFT
        # (adapte les noms d'attributs FFT si besoin)
        if hasattr(S, "fft_amp") and S.fft_amp is not None:
            # Exemple : on trace FFT avec un axe fréquentiel S.fft_f
            if hasattr(S, "fft_f") and S.fft_f is not None:
                self.curve_fft.setData(S.fft_f, S.fft_amp)
                x_fft = np.asarray(S.fft_f)
            else:
                # si tu n'as pas d'axe de fréquence propre, tu peux simplement utiliser le même X que le spectre
                if x_spec is not None:
                    x_fft = x_spec
                    self.curve_fft.setData(x_fft, S.fft_amp)
                else:
                    x_fft = None
        else:
            self.curve_fft.setData([], [])
            x_fft = None

        # Limites du graphe FFT :
        # - X : mêmes que le spectre (si x_spec dispo)
        # - Y : amplitude FFT
        if x_spec is not None and hasattr(S, "fft_amp") and S.fft_amp is not None:
            vb_fft = self.pg_fft.getViewBox()
            # On force les limites X avec x_spec, et Y avec fft_amp
            # -> astuce : on donne x_data = x_spec, y_data = fft_amp
            self._set_viewbox_limits_from_data(vb_fft, x_spec, S.fft_amp, padding=0.02)

        # 6) Limites du graphe dY :
        # X : mêmes que le spectre (x_spec)
        # Y : dY
        if x_spec is not None and hasattr(S, "dY") and S.dY is not None:
            vb_dy = self.pg_dy.getViewBox()
            # même astuce : x_data = x_spec, y_data = dY
            self._set_viewbox_limits_from_data(vb_dy, x_spec, S.dY, padding=0.02)
            self.pg_dy.setLimits(xMin=x_spec[0], xMax=x_spec[-1])

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
        if self.bit_c_reduite:
            self.grid_layout.setColumnStretch(2, 5)
        else:
            self.grid_layout.setColumnStretch(2, 0)
        self.bit_c_reduite = not self.bit_c_reduite  # Inverser l'état

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

    def parcourir_dossier(self):
        # Fonction pour parcourir un dossier et afficher ses fichiers
        options = QFileDialog.Options()
        self.dossier_selectionne = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier", options=options)
        if self.dossier_selectionne:
            files_brute = os.listdir(self.dossier_selectionne)
            files = sorted(
                [f for f in files_brute],
                key=lambda x: os.path.getctime(os.path.join(self.dossier_selectionne, x)),
                reverse=True
            )
            self.liste_fichiers.clear()
            self.liste_fichiers.addItems(files)
            self.liste_chemins_fichiers = [
                os.path.join(self.dossier_selectionne, f) for f in files
            ]

#########################################################################################################################################################################################
#? COMMANDE CEDD
    def f_text_CEDd_print(self, t):
        """Met à jour le texte d'information dDAC dans le TextItem PyQtGraph."""
        dp = None
        if getattr(self, "tstart", None) is not None and getattr(self, "tend", None) is not None \
        and self.tstart != self.tend:
            dp = (self.Pstart - self.Pend) / (self.tstart - self.tend) * 1e-3

        txt = (
            f"$t_s$={self.x1*1e3:.3f}ms n°s={self.x5:.2f}\n\n"
            f"$t_i$={t*1e6:.3f}µs n°i={self.Num_im}\n\n"
            f"P={self.y1:.2f}GPa  T or dP/dt={self.y3:.2f} K or GPa/ms\n\n"
            f"dP/dt={0.0 if dp is None else dp:.3f} GPa/ms"
            )
        self.pg_text_label.setText(txt)

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
    


#########################################################################################################################################################################################
#? MOVIE 
#########################################################################################################################################################################################
#? COMMANDE FILE
    def f_select_directory(self,file_name,file_label,name,type_file=".asc"):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
         # Créer une instance de QFileDialog
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)  # Permet de sélectionner un seul fichier
        dialog.setNameFilter(f"Text Files (*{type_file});;All Files (*)")
        dialog.setViewMode(QFileDialog.Detail)  # Affiche les fichiers avec des détails comme la date, la taille, etc.

        # Définir le répertoire initial
        if file_name is None:
            dialog.setDirectory(self.folder_start)
        else:
            dialog.setDirectory(os.path.dirname(file_name))

         # Afficher la boîte de dialogue et récupérer le fichier sélectionné
        if dialog.exec_():
            file_name_bis = dialog.selectedFiles()[0]  # Récupère le premier fichier sélectionné
            file_name = file_name_bis  # Met à jour la variable locale
            file_label.setText(f"Selected directory: {os.path.basename(file_name)}")
            return file_name  # Retourn
    
    def f_data_spectro(self):
        if self.loaded_filename_spectro:
            try:
                self.data_Spectro = pd.read_csv(
                    self.loaded_filename_spectro,
                    sep=r"\s+",
                    header=None,
                    skiprows=43,
                    engine="python",
                )
            except Exception as e:
                self.text_box_msg.setText("ERROR FILE")
                return

        # Cas 2 colonnes : (X, Y empilés) → reshape en [X, Spec1, Spec2, ...]
        if len(self.data_Spectro.columns) == 2:
            wave = self.data_Spectro.iloc[:, 0]
            Iua = self.data_Spectro.iloc[:, 1]
            wave_unique = np.unique(wave)
            num_spec = len(wave) // len(wave_unique)

            if num_spec >= 1:
                Iua = Iua.values.reshape(num_spec, len(wave_unique)).T
                self.data_Spectro = pd.DataFrame(
                    np.column_stack([wave_unique, Iua]),
                    columns=[0] + [i + 1 for i in range(num_spec)],
                )

        # ---- MISE À JOUR DE LA SPINBOX DE SPECTRE ----
        # nb de spectres = nb de colonnes - 1 (colonne 0 = X)
        n_spec = max(0, self.data_Spectro.shape[1] - 1)

        if hasattr(self, "spinbox_spec_index"):
            self.spinbox_spec_index.blockSignals(True)
            # on garde un index 0-based pour index_spec
            self.spinbox_spec_index.setRange(0, max(0, n_spec - 1))
            self.spinbox_spec_index.setValue(0)
            self.spinbox_spec_index.blockSignals(False)

        # petit message d’info
        self.text_box_msg.setText(f"{n_spec} spectres chargés depuis le fichier spectro.")

    def select_spectro_file(self):
        self.loaded_filename_spectro=self.f_select_directory(self.loaded_filename_spectro,self.dir_label_spectro,"Spectrum",type_file=".asc")
        self.f_data_spectro()

    def select_oscilo_file(self):
        self.loaded_filename_oscilo=self.f_select_directory(self.loaded_filename_oscilo,self.dir_label_oscilo,"Oscilo",type_file="")

    def select_movie_file(self):
        self.loaded_filename_movie=self.f_select_directory(self.loaded_filename_movie,self.dir_label_movie,"Movie",type_file=".cine")

    def f_load_latest_file(self,folder,extend,dir_name,dir_label):
        dir_name,f=CL.Load_last(Folder=folder,extend=extend)
        dir_label.setText(f"Selected file: {f}")
        return dir_name

    def load_latest_file(self):
        folder_start_spectro=os.path.join(self.folder_start,"Aquisition_ANDOR_Banc_CEDd")
        # Lister tous les dossiers dans le dossier principal
        folders = [d for d in os.listdir(folder_start_spectro) if os.path.isdir(os.path.join(folder_start_spectro, d))]
        # Filtrer pour ne garder que ceux qui ont un format de date valide
        folders_dates = []
        for fold in folders:
            try:
                # Vérifie que le dossier est au format AAMMJJ en tentant de le convertir en date
                datetime.strptime(fold, "%y%m%d")
                folders_dates.append(fold)
            except ValueError:
                pass  # Ignorer les dossiers qui ne sont pas au format date
        # Trier les dossiers par date pour trouver le plus récent
        if folders_dates:
            latest_folder = max(folders_dates, key=lambda d: datetime.strptime(d, "%y%m%d"))
            latest_folder_spectro= os.path.join(folder_start_spectro, latest_folder)
            self.loaded_filename_spectro=self.f_load_latest_file(latest_folder_spectro,".asc",self.loaded_filename_spectro,self.dir_label_spectro)
            self.f_data_spectro()
        self.loaded_filename_oscilo=self.f_load_latest_file(os.path.join(self.folder_start,"Aquisition_LECROY_Banc_CEDd"),None,self.loaded_filename_oscilo,self.dir_label_oscilo)
        self.loaded_filename_movie=self.f_load_latest_file(os.path.join(self.folder_start,"Aquisition_PHANTOME_Banc_CEDd"),".cine",self.loaded_filename_movie,self.dir_label_movie)
        #self.loaded_filename_movie=self.f_load_latest_file(os.path.join(folder_start,"Fichier_CED"),".CLUpdate",self.loaded_filename_movie,self.dir_label_movie)

    def f_filter_files(self):
        filter_text = self.search_bar.text().lower()
        filtered_files = [f.lower() for f in self.liste_chemins_fichiers  if filter_text in f.lower()]
        self.liste_fichiers.clear()
        self.liste_fichiers.addItems(filtered_files)

#########################################################################################################################################################################################
#? COMMANDE PRINT
    def f_dell_lines(self):
        """Supprime les marqueurs de jauge (barres verticales) dans le plot PyQtGraph."""
        for item in self.lines:
            try:
                self.pg_spectrum.removeItem(item)
            except Exception:
                pass
        self.lines = []

    
    def f_p_move(self, J_select, value):
        """Met à jour la jauge J_select à partir d'une pression P (value)."""
        J_select.P = round(value, 3)

        if self.Spectrum is None or self.Spectrum.y_corr is None:
            M = 1.0
        else:
            M = float(max(self.Spectrum.y_corr))

        # calcul de λ à partir de P
        try:
            x = round(float(J_select.inv_f_P(value)), 3)
            self.deltalambdaP = x - J_select.lamb0
        except Exception:
            x = 0.0
            self.deltalambdaP = 0.0

        # Effacer anciens marqueurs
        self.f_dell_lines()

        # Dessiner une barre | pour chaque pic de la jauge
        for i, p in enumerate(J_select.pics):
            ctr = x + J_select.deltaP0i[i][0]
            y_top = M * J_select.deltaP0i[i][1]
            item = self.pg_spectrum.plot(
                [ctr, ctr],
                [0.0, y_top],
                pen=pg.mkPen(J_select.color_print[0], width=2)
            )
            self.lines.append(item)

        # Mettre à jour le spinbox λ (avec correction T)
        self.bit_bypass = True
        try:
            self.spinbox_x.setValue(x + self.deltalambdaT)
        finally:
            self.bit_bypass = False

        return J_select

    def spinbox_p_move(self, value):
        """Callback du spinbox_P : changement de P."""
        if self.bit_modif_PTlambda:
            return

        try:
            if self.bit_load_jauge:
                self.Gauge_select.lamb_fit = \
                    self.Gauge_select.inv_f_P(value) + self.deltalambdaT
                self.Gauge_select = self.f_p_move(self.Gauge_select, value)

            if self.bit_modif_jauge and self.Spectrum is not None and self.index_jauge >= 0:
                G = self.Spectrum.Gauges[self.index_jauge]
                G.lamb_fit = G.inv_f_P(value) + self.deltalambdaT
                self.Spectrum.Gauges[self.index_jauge] = self.f_p_move(G, value)

            self.save_value = value

        except Exception as e:
            print("spinbox_p_move error:", e)

    def f_x_move(self, J_select, value):
        """Met à jour la jauge J_select à partir de λ (value)."""
        J_select.lamb_fit = value

        if self.Spectrum is None or self.Spectrum.y_corr is None:
            M = 1.0
        else:
            M = float(max(self.Spectrum.y_corr))

        # P depuis λ - correction T
        try:
            J_select.P = round(float(J_select.f_P(value - self.deltalambdaT)), 3)
        except Exception:
            J_select.P = 0.0

        # Efface les anciens marqueurs
        self.f_dell_lines()

        for i, p in enumerate(J_select.pics):
            ctr = value + J_select.deltaP0i[i][0]
            y_top = M * J_select.deltaP0i[i][1]
            item = self.pg_spectrum.plot(
                [ctr, ctr],
                [0.0, y_top],
                pen=pg.mkPen(J_select.color_print[0], width=2)
            )
            self.lines.append(item)

        # MAJ spinbox P
        self.bit_modif_PTlambda = True
        try:
            self.spinbox_P.setValue(J_select.P)
        finally:
            self.bit_modif_PTlambda = False

        return J_select

    def spinbox_x_move(self, value):
        if self.bit_modif_PTlambda:
            return

        try:
            if self.bit_load_jauge:
                self.Gauge_select.lamb_fit = value
                self.Gauge_select = self.f_x_move(self.Gauge_select, value)

            if self.bit_modif_jauge and self.Spectrum is not None and self.index_jauge >= 0:
                G = self.Spectrum.Gauges[self.index_jauge]
                G.lamb_fit = value
                self.Spectrum.Gauges[self.index_jauge] = self.f_x_move(G, value)

            self.save_value = value
        except Exception as e:
            print("spinbox_x_move error:", e)

    def f_t_move(self, J_select, value):
        """Met à jour la jauge J_select à partir de T (value)."""
        J_select.T = round(value, 3)

        if self.Spectrum is None or self.Spectrum.y_corr is None:
            M = 1.0
        else:
            M = float(max(self.Spectrum.y_corr))

        # λ depuis T et P en utilisant CL.T_Ruby_by_P
        try:
            x = round(float(inversefunc(
                lambda x_: CL.T_Ruby_by_P(x_, P=J_select.P, lamb0R=J_select.lamb0),
                value
            )), 3)
            self.deltalambdaT = x - J_select.lamb0
        except Exception:
            x = 0.0
            self.deltalambdaT = 0.0

        self.f_dell_lines()
        for i, p in enumerate(J_select.pics):
            ctr = x + J_select.deltaP0i[i][0]
            y_top = M * J_select.deltaP0i[i][1]
            item = self.pg_spectrum.plot(
                [ctr, ctr],
                [0.0, y_top],
                pen=pg.mkPen(J_select.color_print[0], width=2)
            )
            self.lines.append(item)

        # MAJ spinbox λ
        self.bit_modif_PTlambda = True
        try:
            self.spinbox_x.setValue(x + self.deltalambdaP)
        finally:
            self.bit_modif_PTlambda = False

        return J_select

    def spinbox_t_move(self, value):
        if self.bit_modif_PTlambda:
            return
        try:
            if self.bit_load_jauge:
                self.Gauge_select.lamb_fit = round(float(inversefunc(
                    lambda x_: CL.T_Ruby_by_P(x_, P=self.Gauge_select.P,
                                            lamb0R=self.Gauge_select.lamb0),
                    value
                )), 3)
                self.Gauge_select = self.f_t_move(self.Gauge_select, value)

            if self.bit_modif_jauge and self.Spectrum is not None and self.index_jauge >= 0:
                G = self.Spectrum.Gauges[self.index_jauge]
                G.lamb_fit = round(float(inversefunc(
                    lambda x_: CL.T_Ruby_by_P(x_, P=G.P, lamb0R=G.lamb0),
                    value
                )), 3)
                self.Spectrum.Gauges[self.index_jauge] = self.f_t_move(G, value)

            self.save_value = value
        except Exception as e:
            print("spinbox_t_move error:", e)

    def Update_Print(self):
        state = self._get_state_for_run()
        if state is None:
            return

        data = []
        show_dPdt = self.var_bouton[0].isChecked()
        show_T = self.var_bouton[1].isChecked()
        show_piezo = self.var_bouton[2].isChecked()
        show_corr = self.var_bouton[3].isChecked()
        show_P = self.var_bouton[6].isChecked()

        for curve in state.curves_T:
            curve.setVisible(show_T)
            if show_T:
                y = curve.getData()[1]
                if y is not None:
                    data.extend(y.tolist())

        for curve in state.curves_P:
            curve.setVisible(show_P)
            if show_P:
                y = curve.getData()[1]
                if y is not None:
                    data.extend(y.tolist())

        for curve in state.curves_dPdt:
            curve.setVisible(show_dPdt)
            if show_dPdt:
                y = curve.getData()[1]
                if y is not None:
                    data.extend(y.tolist())

        for curve in state.curves_sigma:
            curve.setVisible(show_P)
            if show_P:
                y = curve.getData()[1]
                if y is not None:
                    data.extend(y.tolist())

        if state.piezo_curve is not None:
            state.piezo_curve.setVisible(show_piezo)
            if show_piezo:
                y = state.piezo_curve.getData()[1]
                if y is not None:
                    data.extend(y.tolist())

        if state.corr_curve is not None:
            state.corr_curve.setVisible(show_corr)
            if show_corr:
                y = state.corr_curve.getData()[1]
                if y is not None:
                    data.extend(y.tolist())

        if data:
            y_min = np.nanmin(data)
            y_max = np.nanmax(data)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
                self.pg_dPdt.setYRange(y_min * 1.01, y_max * 1.01, padding=0)

        self.pg_P.enableAutoRange(axis='y', enable=True)
        self.pg_dPdt.enableAutoRange(axis='y', enable=True)
        self.pg_sigma.enableAutoRange(axis='y', enable=True)
#? COMMANDE self.update 
    def f_gauge_select(self):
        col1 = self.Gauge_type_selector.model().item(self.Gauge_type_selector.currentIndex()).background().color().getRgb()
        self.Gauge_type_selector.setStyleSheet("background-color: rgba{};	selection-background-color: gray;".format(col1))
        go=False
        new_g=self.Gauge_type_selector.currentText()
        if self.Spectrum is None:
            go=True
        else:
            if new_g not in [G.name for G in self.Spectrum.Gauges]:
                go=True
        if go:
            self.bit_load_jauge=True
            self.bit_modif_jauge=False
            self.Gauge_select=CL.Gauge(name=new_g)
            self.Gauge_select.P=self.spinbox_P.value()
            self.lamb0_entry.setText(str(self.Gauge_select.lamb0))
            self.name_spe_entry.setText(str(self.Gauge_select.name_spe))
            self.f_dell_lines()
            self.f_p_move(self.Gauge_select,value=self.Gauge_select.P)
            self.name_gauge.setText("Add ?")
            self.name_gauge.setStyleSheet("background-color: red;")
            if self.Gauge_select.name == "Ruby":
                self.spinbox_T.setEnabled(True)
                #self.spinbox_T.setValue(self.Gauge_select.T)
            else:
                self.spinbox_T.setEnabled(False)
                self.spinbox_T.setValue(293)
                self.deltalambdaT=0
        else:
            self.index_jauge=[ga.name for ga in self.Spectrum.Gauges].index(self.Gauge_type_selector.currentText())
            self.LOAD_Gauge()

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
            
    def f_lambda0(self):
        lambda0=str(self.Spectrum.Gauges[self.index_jauge].lamb0)
        try:
            self.Spectrum.Gauges[self.index_jauge].lamb0=float(self.lamb0_entry.text())
        except Exception as e:
            self.lamb0_entry.setText(lambda0)
            print("ERROR:",e,"in lambda0")
    
        if self.Gauge_init_box.isChecked() and self.RUN is not None:
            self.RUN.Gauges_init[self.index_jauge].lamb0=float(self.lamb0_entry.text())
            self.REFRESH()
        
    def f_name_spe(self):
        name_spe=str(self.Spectrum.Gauges[self.index_jauge].name_spe)
        try:
            self.Spectrum.Gauges[self.index_jauge].spe=float(self.name_spe_entry.text())
            if self.Gauge_init_box.isChecked() and self.RUN is not None:
                self.RUN.Gauges_init[self.index_jauge].spe=float(self.name_spe_entry.text())
                self.REFRESH()
        except Exception as e:
            self.name_spe_entry.setText(name_spe)
            print("ERROR:",e,"in lambda0")

#########################################################################################################################################################################################
#? COMMANDE FIT
    def _build_pic_fit_problem(self, gauge_indices):
        """
        Construit :
        - la liste des fonctions de pics (list_F)
        - le vecteur initial_guess
        - les bornes bounds_min / bounds_max
        - le x_min/x_max basé sur Param0 (pour définir la zone globale de fit)
        gauge_indices : iterable d'indices de jauges à inclure (ex: range(nb_jauges) ou [j])
        """
        list_F = []
        initial_guess = []
        bounds_min = []
        bounds_max = []

        # x_min / x_max pris sur le premier pic rencontré
        first_j = next(iter(gauge_indices))
        x_min = float(self.Param0[first_j][0][0])
        x_max = float(self.Param0[first_j][0][0])

        inter = float(self.inter_entry.value())

        for j in gauge_indices:
            G = self.Spectrum.Gauges[j]
            for i in range(self.J[j]):
                ctr, ampH, sigma, coef_spe, model_fit = self.Param0[j][i]

                # Mise à jour du pic (modèle lmfit)
                pic = G.pics[i]
                pic.Update(
                    ctr=float(ctr),
                    ampH=float(ampH),
                    coef_spe=coef_spe,
                    sigma=float(sigma),
                    model_fit=model_fit,
                    inter=inter
                )

                # Zone min/max basée sur ctr ± 5σ
                x_min = min(x_min, float(ctr) - float(sigma) * 5.0)
                x_max = max(x_max, float(ctr) + float(sigma) * 5.0)

                # Fonction de modèle (pour Gen_sum_F)
                list_F.append(pic.f_model)

                # Initial guess : ctr, ampH, sigma, puis tous les coef_spe
                initial_guess.extend([ctr, ampH, sigma])
                for c in coef_spe:
                    initial_guess.append(c)

                # Bornes
                bounds_min.extend([
                    pic.ctr[1][0],
                    pic.ampH[1][0],
                    pic.sigma[1][0]
                ])
                bounds_max.extend([
                    pic.ctr[1][1],
                    pic.ampH[1][1],
                    pic.sigma[1][1]
                ])
                for c in pic.coef_spe:
                    bounds_min.append(c[1][0])
                    bounds_max.append(c[1][1])

            # Met à jour le modèle de la jauge
            G.Update_model()

        return (
            list_F,
            np.array(initial_guess, dtype=float),
            np.array(bounds_min, dtype=float),
            np.array(bounds_max, dtype=float),
            float(x_min),
            float(x_max),
        )
    
    def _propose_and_confirm_fit(self, x_fit, y_fit, fit, color, text_base, use_abs=False):
        """
        Gère :
        - le calcul 'is_better' vs self.Spectrum.dY
        - les courbes provisoires sur les plots PyQtGraph (spectre / dY)
        - le message box V/C (sauf en mode bit_bypass)
        - le nettoyage des courbes provisoires

        Retourne:
        accepted (bool), is_better (bool)
        """

        # 1) Critère d'amélioration
        if self.Spectrum.dY is not None:
            resid_new = (y_fit - fit)
            if use_abs:
                new_score = float(np.sum(np.abs(resid_new)))
                old_score = float(np.sum(np.abs(self.Spectrum.dY)))
            else:
                new_score = float(np.sum(resid_new ** 2))
                old_score = float(np.sum(self.Spectrum.dY ** 2))
            is_better = new_score < old_score
        else:
            is_better = True

        text_fit = (
            f"{text_base} BEST you can Validate"
            if is_better
            else f"{text_base} LESS GOOD you can Cancel"
        )

        # 2) Traces provisoires (PyQtGraph)
        temp_curve_fit = self.pg_spectrum.plot(x_fit, fit, pen=pg.mkPen(color, width=2, style=Qt.DashLine))
        temp_curve_dy = self.pg_dy.plot(x_fit, y_fit - fit, pen=pg.mkPen(color, width=2, style=Qt.DashLine))

        # 3) Interaction utilisateur / ou auto en bypass
        if not self.bit_bypass:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("CURVE FIT DONE")
            msg_box.setText(text_fit + '\n Save fit Press "v" Cancel Press "c"')

            v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)
            a_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            msg_box.setDefaultButton(v_button)

            def on_key_press(event):
                if event.key() == Qt.Key_V:
                    v_button.click()
                elif event.key() == Qt.Key_C:
                    a_button.click()

            msg_box.keyPressEvent = on_key_press
            msg_box.exec_()

            accepted = (msg_box.clickedButton() == v_button)
        else:
            accepted = is_better

        # 4) Nettoyage des courbes provisoires
        try:
            self.pg_spectrum.removeItem(temp_curve_fit)
        except Exception:
            pass
        try:
            self.pg_dy.removeItem(temp_curve_dy)
        except Exception:
            pass

        return accepted, is_better

    def FIT_lmfitVScurvfit_ONE_GAUGE(self):
        """Fit uniquement sur la jauge actuellement sélectionnée (sans traçage Matplotlib)."""
        save_jauge = self.index_jauge
        save_pic = self.index_pic_select

        self.Param_FIT = []
        self.nb_jauges = len(self.Spectrum.Gauges)
        j = self.index_jauge

        if j < 0 or j >= self.nb_jauges:
            self.text_box_msg.setText("FIT ONE GAUGE : aucune jauge sélectionnée.")
            return

        # 1) Préparation du problème de fit pour cette seule jauge
        (
            list_F,
            initial_guess,
            bounds_min,
            bounds_max,
            _x_min,
            _x_max,
        ) = self._build_pic_fit_problem([j])

        bounds = (bounds_min, bounds_max)

        # 2) x_fit / y_fit pour cette jauge
        if (
            self.X_e[j] is not None
            and self.X_s[j] is not None
            and self.Spectrum.Gauges[j].indexX is not None
        ):
            self.Zone_fit[j] = np.where(
                (self.Spectrum.wnb >= self.X_s[j])
                & (self.Spectrum.wnb <= self.X_e[j])
            )[0]
            x_fit = self.Spectrum.wnb[self.Zone_fit[j]]
            self.Spectrum.Gauges[j].indexX = self.Zone_fit[j]
            y_fit = self.Spectrum.spec[self.Spectrum.indexX]
        else:
            x_fit = self.Spectrum.x_corr
            y_fit = self.Spectrum.y_corr

        sum_function = CL.Gen_sum_F(list_F)

        try:
            params, params_covar = curve_fit(
                sum_function, x_fit, y_fit, p0=initial_guess, bounds=bounds
            )
        except Exception as e:
            self.Spectrum.bit_fit = True
            self.bit_fit_T = True
            self.text_box_msg.setText("FIT ERROR" + str(e))
            return

        fit = sum_function(x_fit, *params)

        accepted, is_better = self._propose_and_confirm_fit(
            x_fit=x_fit,
            y_fit=y_fit,
            fit=fit,
            color=self.c_m[j],
            text_base="Curve_fit",
            use_abs=True,
        )

        if not accepted:
            self.Spectrum.bit_fit = True
            self.bit_fit_T = True
            self.text_box_msg.setText("BAD FIT r^2 INCREAS")
            return

        # 3) Mise à jour des données de la jauge j uniquement
        self.Spectrum.Gauges[j].Y = fit + self.Spectrum.blfit
        self.Spectrum.Gauges[j].X = self.Spectrum.x_corr
        self.Spectrum.Gauges[j].dY = y_fit - fit
        self.Spectrum.lamb_fit = params[0]

        ij_3 = ij_4 = ij_5 = 0
        for i, J in enumerate(self.Spectrum.Gauges):
            for k, p in enumerate(J.pics):
                n_c = len(self.Param0[i][k][3])
                start_idx = 3 * ij_3 + 4 * ij_4 + 5 * ij_5
                end_idx = start_idx + 3

                if n_c == 0:
                    self.Param0[i][k][:3] = list(params[start_idx:end_idx])
                    ij_3 += 1
                elif n_c == 1:
                    self.Param0[i][k][:4] = list(params[start_idx:end_idx]) + list(
                        np.array(params[end_idx])
                    )
                    ij_4 += 1
                elif n_c == 2:
                    self.Param0[i][k][:4] = list(params[start_idx:end_idx]) + list(
                        np.array(params[end_idx : end_idx + 2])
                    )
                    ij_5 += 1

                p.Update(
                    ctr=float(self.Param0[i][k][0]),
                    ampH=float(self.Param0[i][k][1]),
                    coef_spe=self.Param0[i][k][3],
                    sigma=float(self.Param0[i][k][2]),
                    inter=float(self.inter_entry.value()),
                )
                params_f = p.model.make_params()
                self.list_y_fit_start[i][k] = p.model.eval(
                    params_f, x=self.Spectrum.wnb
                )
                new_name = (
                    f"{self.Nom_pic[i][k]}   X0:{self.Param0[i][k][0]}"
                    f"   Y0:{self.Param0[i][k][1]}"
                    f"   sigma:{self.Param0[i][k][2]}"
                    f"   Coef:{self.Param0[i][k][3]}"
                    f" ; Modele:{self.Param0[i][k][4]}"
                )
                self.list_text_pic[i][k] = str(new_name)

        self.Spectrum.bit_fit = True
        self.text_box_msg.setText("FIT TOTAL \n DONE")
        self.bit_fit_T = True
        self.index_jauge = save_jauge
        self.index_pic_select = save_pic
        self.LOAD_Gauge()
        self.Print_fit_start()   # -> recalc y_fit_start + dY + refresh PyQtGraph

    def FIT_lmfitVScurvfit(self):
        """Fit global sur toutes les jauges (sans tracés Matplotlib)."""
        save_jauge = self.index_jauge
        save_pic = self.index_pic_select

        self.Param_FIT = []
        self.nb_jauges = len(self.Spectrum.Gauges)

        if self.nb_jauges == 0:
            self.text_box_msg.setText("FIT : aucune jauge dans le spectre.")
            return

        # 1) Préparation du problème de fit
        gauge_indices = range(self.nb_jauges)
        (
            list_F,
            initial_guess,
            bounds_min,
            bounds_max,
            x_min,
            x_max,
        ) = self._build_pic_fit_problem(gauge_indices)

        bounds = (bounds_min, bounds_max)

        # 2) Modèle global et pré-traitement
        self.Spectrum.model = None
        for j in gauge_indices:
            G = self.Spectrum.Gauges[j]
            if self.Spectrum.model is None:
                self.Spectrum.model = G.model
            else:
                self.Spectrum.model += G.model

        self.Spectrum.Data_treatement()

        # Zone de fit
        if self.zone_spectrum_box.isChecked():
            self.Spectrum.indexX = np.where(
                (self.Spectrum.wnb >= x_min) & (self.Spectrum.wnb <= x_max)
            )[0]
            x_sub = self.Spectrum.wnb[self.Spectrum.indexX]
            y_sub = self.Spectrum.y_corr[self.Spectrum.indexX]
            blfit = self.Spectrum.blfit[self.Spectrum.indexX]
        else:
            if (
                self.X_e[0] is not None
                and self.X_s[0] is not None
                and self.Spectrum.indexX is not None
            ):
                self.Zone_fit[0] = np.where(
                    (self.Spectrum.wnb >= self.X_s[0])
                    & (self.Spectrum.wnb <= self.X_e[0])
                )[0]
                x_sub = self.Spectrum.wnb[self.Zone_fit[0]]
                self.Spectrum.indexX = self.Zone_fit[0]
                for J in self.Spectrum.Gauges:
                    J.indexX = self.Zone_fit[0]
                y_sub = self.Spectrum.y_corr[self.Spectrum.indexX]
                blfit = self.Spectrum.blfit[self.Spectrum.indexX]
            else:
                y_sub = self.Spectrum.y_corr
                blfit = self.Spectrum.blfit
                x_sub = self.Spectrum.wnb

        # 3) Option : lmfit par jauge (comme avant)
        if self.vslmfit.isChecked():
            self.Spectrum.FIT()
            for i, J in enumerate(self.Spectrum.Gauges):
                for j, p in enumerate(J.pics):
                    params_f = p.model.make_params()
                    y_plot = p.model.eval(params_f, x=self.Spectrum.wnb)
                    self.list_y_fit_start[i][j] = y_plot

        # 4) curve_fit global
        sum_function = CL.Gen_sum_F(list_F)

        try:
            params, params_covar = curve_fit(
                sum_function,
                x_sub,
                y_sub,
                p0=initial_guess,
                bounds=bounds,
            )
        except Exception as e:
            self.Spectrum.bit_fit = True
            self.bit_fit_T = True
            self.text_box_msg.setText("FIT ERROR" + str(e))
            return

        fit = sum_function(x_sub, *params)

        accepted, is_better = self._propose_and_confirm_fit(
            x_fit=x_sub,
            y_fit=y_sub,
            fit=fit,
            color="m",
            text_base="Curve_fit",
            use_abs=False,
        )

        if not accepted:
            # On garde les anciens paramètres et on re-affiche les courbes start
            self.Spectrum.bit_fit = True
            self.bit_fit_T = True
            self.Spectrum.Calcul_study(mini=False)
            self.text_box_msg.setText("BAD FIT r^2 INCREAS")
            self.Print_fit_start()
            return

        # 5) Validation : mise à jour de Spectrum + Param0 + list_y_fit_start
        self.Spectrum.Y = fit + blfit
        self.Spectrum.X = x_sub
        self.Spectrum.dY = y_sub - fit
        self.Spectrum.lamb_fit = params[0]

        ij_3 = ij_4 = ij_5 = 0
        params_list = list(params)

        for i, J in enumerate(self.Spectrum.Gauges):
            for j, p in enumerate(J.pics):
                n_c = len(self.Param0[i][j][3])
                start_idx = 3 * ij_3 + 4 * ij_4 + 5 * ij_5
                end_idx = start_idx + 3

                if n_c == 0:
                    self.Param0[i][j][:4] = params_list[start_idx:end_idx]
                    ij_3 += 1
                elif n_c == 1:
                    self.Param0[i][j][:4] = (
                        params_list[start_idx:end_idx]
                        + [np.array([params_list[end_idx]])]
                    )
                    ij_4 += 1
                elif n_c == 2:
                    self.Param0[i][j][:4] = (
                        params_list[start_idx:end_idx]
                        + [np.array(params_list[end_idx : end_idx + 2])]
                    )
                    ij_5 += 1

                p.Update(
                    ctr=float(self.Param0[i][j][0]),
                    ampH=float(self.Param0[i][j][1]),
                    coef_spe=self.Param0[i][j][3],
                    sigma=float(self.Param0[i][j][2]),
                    inter=float(self.inter_entry.value()),
                )
                params_f = p.model.make_params()
                y_plot = p.model.eval(params_f, x=self.Spectrum.wnb)
                self.list_y_fit_start[i][j] = y_plot

                new_name = (
                    f"{self.Nom_pic[i][j]}   X0:{self.Param0[i][j][0]}"
                    f"   Y0:{self.Param0[i][j][1]}"
                    f"   sigma:{self.Param0[i][j][2]}"
                    f"   Coef:{self.Param0[i][j][3]}"
                    f" ; Modele:{self.Param0[i][j][4]}"""
                )
                self.list_text_pic[i][j] = str(new_name)

            J.lamb_fit = self.Param0[i][0][0]
            J.bit_fit = True

        self.Spectrum.bit_fit = True
        self.Spectrum.Calcul_study(mini=False)
        self.text_box_msg.setText("FIT TOTAL \n DONE")
        self.bit_fit_T = True
        self.index_jauge = save_jauge
        self.index_pic_select = save_pic
        self.LOAD_Gauge()
        self.Print_fit_start()

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
                titre="Movie :1e"+str(round(np.log10(self.RUN.fps),2))+"fps"
            except Exception as e:
                print("fps log ERROR:",e)
                titre="Movie :"+str(round(self.RUN.fps,2))+"fps"
        else:
            titre="No Movie"

        self.setWindowTitle(titre)
        if state.index_cam:
            self.current_index = len(state.index_cam)//2 if state.index_cam else 0
            self.slider.setMaximum(max(0, len(state.index_cam) - 1))
            self.slider.setValue(self.current_index)
        self._update_movie_frame()
        self.label_CED.setText( "CEDd "+item.text()+" select")
        if state.time:
            x_min,x_max=min(state.time),max(state.time)
            self.pg_P.setXRange(x_min,x_max,padding=0.01)
            self.pg_dPdt.setXRange(x_min,x_max,padding=0.01)
            self.pg_sigma.setXRange(x_min,x_max,padding=0.01)
            self.pg_dlambda.setXRange(x_min,x_max,padding=0.01)

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
        state.spectre_number = self.RUN.list_nspec
        state.list_item = item_run

        # Courbes P / dPdt / sigma / T

        for i, G in enumerate(Gauges_RUN):
            print(G.name)
            l_p_filtre = CL.savgol_filter(l_P[i], 10, 1)
            dps = [
                (l_p_filtre[x + 1] - l_p_filtre[x - 1]) / (Time[x + 1] - Time[x - 1]) * 1e-3
                for x in range(2, len(l_p_filtre) - 2)
            ]
            state.curves_P.append(
                self.pg_P.plot(Time, l_P[i], pen=pg.mkPen(G.color_print[0], width=1), symbol='d', symbolPen=None, symbolBrush=G.color_print[0], symbolSize=4)
            )
            state.curves_dPdt.append(
                self.pg_dPdt.plot(Time[2:-2], dps, pen=pg.mkPen(G.color_print[0], width=1), symbol='d', symbolPen=None, symbolBrush=G.color_print[0], symbolSize=4)
            )
            state.curves_sigma.append(
                self.pg_sigma.plot(Time, l_fwhm[i], pen=pg.mkPen(G.color_print[0], width=1), symbol='d', symbolPen=None, symbolBrush=G.color_print[0], symbolSize=4)
            )
        if "RuSmT" in [x.name_spe for x in self.RUN.Gauges_init]:
            state.curves_T.append(
                self.pg_dPdt.plot(Time, l_T[-1], pen=pg.mkPen('darkred'), symbol='t', symbolBrush='darkred', symbolSize=6)
            )
        else:
            state.curves_T.append(self.pg_dPdt.plot([], []))

        # Piezo
        if self.RUN.data_Oscillo is not None:
            state.piezo_curve = self.pg_P.plot(time_amp, amp, pen=pg.mkPen(c))
        else:
            state.piezo_curve = self.pg_P.plot([], [])

        for spe in l_spe:
            if spe is not []:
                state.curves_dlambda.append(
                    self.pg_dlambda.plot(Time, spe, pen=None, symbol='+', symbolPen=pg.mkPen(c))
                )

        # Film
        if self.RUN.folder_Movie is not None:
            t0_movie = getattr(self.RUN, 't0_movie', 0)
            cap, fps, nb_frames = self.Read_Movie(self.RUN)
            state.cap = cap
            gray_l = []
            nb_c = 5

            for i in range(nb_frames):
                t_c = i / fps + t0_movie
                if Time[0] < t_c < Time[-1]:
                    state.t_cam.append(t_c)
                    state.index_cam.append(i)
                    if self.var_bouton[3].isChecked():
                        correlation = 0
                        gray_curr = self.read_frame(cap, i, unit="gray")
                        for gray in gray_l:
                            correlation += cv2.matchTemplate(gray, gray_curr, cv2.TM_CCOEFF_NORMED)[0][0] - 1
                        state.correlations.append(correlation)
                        if len(gray_l) > nb_c:
                            del gray_l[0]
                        gray_l.append(gray_curr)

            if self.var_bouton[3].isChecked() and state.correlations:
                state.correlations = list(np.array(state.correlations) / max(abs(np.array(state.correlations))))
                state.corr_curve = self.pg_dPdt.plot(state.t_cam, state.correlations, pen=pg.mkPen(c))
            else:
                state.corr_curve = self.pg_dPdt.plot([], [])

            self.current_index = len(state.t_cam) // 2 if state.t_cam else 0
            if state.index_cam:
                self.Num_im = state.index_cam[self.current_index]
                t = state.t_cam[self.current_index]
                Frame = self.read_frame(cap, self.Num_im)
                if Frame is not None:
                    self.img_item.setImage(np.array(Frame), autoLevels=True)

            self.slider.setMaximum(max(0, len(state.index_cam) - 1))
            self.slider.setValue(self.current_index)
            self._update_movie_frame()
        else:
            self.Num_im = 0
            t = Time[0]
            state.corr_curve = self.pg_dPdt.plot([], [])

        # Enregistre l'état et finalise l'affichage
        self.runs[run_id] = state
        self._finalize_run_selection(state, name_select)
        self._refresh_ddac_limits(state)

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
        self.bit_load_jauge = False
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

    def ADD_gauge(self): # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ADD JAUGE - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        new_g=self.Gauge_type_selector.currentText()
        if new_g not in self.list_name_gauges:
            self.bit_modif_jauge =False
        if self.bit_modif_jauge is True :
            return print("if you want this gauges DELL AND RELOAD")
        self.bit_load_jauge=False
        new_Jauge=CL.Gauge(name=new_g)
        new_Jauge.P=self.spinbox_P.value()
        new_Jauge.lamb_fit=new_Jauge.inv_f_P(new_Jauge.P)
        self.Spectrum.Gauges.append(new_Jauge)
        
        self.index_jauge=len(self.Spectrum.Gauges)-1
        self.Update_var(new_Jauge.name) #self.list_name_gauges.append(new_g) in update var
        self.LOAD_Gauge()
        self.name_gauge.setText("In")
        self.name_gauge.setStyleSheet("background-color: green;")
        self.Auto_pic()
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

        # Restitue la couleur disponible
        self.c_m.insert(0, state.color)
        if state.color in self.color:
            self.color.remove(state.color)

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
    
    def Dell_Jauge(self):# - - - DELL JAUGE- - -#
        if self.index_jauge == -1:
            return print("jauge not select")
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Warning dell Jauge")
        text='You going to dell '+ self.Spectrum.Gauges[self.index_jauge].name+'\n Press "v" for Validate "c" for Cancel'
        msg_box.setText(text)

        v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)
        a_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

        msg_box.setDefaultButton(v_button)

        def on_key_press(event):
            if event.key() == Qt.Key_V:
                v_button.click()
            elif event.key() == Qt.Key_C:
                a_button.click()

        msg_box.keyPressEvent = on_key_press
        msg_box.exec_()
        

        if msg_box.clickedButton() == v_button:

            if self.index_jauge==0 and len(self.Param0)>2:
                self.z1[self.index_jauge+1]=self.z1[self.index_jauge]
                self.z2[self.index_jauge+1]=self.z2[self.index_jauge]
                self.X_s[self.index_jauge+1],self.X_e[self.index_jauge+1],self.Zone_fit[self.index_jauge+1]=self.X_s[self.index_jauge],self.X_e[self.index_jauge],self.Zone_fit[self.index_jauge]

            self.Click_Clear()
            del(self.Nom_pic[self.index_jauge])
            del(self.Param0[self.index_jauge])
            del(self.list_text_pic[self.index_jauge])
            del(self.J[self.index_jauge])
            del(self.plot_pic_fit[self.index_jauge])
            del(self.list_y_fit_start[self.index_jauge])
            del(self.list_name_gauges[self.index_jauge])
            try:
                del(self.Param_FIT[self.index_jauge]) 
            except Exception as e:
                print("del(Param_FIT[J])",e)
            del(self.X_s[self.index_jauge])
            del(self.X_e[self.index_jauge])
            del(self.z1)[self.index_jauge]
            del(self.z2[self.index_jauge])
            del(self.Zone_fit[self.index_jauge])
            #del(self.bit_plot_fit[self.index_jauge])
            #del(self.plot_fit[self.index_jauge])
            #del(self.bit_fit[self.index_jauge])
            del(self.Spectrum.Gauges[self.index_jauge])
            self.text_box_msg.setText('JAUGE DELL')
            self.name_gauge.setText("Add ?")
            self.name_gauge.setStyleSheet("background-color: red;")
            self.Print_fit_start()
            self.index_jauge-=1
            if len(self.Param0) != 0 :
                self.Gauge_type_selector.setCurrentIndex(self.liste_type_Gauge.index(self.list_name_gauges[self.index_jauge]))
                self.LOAD_Gauge()
            else:
                self.f_gauge_select()
        else:
            print("Function stopped.")
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
            self.listbox_pic.clear()
            # Vide les courbes PyQtGraph
            self._refresh_spectrum_view()
        else:
            if isinstance(self.Spectrum, CL.Spectre):
                self.nb_jauges = len(self.Spectrum.Gauges)
                self.list_name_gauges = [jauge.name for jauge in self.Spectrum.Gauges]
                print("gauges save")
            else:
                print("gauges dell")
                self.nb_jauges = 0
                self.list_name_gauges = []
                self.listbox_pic.clear()

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

        self.X0 = 0
        self.Y0 = 0
        self.selected_file = None

        # Bits de contrôle
        self.bit_print_fit_T = False
        self.bit_fit_T = False
        self.bit_fit = [False for _ in range(self.nb_jauges)]
        self.bit_modif_jauge = False
        self.bit_load_jauge = False
        self.bit_filtre = False
        self.bit_plot_fit = [False for _ in range(self.nb_jauges)]

        # Indice de jauge
        self.index_jauge = -1

    def DEBUG_SPECTRUM(self):
        self.CLEAR_ALL()
        self.bit_bypass=True
        self.LOAD_Spectrum
        self.bit_bypass=False
#########################################################################################################################################################################################
#? COMMANDE SAVE
    def CREAT_new_CEDd_fit(self):
        self.Spectrum_save = copy.deepcopy(self.Spectrum)
        if self.var_bouton[5].isChecked():
            folder_movie = self.loaded_filename_movie
        else:
            folder_movie = None

        New_CEDd = CL.CEDd(
            self.loaded_filename_spectro,
            Gauges_init=copy.deepcopy(self.Spectrum.Gauges),
            data_Oscillo=self.loaded_filename_oscilo,
            folder_Movie=folder_movie,
            fit=True,   # <<<<< fit auto comme avant
            time_index=[2, 4],
            type_filtre=self.Spectrum.type_filtre,
            param_f=self.Spectrum.param_f
        )

        if New_CEDd.Summary.empty:
            print("Loop STOP")
            return

        name_CED = os.path.basename(self.loaded_filename_oscilo)
        name_version = ".CEDUp"
        New_CEDd.CEDd_path = os.path.join(folder_CEDd, (name_CED + name_version))
        print("Save as ", name_CED, "in folder ", folder_CEDd)
        print(New_CEDd.Summary)
        self.PRINT_CEDd(objet_run=New_CEDd, item=None)

    def CREAT_new_CEDd(self):
        """Crée un CEDd sans lancer de fit automatique."""
        self.Spectrum_save = copy.deepcopy(self.Spectrum)

        if self.var_bouton[5].isChecked():
            folder_movie = self.loaded_filename_movie
        else:
            folder_movie = None

        New_CEDd = CL.CEDd(
            self.loaded_filename_spectro,
            Gauges_init=copy.deepcopy(self.Spectrum.Gauges),
            data_Oscillo=self.loaded_filename_oscilo,
            folder_Movie=folder_movie,
            fit=False,  # <<<<< PAS de fit auto
            time_index=[2, 4],
            type_filtre=self.Spectrum.type_filtre,
            param_f=self.Spectrum.param_f
        )

        name_CED = os.path.basename(self.loaded_filename_oscilo)
        name_version = ".CEDUp"
        New_CEDd.CEDd_path = os.path.join(folder_CEDd, (name_CED + name_version))
        print("Created CEDd (no fit) as ", New_CEDd.CEDd_path)

        self.PRINT_CEDd(objet_run=New_CEDd, item=None)
        self.text_box_msg.setText("CEDd sans fit auto chargé.\nLancer ensuite le multi-fit.")


    def CREAT_new_Spectrum(self):
        save_gauges = []

        # Sauvegarde des jauges du spectre courant si possible
        if isinstance(self.Spectrum, CL.Spectre):
            save_gauges = copy.deepcopy(self.Spectrum.Gauges)

        # On ne vide pas tout, seulement ce que CLEAR_ALL(empty=False) fait déjà
        self.CLEAR_ALL(empty=False)
        self.bit_bypass = True

        # Récupération de l'index de spectre depuis la spinbox
        if hasattr(self, "spinbox_spec_index"):
            idx = int(self.spinbox_spec_index.value())   # 0,1,2,...
            # Dans data_Spectro : col 0 = X, col 1..N = spectres
            n_spec = idx + 1
        else:
            # fallback : premier spectre (colonne 1)
            n_spec = 1

        # Sécurité : clamp sur le nombre de colonnes disponibles
        max_col = self.data_Spectro.shape[1] - 1  # dernière colonne de Y
        if n_spec < 1:
            n_spec = 1
        if n_spec > max_col:
            n_spec = max_col

        self.text_box_msg.setText(f"New spec n°{n_spec}")

        x = np.array(self.data_Spectro[0])
        y = np.array(self.data_Spectro[n_spec])

        # Création du nouveau Spectre avec les mêmes jauges que le spectre courant
        new_spectrum = CL.Spectre(x, y, Gauges=save_gauges)

        # Chargement dans l'UI
        try:
            self.LOAD_Spectrum(Spectrum=new_spectrum)
        except TypeError:
            # au cas où ta LOAD_Spectrum lit self.Spectrum / self.index_spec plutôt qu'un argument
            self.Spectrum = new_spectrum
            self.LOAD_Spectrum()

        self.bit_bypass = False

    def SAVE_CEDd(self):
        self.RUN.Spectra[self.index_spec]=self.Spectrum
        CL.SAVE_CEDd(self.RUN)
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

    def Click_Clear(self):
        """Efface tous les pics et la zone de fit de la jauge courante, sans manip Matplotlib."""
        if self.index_jauge == -1:
            print("Clear resolve : pas de jauge")
            return

        j = self.index_jauge

        # Reset des infos de pics pour cette jauge
        self.Nom_pic[j] = []
        self.J[j] = 0
        self.X0 = 0
        self.Y0 = 0
        self.list_text_pic[j] = []

        # Zone de fit
        self.X_s[j] = None
        self.X_e[j] = None
        self.Zone_fit[j] = None
        self.Spectrum.Gauges[j].indexX = None

        # Paramètres et modèles
        self.Param0[j] = []
        self.Spectrum.Gauges[j].pics = []
        try:
            self.Param_FIT[j] = []
        except Exception as e:
            print("Param_FIT[J]=[]", e)

        # Courbes associées
        self.list_y_fit_start[j] = []

        # UI
        self.listbox_pic.clear()
        self.text_box_msg.setText("GL&HF")

        # Filtre éventuel
        if self.bit_filtre and hasattr(self, "filtre_OFF"):
            self.filtre_OFF()
            self.bit_filtre = False

        # Recalcul global des résidus + mise à jour des graphes PyQtGraph
        self.Print_fit_start()

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
            self.Print_fit_start()
    
    def Undo_pic_select(self):
        if self.index_pic_select is not None:
            name=self.listbox_pic.item(self.index_pic_select).text()
            #motif = r'_p(\d+)_'
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
                self.Print_fit_start()
                
    def select_pic(self):
        if not self.bit_bypass:
            self.index_pic_select = self.listbox_pic.currentRow()
            if self.index_pic_select < 0:
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
            self.curve_zoom_pic.setData(S.wnb, y_pic)
            self.curve_spec_pic_select.setData(S.wnb, y_pic)
            self.curve_zoom_data.setData(S.wnb, S.spec - (self.y_fit_start - y_pic) - S.blfit)
            self.curve_zoom_data_brut.setData(S.wnb, S.spec- S.blfit)
        else:
            self.curve_zoom_pic.setData([], [])
            self.curve_spec_pic_select.setData([], [])
            if hasattr(S, "wnb") and hasattr(S, "spec"):
                self.curve_zoom_data.setData(S.wnb, S.spec)
                self.curve_zoom_data_brut.setData(S.wnb, S.spec)
            else:
                self.curve_zoom_data.setData([], [])
                self.curve_zoom_data_brut.setData([], [])

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
        self.curve_zoom_pic.setData(x_data, y_plot_full)

        self.curve_spec_pic_select.setData(x_data, y_plot_full)
        if hasattr(self.Spectrum, "spec") and hasattr(self.Spectrum, "blfit"):
            self.curve_zoom_data.setData(x_data, self.Spectrum.spec - (self.y_fit_start - y_plot_full) - self.Spectrum.blfit)
            self.curve_zoom_data_brut.setData(x_data, self.Spectrum.spec - self.Spectrum.blfit)
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
        self.curve_zoom_pic.setData(x_data, y_plot_full)
        self.curve_spec_pic_select.setData(x_data, y_plot_full)
        if hasattr(self.Spectrum, "spec") and hasattr(self.Spectrum, "blfit"):
            self.curve_zoom_data.setData(x_data, self.Spectrum.spec - (self.y_fit_start - y_plot_full) - self.Spectrum.blfit)
            self.curve_zoom_data_brut.setData(x_data, self.Spectrum.spec- self.Spectrum.blfit)
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

        # Récupération borne de départ / fin depuis l'UI
        index_start = int(self.index_start_entry.value())
        index_stop  = int(self.index_stop_entry.value())

        index_start = max(0, index_start)
        index_stop  = min(len(self.RUN.Spectra) - 1, index_stop)

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
        dlg = ProgressDialog(
            figs=None,              # tu peux passer des figs si tu veux
            cancel_text="Stop",
            value=0,
            Gauges=None,
            parent=self
        )
        dlg.setLabelText(f"Multi-fit : spectres {index_start} → {index_stop}")
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()

        canceled = False

        # ----- BOUCLE SUR LES SPECTRES -----
        try:
            for k, i in enumerate(range(index_start, index_stop + 1), start=1):
                # Vérifier si l'utilisateur a cliqué sur Stop
                if dlg.wasCanceled():
                    canceled = True
                    print("Multi-fit: canceled by user.")
                    break

                print(f"Multi-fit : spectre {i}/{index_stop}")

                # Mise à jour du texte et de la barre de progression
                dlg.setLabelText(f"Fit du spectre {i} / {index_stop}")
                percent = int(100 * k / n_tot)
                dlg.setValue(percent)
                QApplication.processEvents()

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
                    self.FIT_lmfitVScurvfit()
                except Exception as e:
                    print(f"Error during fit on spectrum {i}:", e)
                self.bit_bypass = False

                # On sauvegarde le spectre fitté dans RUN
                self.RUN.Spectra[i] = copy.deepcopy(self.Spectrum)

        finally:
            # On ferme le dialogue quoi qu'il arrive
            dlg.close()

        # ----- APRÈS LA BOUCLE : TOUJOURS Corr_Summary + REFRESH -----
        try:
            self.RUN.Corr_Summary(All=True)
        except Exception as e:
            print("Error in RUN.Corr_Summary(All=True):", e)

        # On revient sur le spectre initial dans l'UI
        self.index_spec = original_index_spec
        self.spinbox_spec_index.blockSignals(True)
        self.spinbox_spec_index.setValue(self.index_spec)
        self.spinbox_spec_index.blockSignals(False)
        self.Spectrum   = self.RUN.Spectra[self.index_spec]
        self.LOAD_Spectrum(Spectrum=self.Spectrum)
        self.REFRESH()

        if canceled:
            self.text_box_msg.setText("Multi-fit chaîné interrompu par l'utilisateur.")
        else:
            self.text_box_msg.setText("Multi-fit chaîné terminé.")

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
