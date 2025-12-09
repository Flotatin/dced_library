# BANC CED Dynamic 

import copy
import io
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from string import Template
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
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyledItemDelegate,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import curve_fit
STYLE_TEMPLATE = Template(
    """
       /* Appliquer la police scientifique */
* {
    font-family: 'Bitstream Vera Sans Mono', monospace;
    font-size: 12pt;
}

/* Fond principal */
QMainWindow {
    background-color: ${window};
}

/* Style général des widgets */
QWidget {
    color: ${text};
    background-color: ${background};
}

/* Combobox */

QComboBox {
    border: 1px solid ${accent};
    border-radius: 4px;
    padding: 5px;
}

QComboBox QAbstractItemView {
    selection-background-color: ${selection};
    selection-color: ${selection_text};
}

/* Boutons */
QPushButton {
    background-color: ${accent};
    color: ${button_text};
    border-radius: 5px;
    padding: 8px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: ${accent_hover};
}
QPushButton:pressed {
    background-color: ${accent_pressed};
}

/* Champs de saisie */
QLineEdit, QSpinBox, QTextEdit ,QDoubleSpinBox{
    background-color: ${input_background};
    color: ${text};
    border: 1px solid ${accent};
    border-radius: 4px;
    padding: 5px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid ${selection};
}

/* Menus et barres de menu */
QMenuBar {
    background-color: ${menu_background};
    color: ${text};
}
QMenu {
    background-color: ${menu_background};
    color: ${text};
}
QMenu::item:selected {
    background-color: ${selection};
    color: ${selection_text};
}

/* Barres de défilement */
QScrollBar:vertical, QScrollBar:horizontal {
    background: ${input_background};
    width: 12px;
}
QScrollBar::handle {
    background: ${accent};
    border-radius: 6px;
}
QScrollBar::handle:hover {
    background: ${accent_hover};
}

/* Cases à cocher et boutons radio */
QCheckBox, QRadioButton {
    color: ${text};
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: ${accent};
}

/* Barres d onglets */
QTabBar::tab {
    background-color: ${input_background};
    color: ${text};
    padding: 6px;
    border-radius: 5px;
}
QTabBar::tab:selected {
    background-color: ${accent};
    color: ${button_text};
}

"""
)

THEMES = {
    "dark": {
        "window": "#2b2b2b",
        "background": "#333333",
        "menu_background": "#222222",
        "text": "#e0e0e0",
        "button_text": "#ffffff",
        "accent": "#0099cc",
        "accent_hover": "#0077aa",
        "accent_pressed": "#005577",
        "input_background": "#444444",
        "selection": "#ffaa55",
        "selection_text": "#ffffff",
        "plot_background": "#333333",
        "axis_pen": "#e0e0e0",
        "grid_alpha": 0.3,
        "pens": {
            "spectrum_data": {"color": "#ffffff"},
            "spectrum_fit": {"color": "#00ff00", "width": 2},
            "spectrum_pic_brush": (255, 0, 0, 120),
            "dy": {"color": "#ffff00"},
            "zero_line": {"color": "#000000"},
            "baseline_brut": {"color": "#ffffff"},
            "baseline_fit": {"color": "#00ffff"},
            "fft": {"color": "#ff00ff"},
            "zoom_data": {"color": "#000000"},
            "zoom_data_brut": {"color": "#ffffff"},
            "zoom_pic_brush": (255, 0, 0, 80),
            "selection_line": {"color": "#00ff00", "width": 1},
            "cross_zoom": {"color": "#ff0000"},
            "text_item": "#ffffff",
            "line_t": {"color": "#00ff00", "width": 1},
            "baseline_time": {"color": "#000000", "width": 2},
            "zone_movie": {"color": "#ffff00", "style": Qt.DashLine},
            "scatter": {"color": "#ffffff", "width": 2},
        },
    },
    "light": {
        "window": "#f2f2f2",
        "background": "#ffffff",
        "menu_background": "#e5e5e5",
        "text": "#1e1e1e",
        "button_text": "#ffffff",
        "accent": "#0066cc",
        "accent_hover": "#005bb5",
        "accent_pressed": "#004c99",
        "input_background": "#f6f6f6",
        "selection": "#ffb347",
        "selection_text": "#1e1e1e",
        "plot_background": "#f7f7f7",
        "axis_pen": "#1e1e1e",
        "grid_alpha": 0.25,
        "pens": {
            "spectrum_data": {"color": "#0c2340"},
            "spectrum_fit": {"color": "#1e8449", "width": 2},
            "spectrum_pic_brush": (220, 20, 60, 120),
            "dy": {"color": "#7f6000"},
            "zero_line": {"color": "#555555"},
            "baseline_brut": {"color": "#1e1e1e"},
            "baseline_fit": {"color": "#1b4f72"},
            "fft": {"color": "#884ea0"},
            "zoom_data": {"color": "#1e1e1e"},
            "zoom_data_brut": {"color": "#5d6d7e"},
            "zoom_pic_brush": (220, 20, 60, 80),
            "selection_line": {"color": "#1e8449", "width": 1},
            "cross_zoom": {"color": "#c0392b"},
            "text_item": "#1e1e1e",
            "line_t": {"color": "#1e8449", "width": 1},
            "baseline_time": {"color": "#555555", "width": 2},
            "zone_movie": {"color": "#d68910", "style": Qt.DashLine},
            "scatter": {"color": "#1e1e1e", "width": 2},
        },
    },
}

THEMES = {
    "dark": {
        "window": "#2b2b2b",
        "background": "#333333",
        "menu_background": "#222222",
        "text": "#e0e0e0",
        "button_text": "#ffffff",
        "accent": "#0099cc",
        "accent_hover": "#0077aa",
        "accent_pressed": "#005577",
        "input_background": "#444444",
        "selection": "#ffaa55",
        "selection_text": "#ffffff",
        "plot_background": "#333333",
        "axis_pen": "#e0e0e0",
        "grid_alpha": 0.3,
        "pens": {
            "spectrum_data": {"color": "#ffffff"},
            "spectrum_fit": {"color": "#00ff00", "width": 2},
            "spectrum_pic_brush": (255, 0, 0, 120),
            "dy": {"color": "#ffff00"},
            "zero_line": {"color": "#000000"},
            "baseline_brut": {"color": "#ffffff"},
            "baseline_fit": {"color": "#00ffff"},
            "fft": {"color": "#ff00ff"},
            "zoom_data": {"color": "#000000"},
            "zoom_data_brut": {"color": "#ffffff"},
            "zoom_pic_brush": (255, 0, 0, 80),
            "selection_line": {"color": "#00ff00", "width": 1},
            "cross_zoom": {"color": "#ff0000"},
            "text_item": "#ffffff",
            "line_t": {"color": "#00ff00", "width": 1},
            "baseline_time": {"color": "#000000", "width": 2},
            "zone_movie": {"color": "#ffff00", "style": Qt.DashLine},
            "scatter": {"color": "#ffffff", "width": 2},
        },
    },
    "light": {
        "window": "#f2f2f2",
        "background": "#ffffff",
        "menu_background": "#e5e5e5",
        "text": "#1e1e1e",
        "button_text": "#ffffff",
        "accent": "#0066cc",
        "accent_hover": "#005bb5",
        "accent_pressed": "#004c99",
        "input_background": "#f6f6f6",
        "selection": "#ffb347",
        "selection_text": "#1e1e1e",
        "plot_background": "#f7f7f7",
        "axis_pen": "#1e1e1e",
        "grid_alpha": 0.25,
        "pens": {
            "spectrum_data": {"color": "#0c2340"},
            "spectrum_fit": {"color": "#1e8449", "width": 2},
            "spectrum_pic_brush": (220, 20, 60, 120),
            "dy": {"color": "#7f6000"},
            "zero_line": {"color": "#555555"},
            "baseline_brut": {"color": "#1e1e1e"},
            "baseline_fit": {"color": "#1b4f72"},
            "fft": {"color": "#884ea0"},
            "zoom_data": {"color": "#1e1e1e"},
            "zoom_data_brut": {"color": "#5d6d7e"},
            "zoom_pic_brush": (220, 20, 60, 80),
            "selection_line": {"color": "#1e8449", "width": 1},
            "cross_zoom": {"color": "#c0392b"},
            "text_item": "#1e1e1e",
            "line_t": {"color": "#1e8449", "width": 1},
            "baseline_time": {"color": "#555555", "width": 2},
            "zone_movie": {"color": "#d68910", "style": Qt.DashLine},
            "scatter": {"color": "#1e1e1e", "width": 2},
        },
    },
}

Setup_mode = False

folder_start=r"F:\Aquisition_Banc_CEDd"
file_help=r"txt_file\Help.txt"
file_command=r"txt_file\Command.txt"
file_variables=r"txt_file\Variables.txt"
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

def creat_spin_label(spinbox,label_text):
    layout=QHBoxLayout()
    label=QLabel(label_text)
    layout.addWidget(label)
    layout.addWidget(spinbox)
    return layout


from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QProgressBar, QWidget
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

class Variables:
    def __init__(self):
        self.valeurs_boutons = [True,True, True,False,True,True,True]
        self.name_boutons= ["dP\dt","T","Piézo","Image Correlation","M2R","use Movie file","print P"]
        self.liste_chemins_fichiers = []
        self.liste_objets = []
        self.dossier_selectionne=folder_CEDd #r"F:\Aquisition_Banc_CEDd\Fichier_CEDd"
        self.Spectrum_save=None
        self.CEDd_save=None
        self.Gauge_select=None
        self.data_Spectro=None


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


class MainWindow(QMainWindow):
    def __init__(self, folder_start=None):
        super().__init__()

        self.current_theme = "dark"

        # --- Variables "métier" de base ---
        self.Spectrum = None
        self.variables = Variables()
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
        self._setup_commande_box()       # (0, 0)
        self._setup_file_box()           # (4, 0) -> file_spectro/oscilo/movie
        self._setup_tools_tabs()         # (0, 1)
        self._setup_param_pic_box()      # (1, 1)
        self._setup_spectrum_box()       # (0, 2)
        self._setup_ddac_box()           # (0, 3)
        self._setup_tools_checks()       # (3, 2)
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
        self.variables.liste_objets = []  # liste des CEDd chargés (legacy)
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
        if plot_item is None:
            return

        viewbox = plot_item.getViewBox()
        if viewbox is not None:
            viewbox.setBackgroundColor(theme["plot_background"])

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

        if hasattr(self, "theme_toggle_button"):
            self.theme_toggle_button.blockSignals(True)
            self.theme_toggle_button.setChecked(self.current_theme == "light")
            self.theme_toggle_button.setText(
                "Light mode" if self.current_theme == "light" else "Dark mode"
            )
            self.theme_toggle_button.blockSignals(False)

        self.setStyleSheet(self._build_stylesheet(theme))

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
            self.cross_zoom.setBrush(pg.mkBrush(pens["cross_zoom"]))
            self.pg_text_label.setColor(pens["text_item"])

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
            for scatter in (
                self.scatter_P,
                self.scatter_dPdt,
                self.scatter_sigma,
                self.scatter_dlambda,
            ):
                scatter.setPen(scatter_pen)

        if hasattr(self, "pg_text"):
            self.pg_text.setBackgroundColor(theme["plot_background"])

    def _toggle_theme(self, checked: bool):
        self._apply_theme("light" if checked else "dark")

    def _get_run_id(self, ced):
        """Retourne une clé stable pour un CEDd donné."""

        if hasattr(ced, "CEDd_path") and ced.CEDd_path:
            return ced.CEDd_path
        return f"memory_{id(ced)}"

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

        if hasattr(self.RUN, "fps") and self.RUN.fps is not None:
            try:
                titre = "Movie :1e" + str(round(np.log10(self.RUN.fps), 2)) + "fps"
            except Exception as e:
                print("fps log ERROR:", e)
                titre = "Movie :" + str(round(self.RUN.fps, 2)) + "fps"
        else:
            titre = "No Movie"

        self.setWindowTitle(titre)

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

        self.label_CED.setText("CEDd " + name_select + " select")

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

    # ==================================================================
    # ===============   OUTILS D'OBJETS PYQTGRAPH   ====================
    # ==================================================================
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
        

    # ==================================================================
    # ===============  CONFIG FENÊTRE & LAYOUT GÉNÉRAL  ================
    # ==================================================================
    def _setup_main_window(self):
        self.setWindowTitle("Liberté égalité dDACité")
        self.setGeometry(100, 100, 800, 800)

    def _setup_layout_stretch(self):
        # Ajustement des proportions dans la grille
        self.grid_layout.setColumnStretch(0, 0)
        self.grid_layout.setColumnStretch(1, 0)
        self.grid_layout.setColumnStretch(2, 5)
        self.grid_layout.setColumnStretch(3, 5)
        self.grid_layout.setRowStretch(0, 5)
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setRowStretch(2, 2)

    # ==================================================================
    # ===============   SECTION : HELP / COMMANDES (0,0)  ==============
    # ==================================================================
    def _setup_commande_box(self):
        self.CommandeBox = QGroupBox("Help and Commande")
        self.CommandeLayout = QVBoxLayout()

        # Liste d’aide (file_help)
        self.helpLabel = QListWidget()
        self.helpLabel.setMinimumHeight(300)
        self.CommandeLayout.addWidget(self.helpLabel)

        try:
            with open(file_help, "r", encoding="utf-8") as file:
                for line in file:
                    command = line.strip()
                    if not command:
                        continue
                    item = QListWidgetItem(command)
                    text = command
                    if command.startswith("#"):
                        font = QFont("Courier New", 10, QFont.Bold)
                        item.setFont(font)
                        item.setForeground(QColor("royalblue"))
                        text = command[1:]
                    else:
                        font = QFont("Arial", 8, QFont.Bold)
                        item.setFont(font)
                        item.setForeground(QColor("white"))
                    item.setText(text)
                    self.helpLabel.addItem(item)
        except Exception as e:
            self.helpLabel.addItem(f"Error loading file: {e}")
        self.helpLabel.itemDoubleClicked.connect(self.try_command)

        # Liste des commandes python
        self.list_Commande = QListWidget()
        self.list_Commande.setMinimumHeight(250)
        self.list_Commande.itemClicked.connect(self.display_command)
        self.CommandeLayout.addWidget(self.list_Commande)

        self.list_Commande_python = QListWidget()
        try:
            with open(file_command, "r") as file:
                for line in file:
                    command = line.strip()
                    if not command:
                        continue
                    item = QListWidgetItem(command)
                    text = command
                    if command.startswith("#"):
                        font = QFont("Courier New", 11, QFont.Bold)
                        item.setFont(font)
                        item.setForeground(QColor("royalblue"))
                        text = command[1:]
                    elif command.startswith("self"):
                        font = QFont("Arial", 10, QFont.Bold)
                        item.setFont(font)
                        item.setForeground(QColor("white"))
                        text = command[5:]
                    elif command.startswith("."):
                        text = command[1:]
                        if command.endswith(")"):
                            font = QFont("Arial", 8, QFont.Bold)
                            item.setFont(font)
                            item.setForeground(QColor("lightgreen"))
                        else:
                            font = QFont("Arial", 9, QFont.Bold)
                            item.setFont(font)
                            item.setForeground(QColor("tomato"))
                    else:
                        font = QFont("Arial", 10, QFont.Bold)
                        item.setFont(font)
                        item.setForeground(QColor("white"))
                    item.setText(text.split("(")[0])
                    self.list_Commande_python.addItem(QListWidgetItem(command))
                    self.list_Commande.addItem(item)
        except FileNotFoundError:
            self.list_Commande.addItem(f"Error: File '{file_command}' not found")

        # Boutons print / len / clear
        self.ButtonPrint = QPushButton("print(...)")
        self.ButtonPrint.clicked.connect(self.code_print)
        self.CommandeLayout.addWidget(self.ButtonPrint)

        self.ButtonLen = QPushButton("len(...)")
        self.ButtonLen.clicked.connect(self.code_len)
        self.CommandeLayout.addWidget(self.ButtonLen)

        self.ButtonClearcode = QPushButton("Clear")
        self.ButtonClearcode.clicked.connect(self.code_clear)
        self.CommandeLayout.addWidget(self.ButtonClearcode)

        self.CommandeBox.setLayout(self.CommandeLayout)
        self.grid_layout.addWidget(self.CommandeBox, 0, 0, 1, 1)

    # ==================================================================
    # ===============   SECTION : FILE LOADING (4,0 -> 4,3) ============
    # ==================================================================
    def _setup_file_box(self):
        # Chemins de base
        file_spectro = os.path.join(self.folder_start, "Aquisition_ANDOR_Banc_CEDd")
        file_oscilo = os.path.join(self.folder_start, "Aquisition_LECROY_Banc_CEDd")
        file_movie = os.path.join(self.folder_start, "Aquisition_PHANTOME_Banc_CEDd")

        FileBox = QGroupBox("File loading")
        FileBoxLayout = QHBoxLayout()

        # Spectro
        self.select_file_spectro_button = QPushButton("Select file_spectro", self)
        self.select_file_spectro_button.clicked.connect(self.select_spectro_file)
        FileBoxLayout.addWidget(self.select_file_spectro_button)

        self.dir_label_spectro = QLabel("No file spectro selected", self)
        FileBoxLayout.addWidget(self.dir_label_spectro)
        self.loaded_filename_spectro = file_spectro

        # Oscilo
        self.select_file_oscilo_button = QPushButton("Select file_oscilo", self)
        self.select_file_oscilo_button.clicked.connect(self.select_oscilo_file)
        FileBoxLayout.addWidget(self.select_file_oscilo_button)

        self.dir_label_oscilo = QLabel("No file oscilo selected", self)
        FileBoxLayout.addWidget(self.dir_label_oscilo)
        self.loaded_filename_oscilo = file_oscilo

        # Movie
        self.select_file_movie_button = QPushButton("Select file_movie", self)
        self.select_file_movie_button.clicked.connect(self.select_movie_file)
        FileBoxLayout.addWidget(self.select_file_movie_button)

        self.dir_label_movie = QLabel("No file movie selected", self)
        FileBoxLayout.addWidget(self.dir_label_movie)
        self.loaded_filename_movie = file_movie

        # Bouton "Load latest"
        self.load_latest_button = QPushButton("Load Latest File", self)
        self.load_latest_button.clicked.connect(self.load_latest_file)
        FileBoxLayout.addWidget(self.load_latest_button)

        FileBox.setLayout(FileBoxLayout)
        self.grid_layout.addWidget(FileBox, 4, 0, 1, 4)

    # ==================================================================
    # ===============   SECTION : TOOLS TABS (0,1)  ====================
    # ==================================================================
    def _setup_tools_tabs(self):
        ParamBox = QGroupBox("Tools")
        ParamBoxLayout = QVBoxLayout()

        self.theme_toggle_button = QPushButton("Dark mode")
        self.theme_toggle_button.setCheckable(True)
        self.theme_toggle_button.setChecked(False)
        self.theme_toggle_button.toggled.connect(self._toggle_theme)
        ParamBoxLayout.addWidget(self.theme_toggle_button)

        # ---- Bouton pour le kernel Python ----
        self.python_kernel_button = QPushButton("Show Python Kernel")
        self.python_kernel_button.setCheckable(True)
        self.python_kernel_button.setChecked(False)
        self.python_kernel_button.toggled.connect(self.toggle_python_kernel)
        ParamBoxLayout.addWidget(self.python_kernel_button)

        # ---- Tabs ----
        self.tools_tabs = QTabWidget()
        self.tools_tabs.setTabPosition(QTabWidget.West)

        self._setup_tab_data_treatment()   # <<<< Nouveau au lieu de baseline+filtre
        self._setup_tab_gauge()
        self._setup_tab_fit()
        self._setup_tab_fit_selection()

        ParamBoxLayout.addWidget(self.tools_tabs)
        ParamBox.setLayout(ParamBoxLayout)
        self.grid_layout.addWidget(ParamBox, 0, 1, 1, 1)
    
    def _setup_tab_data_treatment(self):
        self.tab_data = QWidget()
        layout = QVBoxLayout(self.tab_data)

        title = QLabel("Data treatment section - - -")
        layout.addWidget(title)

        # ===== Sous-section Baseline =====
        baseline_group = QGroupBox("Baseline")
        bl_layout = QVBoxLayout()

        name_bl = QLabel("baseline section - - -")
        bl_layout.addWidget(name_bl)

        self.deg_baseline_entry = QSpinBox()
        self.deg_baseline_entry.valueChanged.connect(self.setFocus)
        self.deg_baseline_entry.setRange(0, 10)
        self.deg_baseline_entry.setSingleStep(1)
        self.deg_baseline_entry.setValue(0)
        bl_layout.addLayout(creat_spin_label(self.deg_baseline_entry, "°Poly basline"))

        baseline_group.setLayout(bl_layout)
        layout.addWidget(baseline_group)

        # ===== Sous-section Filtre =====
        filtre_group = QGroupBox("Filtre")
        f_layout = QVBoxLayout()

        name_f = QLabel("filtre data section - - -")
        f_layout.addWidget(name_f)

        self.filtre_type_selector = QComboBox(self)
        liste_type_filtre = ['svg', 'fft', 'No filtre']
        self.filtre_type_selector.addItems(liste_type_filtre)
        filtre_colors = ['darkblue', 'darkred', 'darkgrey']
        for ind, col in enumerate(filtre_colors):
            self.filtre_type_selector.model().item(ind).setBackground(QColor(col))
        self.filtre_type_selector.currentIndexChanged.connect(self.f_filtre_select)
        f_layout.addLayout(creat_spin_label(self.filtre_type_selector, "Filtre:"))

        layh1 = QHBoxLayout()
        self.param_filtre_1_name = QLabel("")
        layh1.addWidget(self.param_filtre_1_name)
        self.param_filtre_1_entry = QLineEdit()
        self.param_filtre_1_entry.setText("10")
        layh1.addWidget(self.param_filtre_1_entry)
        f_layout.addLayout(layh1)

        layh2 = QHBoxLayout()
        self.param_filtre_2_name = QLabel("")
        layh2.addWidget(self.param_filtre_2_name)
        self.param_filtre_2_entry = QLineEdit()
        self.param_filtre_2_entry.setText("1")
        layh2.addWidget(self.param_filtre_2_entry)
        f_layout.addLayout(layh2)

        self.f_filtre_select()

        filtre_group.setLayout(f_layout)
        layout.addWidget(filtre_group)

        layout.addStretch()

        self.tools_tabs.addTab(self.tab_data, "Data\nTreatment")

    def _setup_tab_gauge(self):
        self.tab_gauge = QWidget()
        layout = QVBoxLayout(self.tab_gauge)

        name = QLabel("Gauge section - - -")
        layout.addWidget(name)

        layh3 = QHBoxLayout()
        self.name_gauge = QLabel("ADD:")
        layh3.addWidget(self.name_gauge)

        self.Gauge_type_selector = QComboBox(self)
        self.liste_type_Gauge = ['Ruby', 'Sm', 'SrFCl', 'Rhodamine6G', 'Diamond_c12', 'Diamond_c13']
        self.Gauge_type_selector.addItems(self.liste_type_Gauge)
        self.gauge_colors = ['darkred', 'darkblue', 'darkorange', 'limegreen', 'silver', "dimgrey", "k"]
        for ind, col in enumerate(self.gauge_colors[:len(self.liste_type_Gauge)]):
            self.Gauge_type_selector.model().item(ind).setBackground(QColor(col))
        self.Gauge_type_selector.currentIndexChanged.connect(self.f_gauge_select)

        layh3.addWidget(self.Gauge_type_selector)
        layout.addLayout(layh3)

        self.lamb0_entry = QLineEdit()
        layout.addLayout(creat_spin_label(self.lamb0_entry, "\u03BB<sub>0</sub>:"))

        self.name_spe_entry = QLineEdit()
        self.name_spe_entry.editingFinished.connect(self.f_name_spe)
        layout.addLayout(creat_spin_label(self.name_spe_entry, "G.spe:"))

        self.tools_tabs.addTab(self.tab_gauge, "Gauge")

    def _setup_tab_fit(self):
        self.tab_fit = QWidget()
        layout = QVBoxLayout(self.tab_fit)

        name = QLabel("fit param section - - -")
        layout.addWidget(name)

        self.spinbox_cycle = QSpinBox()
        self.spinbox_cycle.valueChanged.connect(self.setFocus)
        self.spinbox_cycle.setRange(0, 10)
        self.spinbox_cycle.setSingleStep(1)
        self.spinbox_cycle.setValue(1)
        layout.addLayout(creat_spin_label(self.spinbox_cycle, "nb<sub>cycle</sub> (Y):"))

        self.sigma_pic_fit_entry = QSpinBox()
        self.sigma_pic_fit_entry.valueChanged.connect(self.setFocus)
        self.sigma_pic_fit_entry.setRange(1, 20)
        self.sigma_pic_fit_entry.setSingleStep(1)
        self.sigma_pic_fit_entry.setValue(5)
        layout.addLayout(creat_spin_label(self.sigma_pic_fit_entry, "nb \u03C3 (R)"))

        self.inter_entry = QDoubleSpinBox()
        self.inter_entry.valueChanged.connect(self.setFocus)
        self.inter_entry.setRange(0.1, 5)
        self.inter_entry.setSingleStep(0.1)
        self.inter_entry.setValue(1)
        layout.addLayout(creat_spin_label(self.inter_entry, "% variation fit"))

        self.tools_tabs.addTab(self.tab_fit, "Fit")

    def _setup_tab_fit_selection(self):
        self.tab_fit_selection = QWidget()
        layout = QVBoxLayout(self.tab_fit_selection)

        name = QLabel("fit_selected_spectra section - - -")
        layout.addWidget(name)

        self.add_btn = QPushButton("Ajouter une zone")
        self.add_btn.clicked.connect(self.add_zone)
        layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Supprimer la zone sélectionnée")
        self.remove_btn.clicked.connect(self.dell_zone)
        self.remove_btn.setEnabled(False)
        layout.addWidget(self.remove_btn)

        self.index_start_entry = QSpinBox()
        self.index_start_entry.setRange(0, 2000)
        self.index_start_entry.setValue(1)
        layout.addLayout(creat_spin_label(self.index_start_entry, "Index start"))

        self.index_stop_entry = QSpinBox()
        self.index_stop_entry.setRange(0, 2000)
        self.index_stop_entry.setValue(10)
        layout.addLayout(creat_spin_label(self.index_stop_entry, "Index stop"))

        self.multi_fit_button = QPushButton("Multi fit")
        self.multi_fit_button.clicked.connect(self._CED_multi_fit)
        layout.addWidget(self.multi_fit_button)

        self.tools_tabs.addTab(self.tab_fit_selection, "Fit selection")

    # ==================================================================
    # ===============   SECTION : TEXT BOX MSG (1,0)  ==================
    # ==================================================================
    def _setup_text_box_msg(self):
        self.text_box_msg = QLabel("Good Luck and Have Fun")
        self.grid_layout.addWidget(self.text_box_msg, 1, 0, 1, 1)

    # ==================================================================
    # ===============   SECTION : PARAM PEAK MODEL (1,1)  ==============
    # ==================================================================
    def _setup_param_pic_box(self):
        ParampicBox = QGroupBox("Model peak")
        self.ParampicLayout = QVBoxLayout()

        self.coef_dynamic_spinbox, self.coef_dynamic_label = [], []

        self.model_pic_type_selector = QComboBox(self)
        self.liste_type_model_pic = ['PseudoVoigt', 'Moffat', 'SplitLorentzian', 'PearsonIV', 'Gaussian']
        self.model_pic_type_selector.addItems(self.liste_type_model_pic)
        model_colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkmagenta']
        for ind, col in enumerate(model_colors):
            self.model_pic_type_selector.model().item(ind).setBackground(QColor(col))
        self.model_pic_type_selector.currentIndexChanged.connect(self.f_model_pic_type)
        self.ParampicLayout.addWidget(self.model_pic_type_selector)

        self.spinbox_sigma = QDoubleSpinBox()
        self.spinbox_sigma.valueChanged.connect(self.setFocus)
        self.spinbox_sigma.setRange(0.01, 80)
        self.spinbox_sigma.setSingleStep(0.01)
        self.spinbox_sigma.setValue(0.25)
        self.ParampicLayout.addLayout(creat_spin_label(self.spinbox_sigma, "\u03C3 :"))

        ParampicBox.setLayout(self.ParampicLayout)
        self.grid_layout.addWidget(ParampicBox, 1, 1, 1, 1)

        self.bit_bypass = True
        self.f_model_pic_type()
        self.bit_bypass = False

    # ==================================================================
    # ===============   SECTION : SPECTRUM PLOTS (0,2)  =================
    # ==================================================================
    
    def _setup_spectrum_box(self):
        self.SpectraBox = QGroupBox("Spectrum")
        SpectraBoxFirstLayout = QVBoxLayout()

        theme = self._get_theme()

        # ================== WIDGET PyQtGraph ==================
        self.pg_spec = pg.GraphicsLayoutWidget()
        self.pg_spec.setBackground(theme["plot_background"])

        # ---- Layout 3x2 : (zoom / baseline / FFT) x (spectrum / dY) ----
        # Row 0, Col 0 : ZOOM
        self.pg_zoom = self.pg_spec.addPlot(row=0, col=0)
        self.pg_zoom.hideAxis('bottom')
        self.pg_zoom.hideAxis('left')
        self.pg_zoom.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

        # Row 1, Col 0 : BASELINE
        self.pg_baseline = self.pg_spec.addPlot(row=1, col=0)
        self.pg_baseline.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.pg_baseline.setLabel('bottom', 'X')
        self.pg_baseline.setLabel('left', 'Intensity')

        # Row 2, Col 0 : FFT
        self.pg_fft = self.pg_spec.addPlot(row=2, col=0)
        self.pg_fft.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.pg_fft.setLabel('bottom', 'f')
        self.pg_fft.setLabel('left', '|F|')

        # Row 2, Col 1 : dY
        self.pg_dy = self.pg_spec.addPlot(row=2, col=1)
        self.pg_dy.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.pg_dy.setLabel('bottom', 'X')
        self.pg_dy.setLabel('left', 'dY')

        # Row 0–1, Col 1 : SPECTRUM
        self.pg_spectrum = self.pg_spec.addPlot(row=0, col=1, rowspan=2)
        self.pg_spectrum.showGrid(x=True, y=True, alpha=theme["grid_alpha"])
        self.pg_spectrum.setLabel('bottom', 'X (U.A)')
        self.pg_spectrum.setLabel('left', 'Y (U.A)')

        # ================== COURBES PERSISTANTES ==================
        # Spectre corrigé
        self.curve_spec_data = self.pg_spectrum.plot(pen=self._mk_pen(theme["pens"]["spectrum_data"]))  # Spectrum.y_corr

        # Fit total (somme des pics)
        self.curve_spec_fit = self.pg_spectrum.plot(pen=self._mk_pen(theme["pens"]["spectrum_fit"]))

        # Pic sélectionné (remplissage)
        self.curve_spec_pic_select = self.pg_spectrum.plot(
            pen=None, fillLevel=0, brush=pg.mkBrush(theme["pens"]["spectrum_pic_brush"])
        )

        # dY
        self.curve_dy = self.pg_dy.plot(pen=self._mk_pen(theme["pens"]["dy"]))
        self.line_dy_zero = self.pg_dy.addLine(y=0, pen=self._mk_pen(theme["pens"]["zero_line"]))

        # Baseline : brut + baseline
        self.curve_baseline_brut = self.pg_baseline.plot(pen=self._mk_pen(theme["pens"]["baseline_brut"]))  # Spectrum.spec
        self.curve_baseline_blfit = self.pg_baseline.plot(pen=self._mk_pen(theme["pens"]["baseline_fit"]))  # Spectrum.blfit

        # FFT
        self.curve_fft = self.pg_fft.plot(pen=self._mk_pen(theme["pens"]["fft"]))

        # Zoom : data + pic sélectionné
        self.curve_zoom_data = self.pg_zoom.plot(pen=self._mk_pen(theme["pens"]["zoom_data"]))
        self.curve_zoom_data_brut = self.pg_zoom.plot(pen=self._mk_pen(theme["pens"]["zoom_data_brut"]))
        self.curve_zoom_pic = self.pg_zoom.plot(
            pen=None, fillLevel=0, brush=pg.mkBrush(theme["pens"]["zoom_pic_brush"])
        )

        # ================== CROIX / LIGNES ==================
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=self._mk_pen(theme["pens"]["selection_line"]))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=self._mk_pen(theme["pens"]["selection_line"]))
        self.pg_spectrum.addItem(self.vline)
        self.pg_spectrum.addItem(self.hline)

        self.cross_zoom = pg.ScatterPlotItem(symbol="+",pen=None, brush=pg.mkBrush(theme["pens"]["cross_zoom"]), size=10)
        self.pg_zoom.addItem(self.cross_zoom)

        # ================== ÉTAT LOGIQUE ==================
        self.Spectrum = None
        self.nb_jauges = 0
        self.list_name_gauges = []
        self.lines = []

        self.Nom_pic = []
        self.Param0 = []
        self.Param_FIT = []
        self.list_text_pic = []
        self.list_y_fit_start = []

        self.index_jauge = -1
        self.index_pic_select = -1
        self.index_spec = 0

        self.X0 = 0
        self.Y0 = 0
        self.Zone_fit = []
        self.X_s = []
        self.X_e = []

        self.y_fit_start = None
        self.model_pic_fit = None

        self.bit_fit_T = False
        self.bit_print_fit_T = False
        self.bit_modif_jauge = False
        self.bit_load_jauge = False
        self.bit_filtre = False

        # Couleurs pour les jauges
        self.c_m = [
            "#ffcccc", "#ffe9cc", "#fdffcc", "#e3ffcc", "#ccffef",
            "#ccf0ff", "#ccd6ff", "#e1ccff", "#fbccff", "#ffcce8"
        ]
        self.color = []

        # ================== INTÉGRATION UI ==================
        SpectraBoxFirstLayout.addWidget(self.pg_spec)

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

        SpectraBoxFirstLayout.addLayout(layout_check)
        self.SpectraBox.setLayout(SpectraBoxFirstLayout)
        self.grid_layout.addWidget(self.SpectraBox, 0, 2, 2, 1)

        # ================== EVENTS PyQtGraph ==================
        # clic sur le spectre principal
        self.pg_spectrum.scene().sigMouseClicked.connect(self._on_pg_spectrum_click)
        self.pg_zoom.scene().sigMouseClicked.connect(self._on_pg_spectrum_click)
        # tu peux aussi connecter sigMouseMoved si tu veux un "hover" au lieu de clic

         # ================== FACTEURS LIGNES / COLONNES ==================
        # 3 lignes : (2, 2, 1)  -> les 2 premières 2x plus grandes que la 3ème
        self._spec_row_factors = (2, 1, 1)
        # 2 colonnes : par exemple 30% (col 0) / 70% (col 1)
        self._spec_col_factors = (1, 2)

        # premier ajustement immédiat
        self._update_graphicslayout_sizes(
            self.pg_spec,
            row_factors=self._spec_row_factors,
            col_factors=self._spec_col_factors,
        )

    # ==================================================================
    # ===============   SECTION : dDAC FIGURE (0,3)  ===================
    # ==================================================================
    def _setup_ddac_box(self):
        group_graphique = QGroupBox("dDAC")
        layout_graphique = QVBoxLayout()

        theme = self._get_theme()

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
        self.pg_ddac.setBackground(theme["plot_background"])

        # Col 0 : P, dP/dt/T, sigma
        self.pg_P = self.pg_ddac.addPlot(row=0, col=0)
        self.pg_P.setLabel('bottom', 'Time (s)')
        self.pg_P.setLabel('left', 'P (GPa)')
        self.pg_P.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

        self.pg_dPdt = self.pg_ddac.addPlot(row=1, col=0)
        self.pg_dPdt.setLabel('bottom', 'Time (s)')
        self.pg_dPdt.setLabel('left', 'dP/dt (GPa/ms), T (K)')
        self.pg_dPdt.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

        self.pg_sigma = self.pg_ddac.addPlot(row=2, col=0)
        self.pg_sigma.setLabel('bottom', 'Time (s)')
        self.pg_sigma.setLabel('left', 'sigma (nm)')
        self.pg_sigma.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

        # Col 1 : IMAGE + Δλ
        self.pg_movie = self.pg_ddac.addPlot(row=0, col=1)
        self.pg_movie.setAspectLocked(True)
        self.pg_movie.hideAxis('bottom')
        self.pg_movie.hideAxis('left')
        self.img_item = pg.ImageItem()
        self.pg_movie.addItem(self.img_item)

        self.pg_text = self.pg_ddac.addViewBox(row=1, col=1)
        self.pg_text.setAspectLocked(False)
        self.pg_text_label = pg.TextItem(color=theme["pens"]["text_item"],)
        self.pg_text.addItem(self.pg_text_label)

        self.pg_dlambda = self.pg_ddac.addPlot(row=2, col=1)
        self.pg_dlambda.setLabel('bottom', 'Spectrum index')
        self.pg_dlambda.setLabel('left', 'Δλ12 (nm)')
        self.pg_dlambda.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

        # ================== COURBES PERSISTANTES ==================
        self.curves_P = []        # une courbe par jauge
        self.curves_dPdt = []
        self.curves_sigma = []
        self.curves_T = []
        self.curve_piezo_list = []
        self.curve_corr_list = []
        self.curves_dlambda = []

        # Ligne verticale pour t sélectionné
        self.line_t_P = pg.InfiniteLine(angle=90, movable=False, pen=self._mk_pen(theme["pens"]["line_t"]))
        self.line_t_dPdt = pg.InfiniteLine(angle=90, movable=False, pen=self._mk_pen(theme["pens"]["line_t"]))
        self.line_t_sigma = pg.InfiniteLine(angle=90, movable=False, pen=self._mk_pen(theme["pens"]["line_t"]))
        self.line_p0 = pg.InfiniteLine(0,angle=0, movable=False, pen=self._mk_pen(theme["pens"]["baseline_time"]))
        self.pg_P.addItem(self.line_p0)
        self.pg_P.addItem(self.line_t_P)
        self.pg_dPdt.addItem(self.line_t_dPdt)
        self.pg_sigma.addItem(self.line_t_sigma)

        self.line_nspec = pg.InfiniteLine(angle=90, movable=False, pen=self._mk_pen(theme["pens"]["line_t"]))
        self.pg_dlambda.addItem(self.line_nspec)

        # Zone temporelle film (bornes)
        self.zone_movie = [None, None]
        self.zone_movie_lines = [
            pg.InfiniteLine(angle=90, movable=False, pen=self._mk_pen(theme["pens"]["zone_movie"])),
            pg.InfiniteLine(angle=90, movable=False, pen=self._mk_pen(theme["pens"]["zone_movie"]))
        ]
        for line in self.zone_movie_lines:
            self.pg_P.addItem(line)


                # ================== MARQUEURS DE CLIC (SCATTER CROIX) ==================
        # Un scatter par graphe pour montrer la position exacte du clic

        self.scatter_P = pg.ScatterPlotItem(
            x=[], y=[],
            pen=self._mk_pen(theme["pens"]["scatter"]),
            brush=None,          # pas de remplissage
            size=10,
            symbol='+'           # croix
        )
        self.pg_P.addItem(self.scatter_P)

        self.scatter_dPdt = pg.ScatterPlotItem(
            x=[], y=[],
            pen=self._mk_pen(theme["pens"]["scatter"]),
            brush=None,
            size=10,
            symbol='+'
        )
        self.pg_dPdt.addItem(self.scatter_dPdt)

        self.scatter_sigma = pg.ScatterPlotItem(
            x=[], y=[],
            pen=self._mk_pen(theme["pens"]["scatter"]),
            brush=None,
            size=10,
            symbol='+'
        )
        self.pg_sigma.addItem(self.scatter_sigma)

        self.scatter_dlambda = pg.ScatterPlotItem(
            x=[], y=[],
            pen=self._mk_pen(theme["pens"]["scatter"]),
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
        self._ddac_row_factors = (3, 1, 1)
        # exemple : colonne temps / colonne image de même largeur
        self._ddac_col_factors = (1, 1)

        self._update_graphicslayout_sizes(
            self.pg_ddac,
            row_factors=self._ddac_row_factors,
            col_factors=self._ddac_col_factors,
        )
    # ==================================================================
    # ===============   SECTION : TOOLS CHECKS (3,2)  ==================
    # ==================================================================
    def _setup_tools_checks(self):
        group_boutons = QGroupBox("Check")
        layout_boutons = QHBoxLayout()

        self.fit_start_box = QCheckBox("Fit (f)", self)
        self.fit_start_box.setChecked(True)
        self.fit_start_box.stateChanged.connect(self.Print_fit_start)
        layout_boutons.addWidget(self.fit_start_box)

        self.var_bouton = []
        for i, valeur in enumerate(self.variables.valeurs_boutons):
            var = QCheckBox(self.variables.name_boutons[i], self)
            var.setChecked(valeur)
            var.stateChanged.connect(self.Update_Print)
            self.var_bouton.append(var)
            layout_boutons.addWidget(var)

        group_boutons.setLayout(layout_boutons)
        self.grid_layout.addWidget(group_boutons, 3, 2, 1, 2)

    # ==================================================================
    # ===============   SECTION : FILE GESTION (2,0)  ==================
    # ==================================================================
    def _setup_file_gestion(self):
        group_fichiers = QGroupBox("File gestion")
        layout_fichiers = QVBoxLayout()

        bouton_dossier = QPushButton("Select folder")
        bouton_dossier.clicked.connect(self.parcourir_dossier)
        layout_fichiers.addWidget(bouton_dossier)

        # Sélecteur d'index de spectre
        self.spinbox_spec_index = QSpinBox()
        self.spinbox_spec_index.setRange(0, 0)      # sera mis à jour quand un CEDd sera chargé
        self.spinbox_spec_index.setValue(0)
        self.spinbox_spec_index.valueChanged.connect(self.on_spec_index_changed)
        layout_fichiers.addLayout(creat_spin_label(self.spinbox_spec_index, "Spec index"))


        self.liste_fichiers = QListWidget()
        self.liste_fichiers.itemDoubleClicked.connect(self.PRINT_CEDd)
        layout_fichiers.addWidget(self.liste_fichiers)

        files_brute = os.listdir(self.variables.dossier_selectionne)
        files = sorted(
            [f for f in files_brute],
            key=lambda x: os.path.getctime(os.path.join(self.variables.dossier_selectionne, x)),
            reverse=True
        )
        self.liste_fichiers.addItems([f for f in files])
        self.variables.liste_chemins_fichiers = [os.path.join(self.variables.dossier_selectionne, f) for f in files]

        self.liste_objets_widget = QListWidget(self)
        self.liste_objets_widget.itemDoubleClicked.connect(self.SELECT_CEDd)
        layout_fichiers.addWidget(self.liste_objets_widget)

        group_fichiers.setLayout(layout_fichiers)
        self.grid_layout.addWidget(group_fichiers, 2, 0, 2, 2)

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

    # ==================================================================
    # ===============   SECTION : PYTHON KERNEL (2,2)  =================
    # ==================================================================
    def _setup_python_kernel(self):
        self.promptBox = QGroupBox("Python Kernel")
        promptLayout = QVBoxLayout()

        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText(
            "Enter your python code here, to use libraries start with CL., example: np.pi -> CL.np.pi..."
        )
        promptLayout.addWidget(self.text_edit)

        self.execute_button = QPushButton("Click to run code (Shift + Entry)", self)
        self.execute_button.clicked.connect(self.execute_code)
        promptLayout.addWidget(self.execute_button)

        self.output_display = QTextEdit(self)
        self.output_display.setReadOnly(True)
        self.output_display.setPlaceholderText("Output print...")
        promptLayout.addWidget(self.output_display)

        self.promptBox.setLayout(promptLayout)

        # On l'ajoute à la grille mais on pourra le déplacer/reconfigurer
        self.grid_layout.addWidget(self.promptBox, 2, 2, 1, 1)
        self.promptBox.hide()  # caché au démarrage



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
    # ===============   SECTION : GAUGE INFO (1,2)  ====================
    # ==================================================================
    def _setup_gauge_info(self):
        self.AddBox = QGroupBox("Gauge information")
        AddLayout = QVBoxLayout()

        layh4 = QHBoxLayout()
        layh4.addWidget(QLabel("P="))
        self.spinbox_P = QDoubleSpinBox()
        self.spinbox_P.setRange(-10.0, 1000.0)
        self.spinbox_P.setSingleStep(0.1)
        self.spinbox_P.setValue(0.0)
        self.spinbox_P.valueChanged.connect(self.spinbox_p_move)
        layh4.addWidget(self.spinbox_P)
        layh4.addWidget(QLabel("GPa"))

        self.deltalambdaP = 0

        layh4.addWidget(QLabel("\u03BB or \u03C3="))
        self.spinbox_x = QDoubleSpinBox()
        self.spinbox_x.setRange(0, 4000)
        self.spinbox_x.setSingleStep(0.1)
        self.spinbox_x.setValue(0.0)
        self.spinbox_x.valueChanged.connect(self.spinbox_x_move)
        layh4.addWidget(self.spinbox_x)
        layh4.addWidget(QLabel("nm or cm<sup>-1<\sup>"))

        layh4.addWidget(QLabel("T="))
        self.spinbox_T = QDoubleSpinBox()
        self.spinbox_T.setRange(0, 3000)
        self.spinbox_T.setSingleStep(1)
        self.spinbox_T.setValue(293)
        self.spinbox_T.valueChanged.connect(self.spinbox_t_move)
        self.spinbox_T.setEnabled(False)
        layh4.addWidget(self.spinbox_T)
        layh4.addWidget(QLabel("K"))

        self.deltalambdaT = 0

        AddLayout.addLayout(layh4)

        self.listbox_pic = QListWidget()
        self.listbox_pic.doubleClicked.connect(self.select_pic)
        AddLayout.addWidget(self.listbox_pic)

        self.AddBox.setLayout(AddLayout)
        self.grid_layout.addWidget(self.AddBox, 2, 2, 1, 1)
        self.bit_modif_PTlambda = False

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

#########################################################################################################################################################################################
#? COMMANDE
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

        if not (clicked_on_spec or clicked_on_zoom):
            return

        if clicked_on_spec:
            vb = self.pg_spectrum.getViewBox()
        else:  # clicked_on_zoom
            vb = self.pg_zoom.getViewBox()

        mouse_point = vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        # Conversion scène -> axes du plot : x = position spectrale, y = intensité
        self.X0, self.Y0 = x, y
        self.vline.setPos(x)
        self.hline.setPos(y)
        self.cross_zoom.setData([x], [y])

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
        self.variables.dossier_selectionne = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier", options=options)
        if self.variables.dossier_selectionne:
            files_brute = os.listdir(self.variables.dossier_selectionne)
            files = sorted(
                [f for f in files_brute],
                key=lambda x: os.path.getctime(os.path.join(self.variables.dossier_selectionne, x)),
                reverse=True
            )
            self.liste_fichiers.clear()
            self.liste_fichiers.addItems(files)
            self.variables.liste_chemins_fichiers = [
                os.path.join(self.variables.dossier_selectionne, f) for f in files
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
    


#########################################################################################################################################################################################
#? MOVIE 
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
                self.variables.data_Spectro = pd.read_csv(
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
        if len(self.variables.data_Spectro.columns) == 2:
            wave = self.variables.data_Spectro.iloc[:, 0]
            Iua = self.variables.data_Spectro.iloc[:, 1]
            wave_unique = np.unique(wave)
            num_spec = len(wave) // len(wave_unique)

            if num_spec >= 1:
                Iua = Iua.values.reshape(num_spec, len(wave_unique)).T
                self.variables.data_Spectro = pd.DataFrame(
                    np.column_stack([wave_unique, Iua]),
                    columns=[0] + [i + 1 for i in range(num_spec)],
                )

        # ---- MISE À JOUR DE LA SPINBOX DE SPECTRE ----
        # nb de spectres = nb de colonnes - 1 (colonne 0 = X)
        n_spec = max(0, self.variables.data_Spectro.shape[1] - 1)

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
                self.variables.Gauge_select.lamb_fit = \
                    self.variables.Gauge_select.inv_f_P(value) + self.deltalambdaT
                self.variables.Gauge_select = self.f_p_move(self.variables.Gauge_select, value)

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
                self.variables.Gauge_select.lamb_fit = value
                self.variables.Gauge_select = self.f_x_move(self.variables.Gauge_select, value)

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
                self.variables.Gauge_select.lamb_fit = round(float(inversefunc(
                    lambda x_: CL.T_Ruby_by_P(x_, P=self.variables.Gauge_select.P,
                                            lamb0R=self.variables.Gauge_select.lamb0),
                    value
                )), 3)
                self.variables.Gauge_select = self.f_t_move(self.variables.Gauge_select, value)

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
            self.variables.Gauge_select=CL.Gauge(name=new_g)
            self.variables.Gauge_select.P=self.spinbox_P.value()
            self.lamb0_entry.setText(str(self.variables.Gauge_select.lamb0))
            self.name_spe_entry.setText(str(self.variables.Gauge_select.name_spe))
            self.f_dell_lines()
            self.f_p_move(self.variables.Gauge_select,value=self.variables.Gauge_select.P)
            self.name_gauge.setText("Add ?")
            self.name_gauge.setStyleSheet("background-color: red;")
            if self.variables.Gauge_select.name == "Ruby":
                self.spinbox_T.setEnabled(True)
                #self.spinbox_T.setValue(self.variables.Gauge_select.T)
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

    def Baseline_spectrum(self):
        param = [float(self.param_filtre_1_entry.text()), float(self.param_filtre_2_entry.text())]
        if self.filtre_type_selector.currentText() == "svg":
            param[0], param[1] = int(param[0]), int(param[1])

        self.Spectrum.Data_treatement(
            deg_baseline=int(self.deg_baseline_entry.value()),
            type_filtre=self.filtre_type_selector.currentText(),
            param_f=param,
            print_data=False  # plus besoin de passer ax=ax_baseline, ax2=ax_fft
        )
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
#########################################################################################################################################################################################
#? COMMANDE LOAD 
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
            chemin_fichier = os.path.join(self.variables.dossier_selectionne, item.text())
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

        self.variables.liste_objets.append(objet_run)
        self.index_select = len(self.variables.liste_objets) - 1
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
        nb_j_old = len(self.Spectrum.Gauges) if self.Spectrum is not None else 0

        if Spectrum is None:
            # On vient de l'UI (listbox)
            if not self.bit_bypass:
                try:
                    self.RUN.Spectra[self.index_spec] = self.Spectrum
                except Exception:
                    self.text_box_msg.setText("ERROR Load Spec")
                    return
                save_index_spec = self.listbox_Spec.currentRow()

            if save_index_spec != -1 or save_index_spec != self.index_spec:
                bypass = False
                if not self.bit_bypass:
                    self.index_spec = save_index_spec
                if self.RUN.Spectra[self.index_spec].bit_fit:
                    self.Spectrum = self.RUN.Spectra[self.index_spec]
                elif self.Spectrum.bit_fit:
                    save_J = copy.deepcopy(self.RUN.Spectra[self.index_spec].Gauges)
                    save_M = copy.deepcopy(self.RUN.Spectra[self.index_spec].model)
                    self.Spectrum = copy.deepcopy(self.RUN.Spectra[self.index_spec])
                    self.Spectrum.Gauges = save_J
                    self.Spectrum.model = save_M
                    bypass = True
                else:
                    self.Spectrum = copy.deepcopy(self.RUN.Spectra[self.index_spec])
        else:
            # Chargement direct d'un objet Spectre
            self.Spectrum = Spectrum
            bypass = True
            self.bit_bypass = False

        # Réinitialise toute la structure de fit (Param0, Nom_pic, etc.)
        if (
            self.Spectrum.bit_fit
            or bypass
            or self.bit_bypass
            or any(G.bit_fit for G in self.Spectrum.Gauges)
        ):
            nb_j = len(self.Spectrum.Gauges)
            self.Zone_fit = [None for _ in range(nb_j)]
            self.X_s = [None for _ in range(nb_j)]
            self.X_e = [None for _ in range(nb_j)]
            self.bit_fit = [False for _ in range(nb_j)]

            # Gestion des zones de fit : uniquement logique (plus de remplissage Matplotlib)
            if self.Spectrum.indexX is not None:
                for i in range(nb_j):
                    self.Zone_fit[i] = self.Spectrum.Gauges[i].indexX
                self.Zone_fit[0] = self.Spectrum.indexX

            # Reconstruction du modèle global si besoin
            if self.Spectrum.model is None and self.Spectrum.Gauges:
                for j in range(nb_j):
                    self.Spectrum.Gauges[j].Update_model()
                    if j == 0:
                        self.Spectrum.model = self.Spectrum.Gauges[0].model
                    else:
                        self.Spectrum.model += self.Spectrum.Gauges[j].model

            # (re)construction des listes de pics
            self.Nom_pic = [[] for _ in range(nb_j)]
            self.list_text_pic = [[] for _ in range(nb_j)]
            self.Param0 = [[] for _ in range(nb_j)]
            self.J = [0 for _ in range(nb_j)]
            self.list_y_fit_start = [[] for _ in range(nb_j)]
            self.list_name_gauges = []

            for i, Jg in enumerate(self.Spectrum.Gauges):
                self.list_name_gauges.append(Jg.name)
                for j, p in enumerate(Jg.pics):
                    self.Nom_pic[i].append(p.name)
                    new_P0, param = p.Out_model()
                    self.Param0[i].append(new_P0 + [p.model_fit])
                    new_name = (
                        p.name
                        + "   X0:"
                        + str(self.Param0[i][-1][0])
                        + "   Y0:"
                        + str(self.Param0[i][-1][1])
                        + "   sigma:"
                        + str(self.Param0[i][-1][2])
                        + "   Coef:"
                        + str(self.Param0[i][-1][3])
                        + " ; Modele:"
                        + str(self.Param0[i][-1][4])
                    )
                    self.J[i] += 1
                    self.list_text_pic[i].append(new_name)
                    y_plot = p.model.eval(param, x=self.Spectrum.wnb)
                    self.list_y_fit_start[i].append(y_plot)

            # Mise à jour des infos de filtre et baseline
            index = self.filtre_type_selector.findText(self.Spectrum.type_filtre)
            if index != -1:
                self.filtre_type_selector.setCurrentIndex(index)
            self.param_filtre_1_entry.setText(str(self.Spectrum.param_f[0]))
            self.param_filtre_2_entry.setText(str(self.Spectrum.param_f[1]))
            self.deg_baseline_entry.setValue(self.Spectrum.deg_baseline)
            self._spectrum_limits_initialized = False
            self._zoom_limits_initialized = False
            self.Baseline_spectrum()  # -> Data_treatement + y_fit_start + refresh graph

            # Mise à jour UI jauge/pics
            self.listbox_pic.clear()
            if self.Spectrum.Gauges:
                self.index_jauge = 0
                self.Gauge_type_selector.setCurrentIndex(
                    self.liste_type_Gauge.index(self.list_name_gauges[self.index_jauge])
                )
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
        self.variables.liste_objets = [ced for ced in self.variables.liste_objets if self._get_run_id(ced) != run_id]

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
        self.variables.Spec_fit = [None for _ in range(self.nb_jauges)]
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
        self.variables.Spectrum_save = copy.deepcopy(self.Spectrum)
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
        self.variables.Spectrum_save = copy.deepcopy(self.Spectrum)

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
        max_col = self.variables.data_Spectro.shape[1] - 1  # dernière colonne de Y
        if n_spec < 1:
            n_spec = 1
        if n_spec > max_col:
            n_spec = max_col

        self.text_box_msg.setText(f"New spec n°{n_spec}")

        x = np.array(self.variables.data_Spectro[0])
        y = np.array(self.variables.data_Spectro[n_spec])

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
        self.variables.Spec_fit.append(None)
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
            self.y_fit_start = None
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
            self._set_viewbox_limits_from_data(vb_zoom, xz_zoom, yz_zoom, padding=0.1)

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

        if self.Spectrum.indexX is not None:
            y_sub = (
                self.Spectrum.y_corr[self.Spectrum.indexX]
                - self.y_fit_start[self.Spectrum.indexX]
            )[indexX]
        else:
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

    def Replace_pic_fit(self):
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
            y_plot = X_pic.model.eval(p, x=self.Spectrum.wnb)

        self.list_y_fit_start[self.index_jauge][self.index_pic_select] = y_plot
        self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select] = X_pic

        self.curve_zoom_pic.setData(self.Spectrum.wnb, y_plot)
        self.curve_spec_pic_select.setData(self.Spectrum.wnb, y_plot)
        self.curve_zoom_data.setData(self.Spectrum.wnb, self.Spectrum.spec - (self.y_fit_start - y_plot) - self.Spectrum.blfit)
        self.curve_zoom_data_brut.setData(S.wnb, S.spec- S.blfit)
        self.Print_fit_start()  # -> recalc global + refresh
      
    def Replace_pic(self):
        if self.index_pic_select is None:
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
        params = X_pic.model.make_params()
        y_plot = X_pic.model.eval(params, x=self.Spectrum.wnb)

        self.list_y_fit_start[self.index_jauge][self.index_pic_select] = y_plot
        self.Spectrum.Gauges[self.index_jauge].pics[self.index_pic_select] = X_pic

  
        self.curve_zoom_pic.setData(self.Spectrum.wnb, y_plot)
        self.curve_spec_pic_select.setData(self.Spectrum.wnb, y_plot)
        self.curve_zoom_data.setData(self.Spectrum.wnb, self.Spectrum.spec - (self.y_fit_start - y_plot) - self.Spectrum.blfit)
        self.curve_zoom_data_brut.setData(self.Spectrum.wnb, self.Spectrum.spec- self.Spectrum.blfit)
        self.Print_fit_start()

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
    # ============================================================
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

    def add_zone(self):
        return print("A CODER")
    
    def dell_zone(self):
        return print("A CODER")
    
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow(folder_start)
    window.show()
    sys.exit(app.exec_())
