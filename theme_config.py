from string import Template

from PyQt5.QtCore import Qt

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

/* GroupBox */
QGroupBox {
    font-weight: bold;
    border: 1px solid ${accent};
    border-radius: 6px;
    margin-top: 8px;
    padding: 6px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
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
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 9pt;
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
            "cross_zoom": "#ff0000",
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
            "cross_zoom": "#c0392b",
            "text_item": "#1e1e1e",
            "line_t": {"color": "#1e8449", "width": 1},
            "baseline_time": {"color": "#555555", "width": 2},
            "zone_movie": {"color": "#d68910", "style": Qt.DashLine},
            "scatter": {"color": "#1e1e1e", "width": 2},
        },
    },
}
