import os

import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAction,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

file_help = r"txt_files\Help.txt"
file_command = r"txt_files\Command.txt"


class _StatusMessageProxy:
    """Proxy compatible QLabel (setText/text) vers la status bar."""

    def __init__(self, window, default_text: str = ""):
        self._window = window
        self._text = default_text

    def setText(self, text):
        self._text = str(text)
        if hasattr(self._window, "statusBar"):
            self._window.statusBar().showMessage(self._text)

    def text(self):
        return self._text


def creat_spin_label(spinbox, label_text):
    layout = QHBoxLayout()
    label = QLabel(label_text)
    layout.addWidget(label)
    layout.addWidget(spinbox)
    return layout


class UiLayoutMixin:
    def _setup_main_window(self):
        self.setWindowTitle("0.1 CEDd S&I F.Dembele")
        self.setGeometry(100, 100, 800, 800)

    def _setup_layout_stretch(self):
        # Nouvelle grille compacte : barre de menus en haut, workspace au centre,
        # lignes Peak/Gauge et Fit/Multi en bas.
        self.grid_layout.setColumnStretch(0, 1)   # Fichiers CEDd
        self.grid_layout.setColumnStretch(1, 5)   # Spectrum
        self.grid_layout.setColumnStretch(2, 5)   # dDAC / Movie
        self.grid_layout.setColumnStretch(3, 2)   # panneau secondaire masqué par défaut
        self.grid_layout.setRowStretch(0, 0)
        self.grid_layout.setRowStretch(1, 5)
        self.grid_layout.setRowStretch(2, 0)
        self.grid_layout.setRowStretch(3, 0)
        self.grid_layout.setHorizontalSpacing(6)
        self.grid_layout.setVerticalSpacing(6)
        self.grid_layout.setContentsMargins(6, 6, 6, 6)

    def _setup_file_box(self):
        """Barre supérieure compacte : menus + actions rapides fichier."""

        file_spectro = os.path.join(self.folder_start, "Aquisition_ANDOR_Banc_CEDd")
        file_oscilo = os.path.join(self.folder_start, "Aquisition_LECROY_Banc_CEDd")
        file_movie = os.path.join(self.folder_start, "Aquisition_PHANTOME_Banc_CEDd")

        self.loaded_filename_spectro = file_spectro
        self.loaded_filename_oscilo = file_oscilo
        self.loaded_filename_movie = file_movie

        # Labels conservés pour les services existants qui les mettent à jour.
        self.dir_label_spectro = QLabel("No file spectro", self)
        self.dir_label_oscilo = QLabel("No file oscilo", self)
        self.dir_label_movie = QLabel("No file movie", self)

        menu_bar = self.menuBar()
        self.menu_fichier = menu_bar.addMenu("Fichier")
        self.menu_affichage = menu_bar.addMenu("Affichage")
        self.menu_spectrum = menu_bar.addMenu("Spectrum")
        self.menu_outils = menu_bar.addMenu("Outils")

        self.action_select_spectro = QAction("Ouvrir Spectrum...", self)
        self.action_select_spectro.triggered.connect(self.select_spectro_file)
        self.menu_fichier.addAction(self.action_select_spectro)

        self.action_select_oscilo = QAction("Ouvrir Oscillo...", self)
        self.action_select_oscilo.triggered.connect(self.select_oscilo_file)
        self.menu_fichier.addAction(self.action_select_oscilo)

        self.action_select_movie = QAction("Ouvrir Movie...", self)
        self.action_select_movie.triggered.connect(self.select_movie_file)
        self.menu_fichier.addAction(self.action_select_movie)
        self.menu_fichier.addSeparator()

        self.action_load_latest = QAction("Charger le dernier fichier", self)
        self.action_load_latest.triggered.connect(self.load_latest_file)
        self.menu_fichier.addAction(self.action_load_latest)

        self.action_select_folder = QAction("Sélectionner dossier CEDd...", self)
        self.action_select_folder.triggered.connect(self.parcourir_dossier)
        self.menu_fichier.addAction(self.action_select_folder)
        self.menu_fichier.addSeparator()

        self.action_quit = QAction("Quitter", self)
        self.action_quit.triggered.connect(self.close)
        self.menu_fichier.addAction(self.action_quit)

        self.load_latest_button = QPushButton("Charger dernier", self)
        self.load_latest_button.clicked.connect(self.load_latest_file)
        self.folder_button = QPushButton("Dossier...", self)
        self.folder_button.clicked.connect(self.parcourir_dossier)

        self.spinbox_spec_index = QSpinBox()
        self.spinbox_spec_index.setRange(0, 0)
        self.spinbox_spec_index.setValue(0)
        self.spinbox_spec_index.valueChanged.connect(self.on_spec_index_changed)

        quick_bar = QWidget(self)
        quick_layout = QHBoxLayout(quick_bar)
        quick_layout.setContentsMargins(4, 0, 4, 0)
        quick_layout.setSpacing(6)
        quick_layout.addWidget(self.load_latest_button)
        quick_layout.addWidget(self.folder_button)
        quick_layout.addWidget(QLabel("Spec:"))
        quick_layout.addWidget(self.spinbox_spec_index)
        menu_bar.setCornerWidget(quick_bar, Qt.TopRightCorner)

    def _setup_tools_tabs(self):
        """Crée les lignes compactes et le panneau secondaire à la place des tabs."""

        self._setup_secondary_panel()
        self._setup_tab_data_treatment()
        self._setup_tab_help_and_commande()
        self._setup_tab_tools_checks()
        self._setup_tab_gauge()
        self._populate_compact_menus()

    def _setup_secondary_panel(self):
        self.secondary_panel = QGroupBox("Panneau secondaire")
        secondary_layout = QVBoxLayout(self.secondary_panel)
        secondary_layout.setContentsMargins(6, 6, 6, 6)

        header_layout = QHBoxLayout()
        self.secondary_panel_title = QLabel("Outils")
        header_layout.addWidget(self.secondary_panel_title)
        header_layout.addStretch(1)
        self.secondary_close_button = QPushButton("×")
        self.secondary_close_button.setFixedWidth(28)
        self.secondary_close_button.clicked.connect(self._hide_secondary_panel)
        header_layout.addWidget(self.secondary_close_button)
        secondary_layout.addLayout(header_layout)

        self.secondary_stack = QStackedWidget(self.secondary_panel)
        secondary_layout.addWidget(self.secondary_stack)
        self.secondary_panel.setLayout(secondary_layout)
        self.secondary_panel.hide()
        self.grid_layout.addWidget(self.secondary_panel, 1, 3, 1, 1)

    def _show_secondary_page(self, widget, title: str):
        if widget is None:
            return
        self.secondary_panel_title.setText(title)
        self.secondary_stack.setCurrentWidget(widget)
        self.secondary_panel.show()

    def _hide_secondary_panel(self):
        if (
            hasattr(self, "promptBox")
            and self.secondary_stack.currentWidget() == self.promptBox
            and hasattr(self, "python_kernel_button")
        ):
            self.python_kernel_button.blockSignals(True)
            self.python_kernel_button.setChecked(False)
            self.python_kernel_button.setText("Show Python Kernel")
            self.python_kernel_button.blockSignals(False)
            self.promptBox.hide()
        self.secondary_panel.hide()

    def _sync_checkable_action(self, action, checkbox):
        action.setCheckable(True)
        action.setChecked(checkbox.isChecked())
        action.toggled.connect(checkbox.setChecked)
        checkbox.stateChanged.connect(lambda state, act=action: act.setChecked(state == Qt.Checked))
        return action

    def _populate_compact_menus(self):
        self.theme_toggle_button = QAction("Dark mode", self)
        self.theme_toggle_button.setCheckable(True)
        self.theme_toggle_button.toggled.connect(self._toggle_theme)
        self.menu_affichage.addAction(self.theme_toggle_button)
        self.menu_affichage.addSeparator()

        ddac_menu = self.menu_affichage.addMenu("Courbes dDAC")
        for text, checkbox in (
            ("dP/dt", self.chk_show_dpdt),
            ("T", self.chk_show_T),
            ("Piézo", self.chk_show_piezo),
            ("Image Correlation", self.chk_show_corr),
            ("M2R", self.chk_show_m2r),
            ("P", self.chk_show_P),
        ):
            action = QAction(text, self)
            ddac_menu.addAction(self._sync_checkable_action(action, checkbox))

        sources_menu = self.menu_affichage.addMenu("Sources")
        for text, checkbox in (
            ("Utiliser Movie file", self.chk_use_movie),
            ("Utiliser Oscillo file", self.chk_use_oscillo),
        ):
            action = QAction(text, self)
            sources_menu.addAction(self._sync_checkable_action(action, checkbox))

        spectrum_settings_action = QAction("Baseline / Filtre", self)
        spectrum_settings_action.triggered.connect(
            lambda: self._show_secondary_page(self.tab_data, "Spectrum settings")
        )
        self.menu_spectrum.addAction(spectrum_settings_action)

        help_action = QAction("Help & Commande", self)
        help_action.triggered.connect(
            lambda: self._show_secondary_page(self.tab_help_and_commande, "Help & Commande")
        )
        self.menu_outils.addAction(help_action)

        kernel_action = QAction("Console Python", self)
        kernel_action.triggered.connect(lambda: self.toggle_python_kernel(True))
        self.menu_outils.addAction(kernel_action)

    def _setup_tab_data_treatment(self):
        self.tab_data = QWidget()
        layout = QVBoxLayout(self.tab_data)

        # ===== Sous-section Baseline =====
        title = QLabel("Baseline")
        layout.addWidget(title)
        self.deg_baseline_entry = QSpinBox()
        self.deg_baseline_entry.valueChanged.connect(self.setFocus)
        self.deg_baseline_entry.setRange(0, 10)
        self.deg_baseline_entry.setSingleStep(1)
        self.deg_baseline_entry.setValue(1)
        layout.addLayout(creat_spin_label(self.deg_baseline_entry, "°Poly basline"))


        self.baseline_preview_checkbox = QCheckBox("Print baseline")
        self.baseline_preview_checkbox.setChecked(False)
        self.baseline_preview_checkbox.stateChanged.connect(self._on_print_baseline_toggled)
        layout.addWidget(self.baseline_preview_checkbox)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # ===== Sous-section Filtre =====
        title = QLabel("Filtre")
        layout.addWidget(title)
        self.filtre_type_selector = QComboBox(self)
        self.liste_type_filtre = ['svg', 'fft', 'No filtre']
        self.filtre_type_selector.addItems(self.liste_type_filtre)
        filtre_colors = ['darkblue', 'darkred', 'darkgreen']
        for ind, col in enumerate(filtre_colors):
            self.filtre_type_selector.model().item(ind).setBackground(QColor(col))
        self.filtre_type_selector.currentIndexChanged.connect(self.f_filtre_select)

        layout.addLayout(creat_spin_label(self.filtre_type_selector, "Filtre:"))

        self.param_filtre_1_name = QLabel("")
        layout.addWidget(self.param_filtre_1_name)
        self.param_filtre_1_entry = QLineEdit()
        self.param_filtre_1_entry.setText("1")
        layout.addWidget(self.param_filtre_1_entry)

        layh2 = QHBoxLayout()
        self.param_filtre_2_name = QLabel("")
        layh2.addWidget(self.param_filtre_2_name)
        self.param_filtre_2_entry = QLineEdit()
        self.param_filtre_2_entry.setText("1")
        layh2.addWidget(self.param_filtre_2_entry)
        layout.addLayout(layh2)

        self.f_filtre_select()

        self.secondary_stack.addWidget(self.tab_data)

    def _setup_tab_gauge(self):
        """Lignes compactes Peak/Gauge et Fit/Multi toujours visibles."""

        # ================== Ligne Peak / Gauge ==================
        peak_box = QGroupBox("Peak / Gauge")
        peak_layout = QHBoxLayout(peak_box)
        peak_layout.setContentsMargins(6, 4, 6, 4)
        peak_layout.setSpacing(8)

        peak_layout.addWidget(QLabel("Gauge:"))
        self.name_gauge = QLabel("ADD:")
        peak_layout.addWidget(self.name_gauge)

        self.Gauge_type_selector = QComboBox(self)
        self.liste_type_Gauge = ['Ruby', 'Sm', 'SrFCl', 'Rhodamine6G', 'Diamond_c12', 'Diamond_c13']
        self.Gauge_type_selector.addItems(self.liste_type_Gauge)
        self.gauge_colors = ['darkred', 'darkblue', 'darkorange', 'limegreen', 'silver', "dimgrey", "k"]
        for ind, col in enumerate(self.gauge_colors[:len(self.liste_type_Gauge)]):
            self.Gauge_type_selector.model().item(ind).setBackground(QColor(col))
        self.Gauge_type_selector.currentIndexChanged.connect(self.f_gauge_select)
        peak_layout.addWidget(self.Gauge_type_selector)

        self.lamb0_entry = QLineEdit()
        self.lamb0_entry.setMaximumWidth(90)
        peak_layout.addLayout(creat_spin_label(self.lamb0_entry, "λ<sub>0</sub>:"))

        self.name_spe_entry = QLineEdit()
        self.name_spe_entry.setMaximumWidth(90)
        self.name_spe_entry.editingFinished.connect(self.f_name_spe)
        peak_layout.addLayout(creat_spin_label(self.name_spe_entry, "G.spe:"))

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        peak_layout.addWidget(sep)
        peak_layout.addWidget(QLabel("Model:"))

        self.ParampicLayout = QHBoxLayout()
        self.ParampicLayout.setSpacing(6)
        self.coef_dynamic_spinbox, self.coef_dynamic_label = [], []
        self._coef_dynamic_layouts = []

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
        self.spinbox_sigma.setMaximumWidth(90)
        self.ParampicLayout.addLayout(creat_spin_label(self.spinbox_sigma, "σ:"))
        self.spinbox_sigma.valueChanged.connect(
            lambda _value: self._update_fit_window() if getattr(self, "index_pic_select", None) is not None else None
        )
        controls_layout.addLayout(self.ParampicLayout)

        peak_layout.addLayout(self.ParampicLayout)
        peak_layout.addStretch(1)

        self.bit_bypass = True
        self.f_model_pic_type()
        self.bit_bypass = False
        self.grid_layout.addWidget(peak_box, 2, 0, 1, 4)

        # ================== Ligne Fit / Multi-fit ==================
        fit_box = QGroupBox("Fit / Multi-fit")
        fit_layout = QHBoxLayout(fit_box)
        fit_layout.setContentsMargins(6, 4, 6, 4)
        fit_layout.setSpacing(8)

        if hasattr(self, "fit_start_box"):
            fit_layout.addWidget(self.fit_start_box)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        controls_layout.addWidget(sep)

        # ================== Segment Fit / Multi-fit ==================
        if hasattr(self, "fit_start_box"):
            controls_layout.addWidget(self.fit_start_box)

        self.spinbox_cycle = QSpinBox()
        self.spinbox_cycle.valueChanged.connect(self.setFocus)
        self.spinbox_cycle.setRange(0, 10)
        self.spinbox_cycle.setSingleStep(1)
        self.spinbox_cycle.setValue(1)
        self.spinbox_cycle.setMaximumWidth(70)
        fit_layout.addLayout(creat_spin_label(self.spinbox_cycle, "cycle (Y):"))

        self.sigma_pic_fit_entry = QSpinBox()
        self.sigma_pic_fit_entry.valueChanged.connect(self.setFocus)
        self.sigma_pic_fit_entry.setRange(1, 20)
        self.sigma_pic_fit_entry.setSingleStep(1)
        self.sigma_pic_fit_entry.setValue(2)
        self.sigma_pic_fit_entry.setMaximumWidth(70)
        self.sigma_pic_fit_entry.valueChanged.connect(
            lambda _value: self._update_fit_window() if getattr(self, "index_pic_select", None) is not None else None
        )
        fit_layout.addLayout(creat_spin_label(self.sigma_pic_fit_entry, "nb σ (R):"))

        self.inter_entry = QDoubleSpinBox()
        self.inter_entry.valueChanged.connect(self.setFocus)
        self.inter_entry.setRange(0.1, 5)
        self.inter_entry.setSingleStep(0.1)
        self.inter_entry.setValue(1)
        self.inter_entry.setMaximumWidth(70)
        fit_layout.addLayout(creat_spin_label(self.inter_entry, "var %:"))

        self.index_start_entry = QSpinBox()
        self.index_start_entry.setRange(0, 2000)
        self.index_start_entry.setMaximumWidth(80)
        self.index_start_entry.valueChanged.connect(self._on_fit_spin_changed)
        fit_layout.addLayout(creat_spin_label(self.index_start_entry, "start:"))

        self.index_stop_entry = QSpinBox()
        self.index_stop_entry.setRange(0, 2000)
        self.index_stop_entry.setValue(10)
        self.index_stop_entry.setMaximumWidth(80)
        self.index_stop_entry.valueChanged.connect(self._on_fit_spin_changed)
        fit_layout.addLayout(creat_spin_label(self.index_stop_entry, "stop:"))

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        fit_layout.addWidget(sep)

        self.add_btn = QPushButton("+ zone")
        self.add_btn.clicked.connect(self.add_zone)
        fit_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("- zone")
        self.remove_btn.clicked.connect(self.dell_zone)
        self.remove_btn.setEnabled(False)
        fit_layout.addWidget(self.remove_btn)

        self.btn_zone_dpdt = QPushButton("Zone dP/dt")
        self.btn_zone_dpdt.setCheckable(True)
        self.btn_zone_dpdt.setChecked(False)
        self.btn_zone_dpdt.toggled.connect(self.set_ddac_multi_zone_visibility)
        fit_layout.addWidget(self.btn_zone_dpdt)

        self.chk_multi_fit_fast = QCheckBox("Fast")
        self.chk_multi_fit_fast.setChecked(False)
        self.chk_multi_fit_fast.setToolTip(
            "Accélère le multi-fit en limitant les mises à jour de l'UI Spectrum pendant la boucle."
        )
        fit_layout.addWidget(self.chk_multi_fit_fast)

        self.multi_fit_button = QPushButton("Launch multi fit")
        self.multi_fit_button.clicked.connect(self._CED_multi_fit)
        fit_layout.addWidget(self.multi_fit_button)
        fit_layout.addStretch(1)

        self.grid_layout.addWidget(fit_box, 3, 0, 1, 4)

    def _setup_tab_help_and_commande(self):
        self.tab_help_and_commande = QWidget()
        self.CommandeLayout = QVBoxLayout(self.tab_help_and_commande)

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

        # ---- Bouton pour le kernel Python ----
        self.python_kernel_button = QPushButton("Show Python Kernel")
        self.python_kernel_button.setCheckable(True)
        self.python_kernel_button.setChecked(False)
        self.python_kernel_button.toggled.connect(self.toggle_python_kernel)
        self.CommandeLayout.addWidget(self.python_kernel_button)

        self.secondary_stack.addWidget(self.tab_help_and_commande)

    def _setup_tab_tools_checks(self):
        self.tab_tools_checks = QWidget()

        layout_boutons = QVBoxLayout(self.tab_tools_checks)

        self.fit_start_box = QCheckBox("Fit (f)", self)
        self.fit_start_box.setChecked(True)
        self.fit_start_box.stateChanged.connect(self.Print_fit_start)
        layout_boutons.addWidget(self.fit_start_box)

        self.chk_show_dpdt = QCheckBox("dP\\dt", self)
        self.chk_show_dpdt.setChecked(True)
        self.chk_show_dpdt.stateChanged.connect(self.Update_Print)
        layout_boutons.addWidget(self.chk_show_dpdt)

        self.chk_show_T = QCheckBox("T", self)
        self.chk_show_T.setChecked(True)
        self.chk_show_T.stateChanged.connect(self.Update_Print)
        layout_boutons.addWidget(self.chk_show_T)

        self.chk_show_piezo = QCheckBox("Piézo", self)
        self.chk_show_piezo.setChecked(True)
        self.chk_show_piezo.stateChanged.connect(self.Update_Print)
        layout_boutons.addWidget(self.chk_show_piezo)

        self.chk_show_corr = QCheckBox("Image Correlation", self)
        self.chk_show_corr.setChecked(False)
        self.chk_show_corr.stateChanged.connect(self.Update_Print)
        layout_boutons.addWidget(self.chk_show_corr)

        self.chk_show_m2r = QCheckBox("M2R", self)
        self.chk_show_m2r.setChecked(True)
        self.chk_show_m2r.stateChanged.connect(self.Update_Print)
        layout_boutons.addWidget(self.chk_show_m2r)

        self.chk_use_movie = QCheckBox("use Movie file", self)
        self.chk_use_movie.setChecked(True)
        layout_boutons.addWidget(self.chk_use_movie)

        self.chk_use_oscillo = QCheckBox("use Osillo file", self)
        self.chk_use_oscillo.setChecked(True)
        layout_boutons.addWidget(self.chk_use_oscillo)

        self.input_f_spec = QLineEdit()
        layout_boutons.addLayout(creat_spin_label(self.input_f_spec, "f_spec:"))


        self.chk_show_P = QCheckBox("print P", self)
        self.chk_show_P.setChecked(True)
        self.chk_show_P.stateChanged.connect(self.Update_Print)
        layout_boutons.addWidget(self.chk_show_P)

        # Options exposées via le menu Affichage pour libérer la zone centrale.

    def _setup_text_box_msg(self):
        # Évite de consommer une ligne complète de la grille :
        # on affiche les messages uniquement dans la status bar.
        self.text_box_msg = _StatusMessageProxy(self, "Good Luck and Have Fun")
        self.text_box_msg.setText("Good Luck and Have Fun")

    def _setup_file_gestion(self):
        group_fichiers = QGroupBox("Fichiers CEDd")
        layout_fichiers = QVBoxLayout()
        layout_fichiers.setContentsMargins(6, 6, 6, 6)
        layout_fichiers.setSpacing(6)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.textChanged.connect(self.f_filter_files)
        layout_fichiers.addWidget(self.search_bar)

        self.liste_fichiers = QListWidget()
        self.liste_fichiers.itemDoubleClicked.connect(self.PRINT_CEDd)
        layout_fichiers.addWidget(self.liste_fichiers)

        files_brute = os.listdir(self.dossier_selectionne)
        files = sorted(
            [f for f in files_brute],
            key=lambda x: os.path.getctime(os.path.join(self.dossier_selectionne, x)),
            reverse=True
        )
        self.liste_fichiers.addItems([f for f in files])
        self.liste_chemins_fichiers = [os.path.join(self.dossier_selectionne, f) for f in files]

        self.liste_objets_widget = QListWidget(self)
        self.liste_objets_widget.itemDoubleClicked.connect(self.SELECT_CEDd)
        layout_fichiers.addWidget(self.liste_objets_widget)

        group_fichiers.setLayout(layout_fichiers)
        self.grid_layout.addWidget(group_fichiers, 1, 0, 1, 1)

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

        self.secondary_stack.addWidget(self.promptBox)
        self.promptBox.hide()  # caché au démarrage

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

        layh4.addWidget(QLabel("λ=")) #or σ=
        self.spinbox_x = QDoubleSpinBox()
        self.spinbox_x.setRange(0, 4000)
        self.spinbox_x.setSingleStep(0.1)
        self.spinbox_x.setValue(0.0)
        self.spinbox_x.valueChanged.connect(self.spinbox_x_move)
        layh4.addWidget(self.spinbox_x)
        layh4.addWidget(QLabel("nm")) #or cm<sup>-1<\sup>

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

        self.pic_table = QTableWidget(0, 6)
        self.pic_table.setHorizontalHeaderLabels(["name", "X0", "Y0", "sigma", "coef_spe", "modele"])
        self.pic_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pic_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pic_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pic_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pic_table.verticalHeader().setVisible(False)
        self.pic_table.itemSelectionChanged.connect(self._on_pic_table_selection_changed)
        AddLayout.addWidget(self.pic_table)

        self.AddBox.setLayout(AddLayout)
        self.bit_modif_PTlambda = False

        if hasattr(self, "_spectrum_right_layout"):
            self._spectrum_right_layout.insertWidget(0, self.AddBox)
