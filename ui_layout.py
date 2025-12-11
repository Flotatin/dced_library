import os

import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

file_help = r"txt_files\Help.txt"
file_command = r"txt_files\Command.txt"

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
        # Ajustement des proportions dans la grille
        self.grid_layout.setColumnStretch(0, 0)
        self.grid_layout.setColumnStretch(1, 0)
        self.grid_layout.setColumnStretch(2, 5)
        self.grid_layout.setColumnStretch(3, 5)
        self.grid_layout.setRowStretch(0, 5)
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setRowStretch(2, 2)
        self.grid_layout.setHorizontalSpacing(6)
        self.grid_layout.setVerticalSpacing(6)
        self.grid_layout.setContentsMargins(6, 6, 6, 6)

    def _setup_file_box(self):
        # Chemins de base
        file_spectro = os.path.join(self.folder_start, "Aquisition_ANDOR_Banc_CEDd")
        file_oscilo = os.path.join(self.folder_start, "Aquisition_LECROY_Banc_CEDd")
        file_movie = os.path.join(self.folder_start, "Aquisition_PHANTOME_Banc_CEDd")

        FileBox = QGroupBox("File loading")
        FileBoxLayout = QHBoxLayout()

        # Spectro
        self.select_file_spectro_button = QPushButton("Spectrum File", self)
        self.select_file_spectro_button.clicked.connect(self.select_spectro_file)
        FileBoxLayout.addWidget(self.select_file_spectro_button)

        self.dir_label_spectro = QLabel("No file spectro", self)
        FileBoxLayout.addWidget(self.dir_label_spectro)
        self.loaded_filename_spectro = file_spectro

        # Oscilo
        self.select_file_oscilo_button = QPushButton("Oscillo File", self)
        self.select_file_oscilo_button.clicked.connect(self.select_oscilo_file)
        FileBoxLayout.addWidget(self.select_file_oscilo_button)

        self.dir_label_oscilo = QLabel("No file oscilo", self)
        FileBoxLayout.addWidget(self.dir_label_oscilo)
        self.loaded_filename_oscilo = file_oscilo

        # Movie
        self.select_file_movie_button = QPushButton("Movie File", self)
        self.select_file_movie_button.clicked.connect(self.select_movie_file)
        FileBoxLayout.addWidget(self.select_file_movie_button)

        self.dir_label_movie = QLabel("No file movie", self)
        FileBoxLayout.addWidget(self.dir_label_movie)
        self.loaded_filename_movie = file_movie

        # Bouton "Load latest"
        self.load_latest_button = QPushButton("Load Latest File", self)
        self.load_latest_button.clicked.connect(self.load_latest_file)
        FileBoxLayout.addWidget(self.load_latest_button)

        FileBox.setLayout(FileBoxLayout)
        self.grid_layout.addWidget(FileBox, 3, 3, 1, 1)

    def _setup_tools_tabs(self):
        ParamBox = QGroupBox("Tools")
        ParamBoxLayout = QVBoxLayout()

        self.theme_toggle_button = QPushButton("Dark mode")
        self.theme_toggle_button.setCheckable(True)
        self.theme_toggle_button.setChecked(False)
        self.theme_toggle_button.toggled.connect(self._toggle_theme)
        ParamBoxLayout.addWidget(self.theme_toggle_button)

        # ---- Tabs ----
        self.tools_tabs = QTabWidget()
        self.tools_tabs.setTabPosition(QTabWidget.West)

        self._setup_tab_gauge()
        self._setup_tab_fit()
        self._setup_tab_data_treatment()
        self._setup_tab_help_and_commande()
        self._setup_tab_tools_checks()

        ParamBoxLayout.addWidget(self.tools_tabs)
        ParamBox.setLayout(ParamBoxLayout)
        self.grid_layout.addWidget(ParamBox, 0, 0, 2, 1)

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

        """layh = QHBoxLayout()
        self.deg_baseline_auto = QCheckBox("Auto")
        self.deg_baseline_auto.setChecked(True)
        self.deg_baseline_auto.stateChanged.connect(self.f_auto_deg_baseline)
        layh.addWidget(self.deg_baseline_auto)
        layout.addLayout(layh)"""

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

        self.tools_tabs.addTab(self.tab_data, "Spectrum")

    def _setup_tab_gauge(self):
        # === Onglet Gauge ===
        self.tab_gauge = QWidget()
        layout = QVBoxLayout(self.tab_gauge)

        # --- Partie "Gauge" ---
        title = QLabel("Gauge")
        layout.addWidget(title)

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
        layout.addLayout(creat_spin_label(self.lamb0_entry, "λ<sub>0</sub>:"))

        self.name_spe_entry = QLineEdit()
        self.name_spe_entry.editingFinished.connect(self.f_name_spe)
        layout.addLayout(creat_spin_label(self.name_spe_entry, "G.spe:"))

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # --- Partie "Model peak" dans le même onglet ---

        title = QLabel("Peak")
        layout.addWidget(title)
        self.ParampicLayout = QVBoxLayout()

        # Combobox type de pic
        self.coef_dynamic_spinbox, self.coef_dynamic_label = [], []

        self.model_pic_type_selector = QComboBox(self)
        self.liste_type_model_pic = ['PseudoVoigt', 'Moffat', 'SplitLorentzian', 'PearsonIV', 'Gaussian']
        self.model_pic_type_selector.addItems(self.liste_type_model_pic)
        model_colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkmagenta']
        for ind, col in enumerate(model_colors):
            self.model_pic_type_selector.model().item(ind).setBackground(QColor(col))
        self.model_pic_type_selector.currentIndexChanged.connect(self.f_model_pic_type)

        self.ParampicLayout.addWidget(self.model_pic_type_selector)

        # Spin σ
        self.spinbox_sigma = QDoubleSpinBox()
        self.spinbox_sigma.valueChanged.connect(self.setFocus)
        self.spinbox_sigma.setRange(0.01, 80)
        self.spinbox_sigma.setSingleStep(0.01)
        self.spinbox_sigma.setValue(0.25)
        self.ParampicLayout.addLayout(creat_spin_label(self.spinbox_sigma, "σ :"))
        self.spinbox_sigma.valueChanged.connect(
            lambda _value: self._update_fit_window() if getattr(self, "index_pic_select", None) is not None else None
        )

        # Ajout du groupbox "Model peak" dans le même tab
        layout.addLayout(self.ParampicLayout)

        # Initialisation des coefficients dynamiques du modèle
        self.bit_bypass = True
        self.f_model_pic_type()   # va utiliser self.ParampicLayout, self.model_pic_type_selector, etc.
        self.bit_bypass = False

        # Enfin : ajout de l’onglet dans le QTabWidget
        self.tools_tabs.addTab(self.tab_gauge, "Gauge & Peak")

    def _setup_tab_fit(self):
        self.tab_fit = QWidget()
        layout = QVBoxLayout(self.tab_fit)

        name = QLabel("fit param")
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
        self.sigma_pic_fit_entry.setValue(2)
        self.sigma_pic_fit_entry.valueChanged.connect(
            lambda _value: self._update_fit_window() if getattr(self, "index_pic_select", None) is not None else None
        )

        layout.addLayout(creat_spin_label(self.sigma_pic_fit_entry, "nb σ (R)"))

        self.inter_entry = QDoubleSpinBox()
        self.inter_entry.valueChanged.connect(self.setFocus)
        self.inter_entry.setRange(0.1, 5)
        self.inter_entry.setSingleStep(0.1)
        self.inter_entry.setValue(1)
        layout.addLayout(creat_spin_label(self.inter_entry, "% variation fit"))

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        name = QLabel("Multi fit")
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

        self.multi_fit_button = QPushButton("Launch multi fit")
        self.multi_fit_button.clicked.connect(self._CED_multi_fit)
        layout.addWidget(self.multi_fit_button)

        self.tools_tabs.addTab(self.tab_fit, "Fit")

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

        self.tools_tabs.addTab(self.tab_help_and_commande, "Help & Commande")

    def _setup_tab_tools_checks(self):
        self.tab_tools_checks = QWidget()

        layout_boutons = QVBoxLayout(self.tab_tools_checks)

        self.fit_start_box = QCheckBox("Fit (f)", self)
        self.fit_start_box.setChecked(True)
        self.fit_start_box.stateChanged.connect(self.Print_fit_start)
        layout_boutons.addWidget(self.fit_start_box)

        self.var_bouton = []
        for i, valeur in enumerate(self.valeurs_boutons):
            var = QCheckBox(self.name_boutons[i], self)
            var.setChecked(valeur)
            var.stateChanged.connect(self.Update_Print)
            self.var_bouton.append(var)
            layout_boutons.addWidget(var)

        self.tools_tabs.addTab(self.tab_tools_checks, "Tools & Check")

    def _setup_text_box_msg(self):
        self.text_box_msg = QLabel("Good Luck and Have Fun")
        self.grid_layout.addWidget(self.text_box_msg, 3, 2, 1, 1)

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

        self.pg_text_label = pg.TextItem("Liberté \n égalité \n dDACité")
        self.pg_text.addItem(self.pg_text_label)
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
        self.line_p0 = pg.InfiniteLine(0, angle=0, movable=False)
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

        # On force une taille de colonnes/hauteurs pour mieux voir
        self._ddac_row_factors = (1, 1, 1)
        self._ddac_col_factors = (3, 7)
        self._update_graphicslayout_sizes(
            self.pg_ddac,
            row_factors=self._ddac_row_factors,
            col_factors=self._ddac_col_factors,
        )

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

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.textChanged.connect(self.f_filter_files)
        layout_fichiers.addWidget(self.search_bar)

        group_fichiers.setLayout(layout_fichiers)
        self.grid_layout.addWidget(group_fichiers, 2, 0, 2, 2)

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

        layh4.addWidget(QLabel("λ or σ="))
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
