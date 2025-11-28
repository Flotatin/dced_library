### CODE POUR COMMUNIQUER AVEC L'OSCILLOSCOPE LECROY, AFFICHER LES TRACES ET LES SAUVEGARDER, VERSION PyQt###
import sys
import pandas as pd
import lecroyscope
import pyqtgraph as pg
from PyQt5.QtGui import QKeyEvent,QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QHBoxLayout, QWidget, QFileDialog , QLineEdit , QListWidget, QCheckBox, QLabel ,QGroupBox, QVBoxLayout ,QComboBox
from PyQt5.QtCore import Qt
from scipy.signal import savgol_filter
from tkinter import filedialog
import os
import re







class OscilloscopeViewer(QMainWindow):
    def __init__(self,folder=r"F:\Aquisition_Banc_CEDd\Aquisition_LECROY_Banc_CEDd",folder_bibli=r"F:\Aquisition_Banc_CEDd\Bibli_Rampe"):
        super().__init__()

        self.folder=folder
     
        
        self.setWindowTitle("Oscilloscope Data Viewer")
        self.setGeometry(100, 100, 2600, 1000)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

       

        # Création du groupe pour les autres composants
        
        layout_c1 = QVBoxLayout()
        

        #group_graphe.setMaximumWidth(300)

        # Plot area using PyQtGraph
        self.plot_widget = pg.PlotWidget(background="#2b2b2b")
        self.plot_widget.setLabel('bottom', 'Time (ms)')
        self.plot_widget.setLabel('left', 'Channel (V)')
        self.plot_widget.showGrid(True, True)
        layout_c1.addWidget(self.plot_widget)
        
        # Voyants
        self.voyant_c1 = QLabel("$t_1=$None $V_1=$None")
        self.voyant_c1.setFont(QFont("Arial", 20))
        self.voyant_c1.setStyleSheet("background-color: grey; border: 1px solid black;")

        self.voyant_c2 = QLabel("$t_2=$None $V_2=$None")
        self.voyant_c2.setFont(QFont("Arial", 20))
        self.voyant_c2.setStyleSheet("background-color: grey; border: 1px solid black;")
        layout_cursor=QHBoxLayout()
        layout_cursor.addWidget(self.voyant_c1)
        layout_cursor.addWidget(self.voyant_c2)
        layout_c1.addLayout(layout_cursor)

        layout_check = QVBoxLayout()
        

        self.text= QLabel("",self)
        layout_c1.addWidget(self.text)

        layout_trace = QVBoxLayout()
        self.fichier_trace_listbox=QListWidget(self)
        self.fichier_trace_listbox.itemClicked.connect(self.load_trace)
        layout_trace.addWidget(self.fichier_trace_listbox)
        if self.folder:
            files = os.listdir(self.folder)  # Obtenir la liste des fichiers dans le dossier
            self.fichier_trace_listbox.clear()
            self.fichier_trace_listbox.addItems(files)

        self.file_name_entry = QLineEdit()
        self.file_name_entry.returnPressed.connect(self.search_trace)
        layout_trace.addWidget( self.file_name_entry)

        self.search_button = QPushButton("Rechercher")
        self.search_button.clicked.connect(self.search_trace)
        layout_trace.addWidget(self.search_button)

        if folder_bibli:
            self.folder_bibli=folder_bibli
        self.files = []
        self.file_dict = {}
        
        # Bouton pour choisir le dossier
        self.btn_select_folder = QPushButton("Sélectionner un dossier", self)
        self.btn_select_folder.clicked.connect(self.load_files)
        layout_trace.addWidget(self.btn_select_folder)
        
        # Combobox pour SCLK
        self.sclk_combo = QComboBox(self)
        self.sclk_combo.currentIndexChanged.connect(self.update_name_combo)
        layout_trace.addWidget(QLabel("Sélectionnez SCLK:"))
        layout_trace.addWidget(self.sclk_combo)
        
        # Combobox pour Nom
        self.name_combo = QComboBox(self)
        self.name_combo.currentIndexChanged.connect(self.update_file_combo)
        layout_trace.addWidget(QLabel("Sélectionnez un Nom:"))
        layout_trace.addWidget(self.name_combo)
        
        # Combobox pour fichier
        self.file_combo = QComboBox(self)
        layout_trace.addWidget(QLabel("Sélectionnez un fichier:"))
        layout_trace.addWidget(self.file_combo)
        
        # Bouton pour afficher le fichier sélectionné
        self.btn_show_file = QPushButton("Afficher le fichier sélectionné", self)
        self.btn_show_file.clicked.connect(self.show_selected_file)
        layout_trace.addWidget(self.btn_show_file)




        
        self.liste_check_load =[ QCheckBox(f"Channel{i} Scope", self) for i in range(1,5)] + [ QCheckBox(f"Channel{i} Load", self) for i in range(1,5)]
        for check in self.liste_check_load:
            check.setChecked(True)
            check.clicked.connect(self.print_plot)
            layout_check.addWidget(check)

        # Bouton pour acquérir et afficher les traces
        self.acquire_button = QPushButton("Acquire and Display", self)
        self.acquire_button.clicked.connect(self.acquire_and_display)
        layout_check.addWidget(self.acquire_button)

        self.name_save_entry= QLineEdit(self)
        layout_check.addWidget(self.name_save_entry)
        # Bouton pour enregistrer les données
        self.save_button = QPushButton("Save Data", self)
        self.save_button.clicked.connect(self.save_data)
        layout_check.addWidget(self.save_button)

        # Bouton pour acquérir et afficher les traces
        self.load_button = QPushButton("Load Trace", self)
        self.load_button.clicked.connect(self.load_trace_button)
        layout_check.addWidget(self.load_button)

        self.clear_button1=QPushButton("CLEAR LOAD", self)
        self.clear_button1.clicked.connect(self.clear_plot_load)
        layout_check.addWidget(self.clear_button1)

        self.clear_button2=QPushButton("CLEAR OSCILO", self)
        self.clear_button2.clicked.connect(self.clear_plot_oscilo)
        layout_check.addWidget(self.clear_button2)

        self.clear_button2=QPushButton("CLEAR BIBLI", self)
        self.clear_button2.clicked.connect(self.clear_plot_bibli)
        layout_check.addWidget(self.clear_button2)



        self.id_entry= QLineEdit(self)
        self.id_entry.setText("100.100.143.2")
        layout_check.addWidget(self.id_entry)
        self.scope_button=QPushButton("Connect scope", self)
        self.scope_button.clicked.connect(self.connect_scope)
        layout_check.addWidget(self.scope_button)
       
        group_trace = QGroupBox("Trace")
        group_trace.setLayout(layout_trace)
        self.layout.addWidget(group_trace)


        group_graphe = QGroupBox("Plot")
        group_graphe.setLayout(layout_c1)
        self.layout.addWidget(group_graphe)

        group_check = QGroupBox("Print or not")
        group_check.setLayout(layout_check)
        self.layout.addWidget(group_check)

         # Création du groupe pour les autres composants


        self.text_scope= QLabel("Scope not connect",self)
        layout_c1.addWidget(self.text)
        layout_c1.addWidget(self.text_scope)
    

        

        self.plot_load = []
        self.plot_bibli = []
        self.plot_oscilo = []
        self.plot_cursor1 = [
            pg.InfiniteLine(angle=90, pen=pg.mkPen('r', width=2, style=Qt.DashLine)),
            pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=2, style=Qt.DashLine))
        ]
        self.plot_cursor2 = [
            pg.InfiniteLine(angle=90, pen=pg.mkPen('b', width=2, style=Qt.DashLine)),
            pg.InfiniteLine(angle=0, pen=pg.mkPen('b', width=2, style=Qt.DashLine))
        ]
        for line in self.plot_cursor1 + self.plot_cursor2:
            self.plot_widget.addItem(line)
        self.cursor_value=[0,0,0,0,0,0]
    
        self.text_cursor="t1: {} t2: {} Dt: {} \nV1: {} V2: {} DV: {}".format(0,0,0,0,0,0)
        
        self.layout.setStretch(0,2)
        self.layout.setStretch(1,6)
        self.layout.setStretch(2,1)



        self.text_code="Welcome"

        self.text_help="no cursor select"

        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
        self.text.setFont(QFont("Arial", 12))
        self.plot_widget.scene().sigMouseClicked.connect(self.clic_cursor)
        self.connect_scope()
        self.load_files(folder=self.folder_bibli)
        self.bit_bypass=False
        # État des voyants
        self.state_c1 = False
        self.state_c2 = False

    def load_files(self,folder=None):
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier bibli rampe")
        if not folder:
            return
        
        self.folder_path = folder
        self.files = os.listdir(folder)
        self.file_dict.clear()
        
        # Expression régulière pour extraire les informations des fichiers
        pattern = re.compile(r"(\d+)_(\d)SCLK_Ramp_(\w+)")
        
        for file in self.files:
            match = pattern.match(file)
            if match:
                X, Y, name = match.groups()
                sclk = X+"e"+Y #int(X) * (10 ** int(Y))
                
                if sclk not in self.file_dict:
                    self.file_dict[sclk] = {}
                if name not in self.file_dict[sclk]:
                    self.file_dict[sclk][name] = []
                
                self.file_dict[sclk][name].append(file)
        
        self.sclk_combo.clear()
        self.sclk_combo.addItems(map(str, sorted(self.file_dict.keys())))
        self.update_name_combo()
    
    def update_name_combo(self):
        self.name_combo.clear()
        sclk = self.get_selected_sclk()
        if sclk is not None and sclk in self.file_dict:
            self.name_combo.addItems(sorted(self.file_dict[sclk].keys()))
        self.update_file_combo()
    
    def update_file_combo(self):
        self.file_combo.clear()
        sclk = self.get_selected_sclk()
        name = self.get_selected_name()
        if sclk is not None and name and name in self.file_dict[sclk]:
            self.file_combo.addItems(self.file_dict[sclk][name])
    
    def get_selected_sclk(self):
        try:
            return self.sclk_combo.currentText()#int(self.sclk_combo.currentText())
        except ValueError:
            return None
    
    def get_selected_name(self):
        return self.name_combo.currentText()
    
    def show_selected_file(self):
        c_t=["k","gold","m","b","chartreuse"]
        selected_file = self.file_combo.currentText()
        if selected_file:
            try:
                oscilo=pd.read_csv(os.path.join(self.folder_path, selected_file), sep='\s+', skipfooter=0, engine='python')
                if self.plot_bibli != []:
                    for p in self.plot_bibli:
                        if p is not None:
                            p.remove()
                self.plot_bibli =[]
                for i in range(1,5):
                    curve = self.plot_widget.plot(
                        oscilo["Time"]*1e3,
                        savgol_filter(oscilo[f"Channel{i}"],50,2),
                        pen=pg.mkPen(c_t[i])
                    )
                    self.plot_bibli.append(curve)
                self.f_autoscale()
            except Exception as e:
                print(e)
                self.text_code="Error in loading"
        
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
    


    def connect_scope(self):
        try:
            self.scope = lecroyscope.Scope( self.id_entry.text())  # IP address of the scope "169.254.13.6"
            self.text_scope.setText(f"Scope connect ID: {self.scope.id}")

        except Exception as e:
            self.text_code ="ERROR:"+str(e)
            self.text_scope.setText(f"Scope fail")
        
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
    
    def search_trace(self):# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - SEARCH JCPDS - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        file_name = self.file_name_entry.text()

        self.fichier_trace_listbox.clear()
        for file in os.listdir(self.folder):
            if file_name in file:
                self.fichier_trace_listbox.addItem(file)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_A:
            self.state_c1 = not self.state_c1
            self.voyant_c1.setStyleSheet("background-color: red;" if self.state_c1 else "background-color: grey;")
            #self.text.setText("Curseur 1 sélectionné" if self.state_c1 else "Curseur 1 désélectionné")

        elif key == Qt.Key_B:
            self.state_c2 = not self.state_c2
            self.voyant_c2.setStyleSheet("background-color: blue;" if self.state_c2 else "background-color: grey;")
            #self.text.setText("Curseur 2 sélectionné" if self.state_c2 else "Curseur 2 désélectionné")

    def clic_cursor(self, event):
        self.setFocus()
        if self.state_c1 or self.state_c2:
            pos = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x = pos.x()
            y = pos.y()
            if self.state_c1:
                c = 0
                self.plot_cursor1[0].setPos(x)
                self.plot_cursor1[1].setPos(y)
                self.voyant_c1.setText(f"t1 = {round(x,3)}ms V1 = {round(y,3)}V")
            elif self.state_c2:
                c = 1
                self.plot_cursor2[0].setPos(x)
                self.plot_cursor2[1].setPos(y)
                self.voyant_c2.setText(f"t2 = {round(x,3)}ms V2 = {round(y,3)} V")

            self.cursor_value[0 + c] = round(x, 3)
            self.cursor_value[3 + c] = round(y, 3)
            self.cursor_value[2] = round(self.cursor_value[1] - self.cursor_value[0], 3)
            self.cursor_value[5] = round(self.cursor_value[4] - self.cursor_value[3], 3)
            self.text_cursor = f"D21t = {self.cursor_value[2]}ms \n D21V = {self.cursor_value[5]}V"

            self.text.setText(self.text_cursor + "\n" + self.text_code + "\n" + self.text_help)

    def f_autoscale(self):
        def extract_plot(plot):
            xs, ys = [], []
            for curve in plot:
                data = curve.getData()
                if data is not None:
                    xs.extend(data[0])
                    ys.extend(data[1])
            return xs, ys

        x_load, y_load = extract_plot(self.plot_load)
        x_oscilo, y_oscilo = extract_plot(self.plot_oscilo)
        x_bibli, y_bibli = extract_plot(self.plot_bibli)
        x_data = x_load + x_oscilo + x_bibli
        y_data = y_load + y_oscilo + y_bibli
        if x_data:
            self.plot_widget.setXRange(min(x_data), max(x_data))
        if y_data:
            self.plot_widget.setYRange(min(y_data), max(y_data))

    def print_plot(self):
        for i in range(4):
            if i < len(self.plot_oscilo):
                self.plot_oscilo[i].setVisible(self.liste_check_load[i].isChecked())
            if i < len(self.plot_load):
                self.plot_load[i].setVisible(self.liste_check_load[i+4].isChecked())

        self.f_autoscale()

    def clear_plot_load(self):
        if self.plot_load:
            for p in self.plot_load:
                self.plot_widget.removeItem(p)
        self.plot_load = []
        self.f_autoscale()

    def clear_plot_oscilo(self):
        if self.plot_oscilo:
            for p in self.plot_oscilo:
                self.plot_widget.removeItem(p)
        self.plot_oscilo = []
        self.f_autoscale()

    def clear_plot_bibli(self):
        if self.plot_bibli:
            for p in self.plot_bibli:
                self.plot_widget.removeItem(p)
        self.plot_bibli = []
        self.f_autoscale()

    def load_trace(self,item):
        c_t=["k","darkorange","darkred","darkblue","darkgreen"]
        #file =filedialog.askopenfilename(title="Sélectionner TRACE")
        #if file:
        chemin_fichier = os.path.join(self.folder, item.text())
        #self.liste_objets_widget.setCurrentRow(item)
        index = self.fichier_trace_listbox.row(item)
        
        if index  :
            try:
                oscilo=pd.read_csv(chemin_fichier, sep='\s+', skipfooter=0, engine='python')
                if self.plot_load != []:
                    for p in self.plot_load:
                        if p is not None:
                            p.remove()
                self.plot_load =[]
                for i in range(1,5):
                    curve = self.plot_widget.plot(
                        oscilo["Time"]*1e3,
                        savgol_filter(oscilo[f"Channel{i}"],50,2),
                        pen=pg.mkPen(c_t[i])
                    )
                    self.plot_load.append(curve)
                self.f_autoscale()
            except Exception as e:
                print(e)
                self.text_code="Error in loading"
        
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)
    
    def load_trace_button(self):
        c_t=["k","darkorange","darkred","darkblue","darkgreen"]
        file =filedialog.askopenfilename(title="Sélectionner TRACE") 
        if file:
            oscilo=pd.read_csv(file, sep='\s+', skipfooter=0, engine='python')
            if self.plot_load != []:
                for p in self.plot_load:
                    if p is not None:
                        p.remove(file)
            self.plot_load =[]
            for i in range(1,5):
                curve = self.plot_widget.plot(
                    oscilo["Time"]*1e3,
                    savgol_filter(oscilo[f"Channel{i}"],50,2),
                    pen=pg.mkPen(c_t[i])
                )
                self.plot_load.append(curve)
            self.f_autoscale()

        self.text.setText(self.text_cursor + "\n" +self.text_code)


    def acquire_and_display(self):
        # Code pour acquérir les traces (à partir de votre code existant)
        trace_group = self.scope.read(1, 2, 3, 4)
        time = trace_group.time  # time values are the same for all traces

        # Effacer le contenu précédent du graphique
        if self.plot_oscilo:
            for p in self.plot_oscilo:
                self.plot_widget.removeItem(p)
        self.plot_oscilo = []
        # Afficher les nouvelles traces sur le graphique
        c_t=["k","gold","r","b","g"]
        for i in range(1, len(trace_group) + 1):
            curve = self.plot_widget.plot(trace_group[i].x*1e3, trace_group[i].y, pen=pg.mkPen(c_t[i]))
            self.plot_oscilo.append(curve)

        self.f_autoscale()

        self.text.setText(self.text_cursor + "\n" +self.text_code)
    
    def save_data(self):
        trace_group = self.scope.read(1, 2, 3, 4)
        time = trace_group.time  # time values are the same for all traces
        df = pd.DataFrame({"Time" :pd.Series(time), 
                   "Channel1" :pd.Series(trace_group[1].y),
                   "Channel2" :pd.Series(trace_group[2].y),
                   "Channel3" :pd.Series(trace_group[3].y),
                   "Channel4" :pd.Series(trace_group[4].y),
                  })
        #file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "Data Files (*.dat)")
        file_path =os.path.join(self.folder,str(self.name_save_entry.text()))
        if file_path:
            with open(file_path, 'w') as file2write:
                file2write.write(df.to_string())
            print(f"Data saved to {file_path}")
            self.text_code=f"Data saved to {file_path}"
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)

    def select_folder(self):
            # Fonction pour parcourir un dossier et afficher ses fichiers
        options = QFileDialog.Options()
        self.folder = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier", options=options)
        if self.folder:
            files = os.listdir(self.folder)  # Obtenir la liste des fichiers dans le dossier
            self.fichier_trace_listbox.clear()
            self.fichier_trace_listbox.addItems(files)
            self.text_code=f"Folder select {self.folder}"
        self.text.setText(self.text_cursor + "\n"+ self.text_code  + "\n"+ self.text_help)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = OscilloscopeViewer()
    viewer.show()
    sys.exit(app.exec_())