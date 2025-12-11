import dill
import os

def SAVE_CEDd(file_CEDd,bit_try=False):
    if file_CEDd:
        if bit_try==True:
            try:
                dill.dump( file_CEDd, open( file_CEDd.CEDd_path, "wb" ) )
            except Exception as e:
                print("ERROR : ",e," in SAVE_CEDd")
        else:
            dill.dump( file_CEDd, open( file_CEDd.CEDd_path, "wb" ) )
            
def LOAD_CEDd(CEDd_path,bit_try=False):
    if CEDd_path:
        if bit_try==True:
            try:
                CEDd = dill.load( open( CEDd_path, "rb" ) )
                CEDd.CEDd_path=CEDd_path
                return CEDd
            except Exception as e:
                print("ERROR : ",e," in LOAD_CEDd")
        else:
            CEDd = dill.load( open( CEDd_path, "rb" ) )
            CEDd.CEDd_path=CEDd_path
            return CEDd

def Load_last(Folder,extend=None,file=True):
    if file ==True:
        if extend != None :
            file_names = [f for f in os.listdir(Folder) if os.path.isfile(os.path.join(Folder, f)) and extend in f]
        else:
            file_names = [f for f in os.listdir(Folder) if os.path.isfile(os.path.join(Folder, f))]
    else:
        file_names = [f for f in os.listdir(Folder)]
    if file_names:
        file_names.sort(key=lambda f: os.path.getmtime(os.path.join(Folder, f)))
        latest_file_name = file_names[-1]
        latest_file_path = os.path.join(Folder, latest_file_name)
    return latest_file_path, latest_file_name

""" ------------------------------------- CLASS OSCILOSCOPE LECROY-------------------------------------"""
import lecroyscope
import matplotlib.pyplot as plt
import pandas as pd

class Oscillo_autosave:
    def __init__(self,IP="100.100.143.2",folder=r"F:\Aquisition_Banc_CEDd\Aquisition_LECROY_Banc_CEDd"):
        self.scope = lecroyscope.Scope(IP)  # IP address of the scope
        self.folder=folder
        print(f"Scope ID: {self.scope.id}")
        print("dossier d'enregistrement"+self.folder)

    def Print(self):
        # Afficher les nouvelles traces sur le graphique
        trace_group = self.scope.read(1, 2, 3, 4)
        for i in range(1, len(trace_group) + 1):
            plt.plot(trace_group[i].x, trace_group[i].y, '.-')
        plt.xlabel('Time')
        plt.ylabel(f'Channel {i}')
        plt.title(f'Trace for Channel {i}')
        plt.grid(True)
        plt.show()


    def save(self,name):
        trace_group = self.scope.read(1, 2, 3, 4)
        time = trace_group.time  # time values are the same for all traces
        df = pd.DataFrame({"Time" :pd.Series(time), 
                   "Channel1" :pd.Series(trace_group[1].y),
                   "Channel2" :pd.Series(trace_group[2].y),
                   "Channel3" :pd.Series(trace_group[3].y),
                   "Channel4" :pd.Series(trace_group[4].y),
                  })
        file_path =os.path.join(self.folder,name)
        if file_path:
            with open(file_path, 'w') as file2write:
                file2write.write(df.to_string())
            print(f"Data saved to {file_path}")
