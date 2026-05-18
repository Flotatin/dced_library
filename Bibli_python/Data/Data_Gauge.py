import pandas as pd


class Data_Gauge:
    def __init__(self, name=""):
        self.name = name
        self.pics = []  # list(Peak)
        self.State = dict()  # P , V , T
        self.study = pd.DataFrame()

