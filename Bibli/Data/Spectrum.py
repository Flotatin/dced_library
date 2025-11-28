class Spectrum:
    def __init__(self,wnb,spec,Gauges=[],type_filtre="svg",param_f=[9,2],deg_baseline=0): #lambda0_s=None,lambda0_r=None,lambda0_SrFCl = None,Temperture=False,Model="psdV",pic
        self.x_raw=np.array(wnb)
        self.y_raw=spec
        self.y_corr=spec
        self.x_corr=wnb
        self.param_f=param_f
        self.deg_baseline=deg_baseline
        self.type_filtre=type_filtre
        self.y_filtre,self.blfit=None,None
        #FIT PIC
        self.Gauges=Gauges
        self.lambda_error=round((self.wnb[-1]-self.wnb[0])*0.5/len(self.wnb),4)
        #SYNTHESE
        self.study=pd.DataFrame()