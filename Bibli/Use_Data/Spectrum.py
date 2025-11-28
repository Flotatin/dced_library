class Spectrum:
    def __init__(self,wnb,spec,Gauges=[],type_filtre="svg",param_f=[9,2],deg_baseline=0,E=None): #lambda0_s=None,lambda0_r=None,lambda0_SrFCl = None,Temperture=False,Model="psdV",pic
        self.wnb=np.array(wnb)
        self.spec=spec
        self.spec_brut=spec
        self.param_f=param_f
        self.deg_baseline=deg_baseline
        self.type_filtre=type_filtre
        self.y_filtre,self.blfit=None,None
        self.x_corr=wnb
        self.Data_treatement(print_data=False)
        self.E=E
        self.X=None
        self.Y=None
        self.dY=None
        self.bit_model=False
        self.model=None
        self.fit="Fit Non effectué"
        self.bit_fit=False
        self.lamb_fit=None
        self.indexX=None
        #FIT PIC
        self.Gauges=Gauges
        self.lambda_error=round((self.wnb[-1]-self.wnb[0])*0.5/len(self.wnb),4)
        #SYNTHESE
        self.study=pd.DataFrame()
        self.help="Spectre: etude de spectre"
    
    def Corr(self,list_lamb0):
        for i in range(len(self.Gauges)):
            if list_lamb0[i] !=None :
                self.Gauges.lamb0=list_lamb0[i]
            self.Gauges[i].Calcul(input_spe=self.Gauges,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)

    def Print(self,ax=None,ax2=None,return_fig=False):
        if ax == None:
            print_fig=True
            fig, (ax,ax2) =plt.subplots(ncols=1,nrows=2,figsize=(8,4),gridspec_kw={'height_ratios': [0.85, 0.15]})      
        else:
            print_fig=False
        ax.plot(self.x_corr,self.blfit,'-.', c='g', markersize=1) #label="Baseline"
        ax.plot(self.wnb,self.spec,'-',color='lightgray',markersize=4) #,label='Data Brut'
        ax.plot(self.x_corr,self.y_corr+self.blfit,'.',color='black',markersize=3) #,label='Data Fit + Baseline'
        
        for G in self.Gauges:
            if G.bit_fit ==True:
                titre_fiti= G.name+":$\lambda_0=$"+str(G.lamb0)
                if G.indexX is None :
                    bf= self.blfit
                else:
                    bf=self.blfit[G.indexX]
                if G.color_print[0] != None:
                    ax.plot(G.X,G.Y,'--',label=titre_fiti,markersize=1,c=G.color_print[0])
                    if ax2 is not None:
                        ax2.plot(G.X,G.dY/max(np.abs(G.dY)),'-',c=G.color_print[0])
                else:
                    ax.plot(G.X,G.Y,'--',label=titre_fiti,markersize=1)
                    if ax2 is not None:
                        ax2.plot(G.X,G.dY/max(np.abs(G.dY)),'-')
                for i,pic in enumerate(G.pics):
                    if pic.best_fit is not None:
                        y_p=pic.best_fit[G.indexX] +bf
                    if G.color_print[1] != None:
                        titre_pic= rf" $p_{i+1}^{(G.name[0])}= {round(pic.ctr[0],3)}$" #fit.best_values[pic[1].name+'center']
                        ax.fill_between(G.X, y_p, bf, where=y_p>min(y_p), alpha=0.3, label=titre_pic,color=G.color_print[1][i])
                    else:
                        ax.fill_between(G.X, y_p, bf, where=y_p>min(y_p), alpha=0.1)
       
        ax.minorticks_on()
        ax.tick_params(which='major', length=10, width=1.5, direction='in')
        ax.tick_params(which='minor', length=5, width=1.5, direction='in')
        ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)
        ax.set_title(f'$Spectre,\Delta\lambda=$'+ str(self.lambda_error))
        ax.set_ylabel('Amplitude (U.A.)')
        ax.set_xlim([min(self.x_corr),max(self.x_corr)])
        ax.ticklabel_format(axis="y",style="sci",scilimits=(0, 0))
        if ax2 is not None:
            ax2.axhline(0,color="k",ls='-.')
            ax2.minorticks_on()
            ax2.tick_params(which='major', length=10, width=1.5, direction='in')
            ax2.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax2.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)

            ax2.set_xlabel(f'$\lambda$ (nm)')
            ax2.set_ylabel(f'$(Data-Fit)/max (U.A.)$')
            ax2.set_xlim([min(self.x_corr),max(self.x_corr)])
            ax2.ticklabel_format(axis="y",style="sci",scilimits=(0, 0))
        else:
            ax.set_xlabel(f'$\lambda$ (nm)')
        #ax.yaxis.get_offset_text().set_fontsize(10)
        ax.legend(loc="best")
        if return_fig is True:
            return fig
        else:
            if print_fig is True:
                plt.show() 
            else:
                return ax
    
    def FIT_One_Jauge(self,num_jauge=0,peakMax0=None,wnb_range=3,coef_spe=None,sigma=None,inter=None,model_fit=None,manuel=False,model_jauge=None,Delta_ctr=None):
        G=self.Gauges[num_jauge]
        y_sub=self.y_corr
        """
        for other_G in self.Gauges:
            if other_G.name != G.name and G.state =="Y":
                param=other_G.model.make_params()
                y_sub=y_sub - other_G.model.eval(param,x=self.wnb)
        """
        if (peakMax0 is not None) and (model_jauge is None):
            G=self.Gauges[num_jauge]
            peakMax=peakMax0


        elif model_jauge is not None:
            G = model_jauge[num_jauge]
            wnb_range_model=(self.wnb[G.indexX][-1]-self.wnb[G.indexX][0])/2
            if  ("Lw" and "Hg") not in G.name_spe:
                if wnb_range_model < self.lambda_error*10:
                    wnb_range=self.lambda_error*10
                elif wnb_range_model <= wnb_range:
                    wnb_range=wnb_range_model+self.lambda_error
            peakMax=G.lamb_fit
        else:
            G=self.Gauges[num_jauge]
            peakMax=G.lamb0
        
        dpic=[dp[0] for dp in G.deltaP0i]
        
        if ("Lw" and "Hg") in G.name_spe :
            match = re.search("Lw(\d+)", G.name_spe)
            Dwnb_low= float(match.group(1))
            match = re.search("Hg(\d+)", G.name_spe)#float(G.name_spe[1:3]) avec l aps Low
            Dwnb_hight= float(match.group(1)) #float(G.name_spe[4:6]) avce H pas Hight 
            G.indexX=np.where((self.wnb > (peakMax-Dwnb_low)) & (self.wnb < (peakMax+Dwnb_hight)))[0]
            wnb_range=Dwnb_hight+Dwnb_low
        else:
            G.indexX=np.where((self.wnb > peakMax-(wnb_range+abs(min(min(dpic),0)))) & (self.wnb < peakMax+(wnb_range+max(max(dpic),0))))[0]

        x_sub = np.array(self.wnb[G.indexX])
        y_sub=np.array(y_sub[G.indexX])

        if Delta_ctr is None:
            Delta_ctr=wnb_range/10
        
        if manuel == False:
            indexX=np.where((x_sub > peakMax-Delta_ctr*2) & (x_sub < peakMax+Delta_ctr*2))[0]
            x_max=x_sub[indexX]
            y_max=y_sub[indexX]
            peakMax = x_max[np.argmax(y_max)]
            ampMax = np.max(y_max)
            
        else:
            i0=np.argmin(abs(peakMax0-x_sub))
            peakMax =x_sub[i0]
            ampMax=y_sub[i0]
        
        G.Update_Fit(crt=peakMax,ampH=ampMax,coef_spe=coef_spe,sigma=sigma,inter=inter,model_fit=model_fit,Delta_ctr=Delta_ctr)


        G.fit=G.model.fit(y_sub, x=x_sub)
        G.model = G.fit.model
        G.Y= G.fit.best_fit + self.blfit[G.indexX]
        G.dY= G.fit.best_fit - y_sub
        G.X=x_sub
        G.lamb_fit =G.fit.best_values[G.name+'_p1center']
        G.bit_fit=True
        #print(G.fit.fit_report())
        if "DRX" in G.name_spe :      
            G.pic=[G.fit.best_values[G.name + '_p'+str(i+1)+'center'] for i in range(G.nb_pic) ]
        
        for p in G.pics:
            new_param=p.Out_model(out=G.fit)
            p.Update(ctr=float(new_param[0]),ampH=float(new_param[1]),coef_spe=new_param[3],sigma=float(new_param[2]))     
            param=p.model.make_params()
            p.best_fit=p.model.eval(param,x=self.wnb)

        self.Gauges[num_jauge]=G
    
    def FIT(self,wnb_range=2,coef_spe=None,sigma=None,inter=None,model_fit=None,model_jauge=None):
        for i,G in enumerate(self.Gauges):
            if G.state == "Y":
                try:
                    self.FIT_One_Jauge(num_jauge=i,peakMax0=G.lamb_fit,wnb_range=wnb_range,coef_spe=coef_spe,sigma=sigma,inter=inter,model_fit=model_fit,model_jauge=model_jauge)
                except Exception as e:
                    G.state="IN_NOISE"
                    print("error:",e,"in fit of :",G.name)
            G.bit_fit=True
        for G in self.Gauges:
            G.Calcul(input_spe=self.Gauges,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)
        self.bit_fit=True
    
    def FIT_Curv(self,inter=1):
        list_F=[]
        initial_guess=[]
        bounds_min,bounds_max=[],[]
        x_min, x_max=float(self.Gauges[0].lamb0),float(self.Gauges[0].lamb0)
        for j,G in enumerate(self.Gauges):
            for i,p in enumerate(G.pics):
                x_min,x_max=min(x_min,p.ctr[0]-p.sigma[0]*5),max(x_max,p.ctr[0]+p.sigma[0]*5)
                list_F.append(p.f_model)
                initial_guess+= [p.ctr[0],p.ampH[0],p.sigma[0]]
                for c in p.coef_spe:
                    initial_guess+=[c[0]]
                bounds_min+=[p.ctr[1][0],p.ampH[1][0],p.sigma[1][0]]
                bounds_max+=[p.ctr[1][1],p.ampH[1][1],p.sigma[1][1]]
                for c in p.coef_spe:
                    bounds_min+=[c[1][0]]
                    bounds_max+=[c[1][1]]
            G.Update_model()
        bounds=[bounds_min,bounds_max]

        for i,G in enumerate(self.Gauges):
            if i ==0:
                self.model = G.model
            else:
                self.model+=G.model
       
        self.Data_treatement()

        self.indexX=np.where((self.wnb >= x_min) & (self.wnb <= x_max))[0]
        x_corr=self.wnb[self.indexX]
        y_sub = self.y_corr[self.indexX]
        blfit = self.blfit[self.indexX]
        sum_function = Gen_sum_F(list_F)
        params , params_covar = curve_fit(sum_function,x_corr,y_sub,p0=initial_guess,bounds=bounds)
        fit=sum_function(x_corr,*params)
        self.Y= fit +  blfit
        self.X=x_corr
        self.dY= y_sub-fit 
        self.lamb_fit =params[0]
        ij_3,ij_4,ij_5=0,0,0
        params_list=list(params)

        for i, J in enumerate(self.Gauges):
            for j , p in enumerate(J.pics):
                n_c=len(p.coef_spe)
                start_idx = 3 * ij_3 + 4 * ij_4 + 5 * ij_5
                end_idx = start_idx + 3
                if n_c == 0:
                    params_pic = params_list[start_idx:end_idx] #list(params[start_idx:end_idx]) 
                    ij_3 += 1
                elif n_c == 1:
                    params_pic = params_list[start_idx:end_idx] + [np.array([params_list[end_idx]])]#list(params[start_idx:end_idx]) + list(np.array(params[end_idx]))
                    ij_4 += 1  
                elif n_c == 2:
                    params_pic = params_list[start_idx:end_idx] + [np.array(params_list[end_idx:end_idx+2])] #list(params[start_idx:end_idx]) + list(np.array(params[end_idx:end_idx+2]))
                    ij_5 += 1
                p.Update(ctr=float(params_pic[0]),ampH=float(params_pic[1]),coef_spe=params_pic[3],sigma=float(params_pic[2]),inter=float(inter))
                param=p.model.make_params()
                p.best_fit=p.model.eval(param,x=self.wnb)
            if j==0:
                J.lamb_fit=params_pic[0]
            param=J.model.make_params()
            J.Y=p.model.eval(param,x=self.wnb)
            J.X=self.wnb
            J.dY=J.Y-self.y_corr
            J.bit_fit=True

        for G in self.Gauges:
            G.Calcul(input_spe=self.Gauges,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)
        self.bit_fit=True

    def Clear_study(self,num_jauge):
        self.Gauges[num_jauge].study.loc[:, :] = None

    def Calcul_study(self,mini=False):
        #self.lambda_error=round((self.wnb[0]-self.wnb[-1])*0.5/len(self.wnb),4)
        for i in range(len(self.Gauges)):
            if ("DRX" in self.Gauges[i].name_spe) and (self.bit_fit is True): #
                self.Gauges[i].bit_fit =True
            self.Gauges[i].Calcul(input_spe=self.Gauges,mini=mini,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)  
    
    def Data_treatement(self,deg_baseline=None,type_filtre=None,param_f=None,print_data=False,ax=None,ax2=None):
        if deg_baseline is not None:
            self.deg_baseline=deg_baseline
        if param_f is not None:
            self.param_f=param_f
        self.blfit = pk.baseline(self.spec, deg=self.deg_baseline) #retrait du dark
        if self.deg_baseline ==0:
            deltaBG=min(self.spec)-self.blfit[0]
            if  deltaBG <0 :
                self.blfit =np.array(self.blfit) + deltaBG*1.05
        
        if type_filtre is not None:
            self.type_filtre=type_filtre

        if "svg" == self.type_filtre: # Appliquer un filtre de Savitzky-Golay pour lisser le spectre
            self.y_filtre = savgol_filter(self.spec,window_length=self.param_f[0],polyorder=self.param_f[1])
        elif "fft" == self.type_filtre:
            # Transformée de Fourier du signal
            spectre_fft = np.fft.fft(self.spec)
            # Fréquences associées
            frequences_fft_brut = np.fft.fftfreq(len(self.spec), d=self.wnb[1]-self.wnb[0])
            # Filtrage en supprimant les basses fréquences
            cutoff_low = self.param_f[0]  # Fréquence de coupure inférieure
            cutoff_high = self.param_f[1] # Fréquence de coupure supérieure
            # Supprimer les fréquences indésirables dans cette plage
            spectre_fft_brut=copy.deepcopy(spectre_fft)
            spectre_fft[(np.abs(frequences_fft_brut) > cutoff_low) & (np.abs(frequences_fft_brut) < cutoff_high)] = 0
            # Transformée de Fourier inverse pour ré
            self.y_filtre = np.real(np.fft.ifft(spectre_fft))
        else:
            self.y_filtre=self.spec
        self.y_corr = self.y_filtre - self.blfit

        if print_data is True:
            if ax == None:
                figure=False
                fig, (ax,ax2) =plt.subplots(ncols=1,nrows=2,figsize=(8,4),gridspec_kw={'height_ratios': [0.7, 0.3]})      
            else:
                figure=True

            if "fft" != self.type_filtre:
                # Transformée de Fourier du signal
                spectre_fft_brut = np.fft.fft(self.spec)
                spectre_fft_fit = np.fft.fft(self.y_corr)
                # Fréquences associées
                frequences_fft_brut = np.fft.fftfreq(len(self.spec), d=self.wnb[1]-self.wnb[0])
                frequences_fft_fit = np.fft.fftfreq(len(self.y_corr), d=self.x_corr[1]-self.x_corr[0])
                ax2.plot(np.abs(frequences_fft_fit), np.abs(spectre_fft_fit),'-.g', label='Data_fit')
            else:
                ax2.fill_between([self.param_f[0],self.param_f[1]],min(np.abs(spectre_fft)),max(np.abs(spectre_fft)) , color="red", alpha=0.2,label="freq filtré")
            
            ax.plot(self.x_corr,self.blfit,'-.', c='g', markersize=1,label="Bkg")
            ax.plot(self.wnb,self.spec,'-',color='gray',markersize=4,label='Brut')
            ax.plot(self.x_corr,self.y_corr+self.blfit,'-.+',color='black',markersize=3,label='Corr + bkg')
            ax.minorticks_on()
            ax.tick_params(which='major', length=10, width=1.5, direction='in')
            ax.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)
            ax.set_title(f'$Spectre \Delta\lambda=$'+ str(self.lambda_error))
            ax.set_xlabel(f'$\lambda$ (nm)')
            ax.set_ylabel('U.A.')
            ax.set_xlim([min(self.x_corr),max(self.x_corr)])
            ax.ticklabel_format(axis="y",style="sci",scilimits=(0, 0))
            ax2.plot(np.abs(frequences_fft_brut), np.abs(spectre_fft_brut),'-.k', label='Data_brut')
            
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.tick_params(which='major', length=10, width=1.5, direction='in')
            ax2.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax2.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)
            ax2.set_xlabel(f'$f$ (Hz)')
            ax2.set_ylabel('Amplitude (u.a.)')
            ax2.set_title('FFT')
            ax2.legend(loc="best")
            #ax2.set_xlim([min(np.abs(frequences_fft)),max(np.abs(frequences_fft))])
            if figure is False:
                plt.show() 
            else:
                return ax,ax2
    
    """------------------------------------- CLASSE BANC CED DYNAMIQUE -------------------------------------"""
