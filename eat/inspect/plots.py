import numpy as np
import pandas as pd
import seaborn as sns
from eat.io import hops, util
from eat.hops import util as hu
import matplotlib.pyplot as plt
from eat.inspect import utils as ut

dict_night = {3597: '04/04/17',3598: '04/05/17',3599: '04/06/17', 3600: '04/09/17',3601: '04/10/17'}
dict_night_color = {'04/04/17': 'b','04/05/17': 'r','04/06/17': 'g', '04/09/17': 'm','04/10/17': 'k'}
SMT2Z = {'ALMA': 'A', 'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R', 'SMA':'S', 'SMT':'Z', 'PV':'P','SPT':'Y'}
Z2SMT = {v: k for k, v in SMT2Z.items()}
Z2SMT['P'] = 'IRAM30'
baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']
markers=["o", "d","v","^",'s']
markers_od = ["o", "d","o","d",'o']
colors=['k','mediumblue','red','darkgreen','lime','magenta','blueviolet','orange','yellow','cyan','olivedrab','salmon','saddlebrown','dodgerblue','tomato','tan']


def plt_all_triangles_CP_CPdiff(brr,bll,sour,snr_cut=0.,saveF=False,add_text=''):
    fmtL = ['bo','ro','go','co','mo','ko','b^','r^','g^','c^','m^','k^','bv','rv','gv','cv','mv','kv']    
    brr = brr[brr['snr'] > snr_cut]; bll = bll[bll['snr'] >snr_cut]
    brr = brr[brr.source==sour]
    bll = bll[bll.source==sour]
    brr, bll = match_2_dataframes(brr, bll, 'triangle')
    if np.shape(brr)[0]>2.:
        AllTri = sorted(list(set(brr.triangle)))
        plt.figure(figsize=(10,10))
        
        for cou in range(len(AllTri)):
            brrTRI = brr[brr.triangle==AllTri[cou]]
            bllTRI = bll[bll.triangle==AllTri[cou]]
            plt.errorbar(np.mod(brrTRI.cphase,360), phase_diff(brrTRI.cphase,bllTRI.cphase)-np.mod(brrTRI.cphase,360),np.sqrt(np.asarray(brrTRI.sigmaCP)**2 +np.asarray(bllTRI.sigmaCP)**2),fmt=fmtL[cou],label = AllTri[cou],markersize='10')
            
        plt.xlabel('closure phase RR',fontsize=15)
        plt.ylabel('closure phases RR-LL difference',fontsize=15)
        plt.title(sour+', snr cut: '+str(snr_cut)+' '+add_text,fontsize=15)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((-20,380,y1,y2))
        plt.legend()
        plt.axhline(y=.0,color='k',linestyle='--')
        plt.grid()
        plt.tight_layout()
        
        if saveF==True:
            if snr_cut > 0.:
                nameF = 'CPRR_CPdiff_'+sour+'_snr_'+str(int(snr_cut))+'_'+add_text+'.pdf'
            else:
                nameF = 'CPRR_CPdiff_'+sour+'_'+add_text+'.pdf'
            plt.savefig(nameF)
        plt.show()
        
def plt_all_triangles_CP_CP(brr,bll,sour,snr_cut=0.,saveF=False,add_text=''):
    fmtL = ['bo','ro','go','co','mo','ko','b^','r^','g^','c^','m^','k^','bv','rv','gv','cv','mv','kv']    
    brr = brr[brr['snr'] > snr_cut]; bll = bll[bll['snr'] >snr_cut]
    brr = brr[brr.source==sour]
    bll = bll[bll.source==sour]
    brr, bll = match_2_dataframes(brr, bll, 'triangle')
    if np.shape(brr)[0]>2.:
        AllTri = sorted(list(set(brr.triangle)))
        plt.figure(figsize=(10,10))
        
        for cou in range(len(AllTri)):
            brrTRI = brr[brr.triangle==AllTri[cou]]
            bllTRI = bll[bll.triangle==AllTri[cou]]
            xpl = np.asarray(np.mod(brrTRI.cphase,360))
            ypl = np.asarray(np.mod(bllTRI.cphase,360))
            xpl = xpl-360.*(xpl>180.)
            ypl = ypl-360.*(ypl>180.)
            #plt.errorbar(np.mod(brrTRI.cphase,360), np.mod(bllTRI.cphase,360),np.sqrt(np.asarray(brrTRI.sigmaCP)**2 +np.asarray(bllTRI.sigmaCP)**2),fmt=fmtL[cou],label = AllTri[cou],markersize='10')
            plt.errorbar(xpl, ypl ,np.asarray(brrTRI.sigmaCP),np.asarray(bllTRI.sigmaCP),fmt=fmtL[cou],label = AllTri[cou],markersize='10')

        plt.xlabel('closure phase RR',fontsize=15)
        plt.ylabel('closure phases LL',fontsize=15)
        plt.title(sour+', snr cut: '+str(snr_cut)+' '+add_text,fontsize=15)
        #x1,x2,y1,y2 = plt.axis()
        #plt.axis((-20,380,y1,y2))
        plt.legend()
        plt.axhline(y=.0,color='k',linestyle='--')
        plt.axvline(x=.0,color='k',linestyle='--')
        plt.grid()
        plt.tight_layout()
        
        if saveF==True:
            if snr_cut > 0.:
                nameF = 'CPRR_CPdiff_'+sour+'_snr_'+str(int(snr_cut))+'_'+add_text+'.pdf'
            else:
                nameF = 'CPRR_CPdiff_'+sour+'_'+add_text+'.pdf'
            plt.savefig(nameF)
        plt.show()        

def plt_all_triangles_datetime_CPdiff(brr,bll,sour,snr_cut=0.,saveF=False, add_text=''):
    fmtL = ['bo','ro','go','co','mo','ko','b^','r^','g^','c^','m^','k^','bv','rv','gv','cv','mv','kv']    
    brr = brr[brr['snr'] > snr_cut]; bll = bll[bll['snr'] >snr_cut]
    brr = brr[brr.source==sour]
    bll = bll[bll.source==sour]
    brr, bll = match_2_dataframes(brr, bll, 'triangle')
    if np.shape(brr)[0]>2.:
        AllTri = sorted(list(set(brr.triangle)))
        plt.figure(figsize=(10,10))
        
        for cou in range(len(AllTri)):
            brrTRI = brr[brr.triangle==AllTri[cou]]
            bllTRI = bll[bll.triangle==AllTri[cou]]
            plt.errorbar(list(brrTRI.datetime), phase_diff(brrTRI.cphase,bllTRI.cphase)-np.mod(brrTRI.cphase,360),np.sqrt(np.asarray(brrTRI.sigmaCP)**2 +np.asarray(bllTRI.sigmaCP)**2),fmt=fmtL[cou],label = AllTri[cou],markersize='10')
            
        plt.xlabel('time',fontsize=15)
        plt.ylabel('closure phases RR-LL difference',fontsize=15)
        plt.title(sour+', snr cut: '+str(snr_cut)+' '+add_text,fontsize=15)
        plt.legend()
        plt.axhline(y=.0,color='k',linestyle='--')
        plt.grid()
        plt.tight_layout()
        if saveF==True:
            if snr_cut > 0.:
                nameF = 'time_CPdiff_'+sour+'_snr_'+str(int(snr_cut))+'_'+add_text+'.pdf'
            else:
                nameF = 'time_CPdiff_'+sour+'_'+add_text+'.pdf'
            plt.savefig(nameF)
        plt.show()



def nights_time_x(data,what='amp',sour='3C279',base='AL',band='lo',polar='RR',time_type='gmst'):
    
    if len(base)==2:
        foo = data[(data.source==sour)&(data.baseline==base)&(data.band==band)&(data.polarization==polar)]
    elif len(base==3):
        foo = data[(data.source==sour)&(data.triangle==base)&(data.band==band)&(data.polarization==polar)]

    if 'Night' not in data.columns:
        data['Night'] =list(map(lambda x: dict_night[x],data.expt_no))
    if time_type=='gmst':
        util.add_gmst(data)
    elif time_type=='fmjd':
        data = ut.add_mjd(data)
        data = ut.add_fmjd(data)
    
    markers = markers[:len(foo.Night.unique())]
    sns.set_style('darkgrid')
    sns.lmplot(x=time_type,y=what,data=foo,hue='Night',fit_reg=False,size = 5,aspect = 1.5,markers=markers,scatter_kws={"s": 80})
    plt.grid()
    #plt.xlabel(time_type,fontsize=13)
    #plt.ylabel(what,fontsize=13)
    plt.title(sour+', '+Z2SMT[base[0]]+'-'+Z2SMT[base[1]]+', '+band+' band',fontsize=13)
    plt.grid()
    plt.show()


def err_nights_time_amp(data,sour='3C279',base='AL',band='lo',polar='RR',errscale=1,errcolumn='sigma',time_type='gmst',y_range=[]):
    
    if 'Night' not in data.columns:
        data['Night'] =list(map(lambda x: dict_night[x],data.expt_no))
    if time_type=='gmst':
        util.add_gmst(data)
    elif time_type=='fmjd':
        data = ut.add_mjd(data)
        data = ut.add_fmjd(data)

    foo = data[(data.source==sour)&(data.baseline==base)&(data.band==band)&(data.polarization==polar)]
    Nights = sorted(list(foo.Night.unique()))
    Ni = len(Nights)
    markers=["bo", "rd","go","md","co","cd"]
    markers = markers[:Ni]
    
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6))
    for cou in range(Ni):
        LocNight = Nights[cou]
        fooNi = foo[foo.Night==LocNight]
        fmtloc = markers[cou]
        plt.errorbar(fooNi.gmst,fooNi.amp,errscale*fooNi.sigma,fmt=fmtloc,capsize=5,label=LocNight)
    
    plt.grid()
    plt.xlabel(time_type+' [h]',fontsize=14)
    plt.ylabel('corr amplitude',fontsize=14)
    #plt.ylabel('correlation amplitude',fontsize=14)
    #plt.title(sour+', '+Z2SMT[base[0]]+'-'+Z2SMT[base[1]]+', '+band+' band'+', '+polar,fontsize=13)
    plt.title(Z2SMT[base[0]]+'-'+Z2SMT[base[1]]+', '+band+' band'+', '+polar,fontsize=13)
    plt.grid()
    plt.legend()
    if y_range!=[]:
        [x1,x2,y1,y2]=plt.axis()
        plt.axis([x1,x2]+y_range)
    plt.show()

def err_nights_time_cphase(data,sour='3C279',triangle='ALX',band='lo',polar=['RR'],shift=0,errscale=1,
savefig=False,time_type='gmst',phase_type='cphase_fix_amp',error_type='sigmaCP',snr_treshold=1,conj=False,
ms=7,line=True,show_both_pol=False,y_range=[],custom_title='',tshift=0,timerange=''):
    fonts=16
    if 'Night' not in data.columns:
        data['Night'] =list(map(lambda x: dict_night[x],data.expt_no))
    if time_type=='gmst':
        util.add_gmst(data)
    elif time_type=='fmjd':
        data = ut.add_mjd(data)
        data = ut.add_fmjd(data)

    if line==True:   
        markers=["bo-", "rd-","go-","md-","co-","cd-"]
    else:
        markers=["bo", "rd","go","md","co","cd"]

    if sour=='any':
        fooG = data[(data.triangle==triangle)&(data.snr>snr_treshold)]
    else:
        fooG = data[(data.source==sour)&(data.triangle==triangle)&(data.snr>snr_treshold)]

    Nrow=len(polar)
    if Nrow>1:
        fig, ax = plt.subplots(Nrow,figsize=(10,1+5*Nrow))
        for couP in range(Nrow):
            foo = fooG[(fooG.band==band)&(fooG.polarization==polar[couP])]
            Nights = sorted(list(foo.Night.unique()))
            Ni = len(Nights)
            for cou in range(Ni):
                LocNight = Nights[cou]
                fooNi = foo[(foo.Night==LocNight)]
                fmtloc = markers[cou]
                cphaseLoc = np.asarray(fooNi[phase_type])
                if conj==True:
                    cphaseLoc= -cphaseLoc
                cphaseLoc = np.mod(cphaseLoc + shift,360) - shift
                ax[couP].errorbar(fooNi.gmst,cphaseLoc,errscale*fooNi[error_type],fmt=fmtloc,capsize=5,label=LocNight,markersize=ms)
            plt.grid()
            [x1,x2,y1,y2]=ax[couP].axis()
            if y_range==[]:
                ax[couP].axis([x1,x2,y1,y2])
            else:
                ax[couP].axis([x1,x2]+y_range)
            ax[couP].axhline(0,linestyle='--',color= (0.25, 0.25, 0.25))
            ax[couP].set_xlabel('GMST [h]',fontsize=14)
            ax[couP].set_ylabel('closure phase [deg]',fontsize=14)
            ax[couP].set_title(sour+', '+Z2SMT[triangle[0]]+'-'+Z2SMT[triangle[1]]+'-'+Z2SMT[triangle[2]]+', '+band+' band'+', '+polar[couP],fontsize=13)
    
            ax[couP].legend()
            plt.tight_layout()
            if savefig==True:
                tit= sour+'_'+Z2SMT[triangle[0]]+'_'+Z2SMT[triangle[1]]+'_'+Z2SMT[triangle[2]]+'_'+band+'_'+polar[couP]
                plt.savefig(tit+'.pdf')
        plt.show()
    else:
        fig, ax = plt.subplots(Nrow,figsize=(10,1+5*Nrow))
        couP=0
        foo = fooG[(fooG.band==band)&(fooG.polarization==polar[couP])]
        if show_both_pol==True:
            if polar[0]=='RR': antyPol='LL'
            else: antyPol='RR'
            foo2 = fooG[(fooG.band==band)&(fooG.polarization==antyPol)]

        Nights = sorted(list(foo.Night.unique()))
        Ni = len(Nights)
        for cou in range(Ni):
            LocNight = Nights[cou]
            fooNi = foo[(foo.Night==LocNight)]
            fmtloc = markers[cou]
            #cphaseLoc = np.asarray(fooNi[phase_type])[:,1]
            cphaseLoc = np.asarray(fooNi[phase_type])
            #print([np.shape(cphaseLoc),np.shape(fooNi.gmst),np.shape(fooNi.sigmaCP)])
            if conj==True:
                cphaseLoc= -cphaseLoc
            cphaseLoc = np.mod(cphaseLoc + shift,360) - shift

            gtime = np.mod(np.asarray(fooNi.gmst)+tshift,24)-tshift
            ax.errorbar(gtime,cphaseLoc,errscale*fooNi[error_type],fmt=fmtloc,capsize=5,label=LocNight,markersize=ms)
            
            if timerange!='':
                xtime = timerange
            if show_both_pol==True:
                fooNi2 = foo2[(foo2.Night==LocNight)&(foo2.snr>snr_treshold)]
                cphaseLoc = np.asarray(fooNi2.cphase)
                if conj==True:
                    cphaseLoc= -cphaseLoc
                cphaseLoc = np.mod(cphaseLoc + shift,360) - shift
                ax.errorbar(fooNi2.gmst,cphaseLoc,errscale*fooNi2.sigmaCP,fmt=fmtloc,capsize=5,label=LocNight+' '+antyPol,markersize=ms,mfc='None')
            try:
                ax.legend()
            except IndexError:
                pass
        plt.grid()
        [x1,x2,y1,y2]=ax.axis()
        if y_range==[]:
            ax.axis([x1,x2,y1,y2])
        else:
            ax.axis([x1,x2]+y_range)
        [x1,x2,y1,y2]=ax.axis()
        if timerange=='':
            pass
        else:
            ax.axis(timerange+[y1,y2])
        ax.axhline(0,linestyle='--',color= (0.25, 0.25, 0.25))
        ax.set_xlabel('GMST [h]',fontsize=fonts)
        ax.set_ylabel('closure phase [deg]',fontsize=fonts)
        #ax.set_title(sour+', '+Z2SMT[triangle[0]]+'-'+Z2SMT[triangle[1]]+'-'+Z2SMT[triangle[2]]+', '+band+' band'+', '+polar[couP],fontsize=13)
        ax.set_title(sour+', '+Z2SMT[triangle[0]]+'-'+Z2SMT[triangle[1]]+'-'+Z2SMT[triangle[2]]+', '+band+' band'+', '+polar[couP],fontsize=fonts)
        plt.tick_params(axis='both', labelsize=fonts-1)
        plt.grid()
        try:     
            #ax.legend(fontsize=fonts-1,bbox_to_anchor=(1.0, 1.0))
            ax.legend(fontsize=fonts-1)


        except IndexError:
            pass
        plt.tight_layout()
        if savefig==True:
            tit= sour+'_'+Z2SMT[triangle[0]]+'_'+Z2SMT[triangle[1]]+'_'+Z2SMT[triangle[2]]+'_'+band+'_'+polar[couP]+custom_title
            plt.savefig(tit+'.pdf')
        plt.show()


baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']  
#Fourth TYPE OF PLOT: u-v coverage by baseline
def plot_uv_coverage(sour = '3C279',baseL = baseL,bandL=['lo','hi'],polarL=['LL','RR'],data=[],savefig=False,custom_title='',snrCut=2.,redundant=False,sc=1e6):
    
    data=data[data.snr>snrCut]
    data2=data[data.snr>snrCut]
    sns.set_context("talk")
    #sns.set_style('darkgrid')
    t= np.linspace(0,2*np.pi,256)
    uas = np.pi/180/60/60/1e6
    w0 = 1/(50*uas)
    w1 = 1/(25*uas)
    
    data2['u'] = -data['u']
    data2['v'] = -data['v']
    dataF = pd.concat([data,data2],ignore_index=True)
    dataF['u'] = 1e6*dataF['u']
    dataF['v'] = 1e6*dataF['v']
    dataF = dataF[dataF.snr>snrCut].copy()
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in polarL,dataF.polarization))]
    #fooRR['ratio_RL2RR'] = np.asarray(fooRL.amp)/np.asarray(fooRR.amp)
    #sns.lmplot(x='mu',y='mv',data=fooRR,hue='expt_no',fit_reg=False,size = 5,aspect = 1)
    Nb = len(set(fooRR.baseline.unique())&set(baseL))
    if redundant==False:
        baseL = ['AX','AZ','AP','AS','AL','AY','ZL','LP','LS','ZP','ZS','PS','LY','PY','SY','ZY']
    else:
        baseL = list(fooRR.baseline.unique())
        baseL = [x for x in baseL if (('X' in x)|('J' in  x))&(x!='AX')]
    #baseL = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','YZ']  
    colors=['k','mediumblue','red','darkgreen','lime','magenta','blueviolet','orange','yellow','cyan','olivedrab','salmon','saddlebrown','dodgerblue','tomato','tan']
    current_palette=dict(zip(baseL,colors))
    #current_palette={'AX':'k','AZ':'b'}
    #current_palette = sns.color_palette("Dark2", Nb)
    #sns.set_palette(current_palette)
    markers=["o"]*8+["o"]*8+["v"]*8
    markers = markers[:Nb]
    sns.lmplot(x='u',y='v',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=1.25,scatter_kws={"s": 80},markers=markers,palette=current_palette)
    sns.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    #sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
    plt.axvline(0,linestyle='--',color= (0.5, 0.5, 0.5))
    plt.axhline(0,linestyle='--',color= (0.5, 0.5, 0.5))
    plt.grid()
    plt.plot(w0*np.sin(t),w0*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    plt.plot(w1*np.sin(t),w1*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    plt.xlabel('u [$\lambda$]',fontsize=13)
    plt.ylabel('v [$\lambda$]',fontsize=13)
    #plt.title(sour,fontsize=13)
    plt.title(sour+' '+custom_title,fontsize=13)
    r1 = 7.4e9; a1 = np.pi*0.3
    r2=3.3e9; a2=np.pi*0.35
    plt.text(r1*np.cos(a1),r1*np.sin(a1), '25 $\mu$as', fontsize=12,rotation=-42)
    plt.text(r2*np.cos(a2),r2*np.sin(a2), '50 $\mu$as', fontsize=12,rotation=-42)
    plt.grid()
    #plt.show()
    
    #sc = 1e6
    plt.axis([-9000*sc,9000*sc,-9000*sc,9000*sc])
    
    shY = 2000*sc
    plt.text(-18000*sc, 0+shY, 'A = ALMA/APEX', fontsize=13)
    plt.text(-18000*sc, -800*sc+shY, 'L = LMT', fontsize=13)
    plt.text(-18000*sc, -1600*sc+shY, 'Z = SMT', fontsize=13)
    plt.text(-18000*sc, -2400*sc+shY, 'S = SMA/JCMT', fontsize=13)
    plt.text(-18000*sc, -3200*sc+shY, 'Y = SPT', fontsize=13)
    plt.text(-18000*sc, -4000*sc+shY, 'P = PICOVEL', fontsize=13)
    plt.text(-18000*sc, -4800*sc+shY, 'X = APEX', fontsize=13)
    
    r1 = 7400*sc; a1 = np.pi*0.3
    r2=3300*sc; a2=np.pi*0.35
    plt.text(r1*np.cos(a1),r1*np.sin(a1), '25 $\mu$as', fontsize=12,rotation=-42)
    plt.text(r2*np.cos(a2),r2*np.sin(a2), '50 $\mu$as', fontsize=12,rotation=-42)
    #plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.3,top=0.95)
    if savefig==True:
        tit = 'uv_cover_'+sour
        plt.savefig(tit+'.pdf')
    plt.show()
    
baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']
def plot_uv_snr(sour = '3C279',exptL=[3597,3598,3599,3600,3601],baseL = baseL,bandL=['lo','hi'],data=[],polar='both',savefig=False,snrCut=2.,snrMax=1010.,snrMin=0):
    data2 = data[data.snr>snrCut].copy()
    
    data2['u'] = -data['u']
    data2['v'] = -data['v']
    
    dataF = pd.concat([data,data2],ignore_index=True)
    dataF = dataF[dataF.snr>snrCut]
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='RR')]
    fooLL = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='LL')]
    cminRR = np.min(fooRR.snr)
    cmaxRR = np.max(fooRR.snr)
    cminLL = np.min(fooLL.snr)
    cmaxLL = np.max(fooLL.snr)
    cmin_both = 0.5*(cminRR+cminLL)
    cmax_both = 0.5*(cmaxRR+cmaxLL)
    cmax = np.log10(snrMax)
    if polar=='both':
        cmin = np.maximum(np.log10(cmin_both),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooLL.snr)+np.asarray(fooRR.snr))/2)))
    elif polar=='LL':
        cmin=np.maximum(np.log10(cminLL),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooLL.snr)+np.asarray(fooLL.snr))/2)))
    elif polar=='RR':
        cmin=np.maximum(np.log10(cminRR),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooRR.snr)+np.asarray(fooRR.snr))/2)))

    plt.figure(figsize=(10,10))
    if polar=='RR':
        plt.scatter(fooRR.u,fooRR.v,c=colors,cmap='jet',s=80,alpha=0.75)
    else:
        plt.scatter(fooLL.u,fooLL.v,c=colors,cmap='jet',s=80,alpha=0.75)
    #sns.lmplot(x='u',y='v',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=1.25)
    plt.axvline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    plt.axhline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    plt.grid()
    plt.xlabel('u [M$\lambda$]',fontsize=16)
    plt.ylabel('v [M$\lambda$]',fontsize=16)
    plt.title(sour,fontsize=16)
    plt.grid()
    cb = plt.colorbar(fraction=0.035, pad=0.04,label='$\log_{10}$(snr)')
    plt.axes().set_aspect('equal')
    plt.axis([-10000,10000,-10000,10000])
    cb.set_label(label='SNR',size=16)
    #print(cmin)

    cb.set_clim(cmin,cmax)
    ticks_all = np.asarray([0.699,1,1.301,1.69897,2,2.301,2.69897,3,3.3005,3.69897])
    desc_all = np.asarray([5,10,20,50,100,200,500,1000,2000,5000])
    ticks = ticks_all[(ticks_all>cmin)&(ticks_all<cmax)]
    desc = desc_all[(ticks_all>cmin)&(ticks_all<cmax)]
    #print(ticks_all)
    #print(desc_all)
    cb.set_ticks(ticks, update_ticks=True)
    cb.set_ticklabels(desc, update_ticks=True)
    #sns.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    #sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
    #cb.set_ticks([1.69897,2,2.301,2.69897,3,3.301], update_ticks=True)
    #cb.set_ticklabels([50,100,200,500,1000,2000], update_ticks=True)
    

    t= np.linspace(0,2*np.pi,256)
    uas = np.pi/180/60/60/1e6
    w0 = 1/(50*uas)/1e6
    w1 = 1/(25*uas)/1e6

    plt.axis([-9000,9000,-9000,9000])
    
   
    plt.plot(w0*np.sin(t),w0*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    plt.plot(w1*np.sin(t),w1*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    
    r1 = 7400; a1 = np.pi*0.3
    r2=3300; a2=np.pi*0.35
    plt.text(r1*np.cos(a1),r1*np.sin(a1), '25 $\mu$as', fontsize=14,rotation=-42)
    plt.text(r2*np.cos(a2),r2*np.sin(a2), '50 $\mu$as', fontsize=14,rotation=-42)
      
    if savefig==True:
        tit = 'uv_snr_'+sour
        plt.savefig(tit+'.pdf')
    #sns.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    #sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})

    plt.show()

def plot_uv_amp(sour = '3C279',exptL=[3597,3598,3599,3600,3601],baseL = baseL,bandL=['lo','hi'],data=[],polar='both',savefig=False,snrCut=2.,snrMax=1010.,snrMin=0):
    data2 = data[data.snr>snrCut].copy()
    
    data2['u'] = -data['u']
    data2['v'] = -data['v']
    
    dataF = pd.concat([data,data2],ignore_index=True)
    dataF = dataF[dataF.snr>snrCut]
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='RR')]
    fooLL = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='LL')]
    cminRR = np.min(fooRR.amp)
    cmaxRR = np.max(fooRR.amp)
    cminLL = np.min(fooLL.amp)
    cmaxLL = np.max(fooLL.amp)
    cmin_both = 0.5*(cminRR+cminLL)
    cmax_both = 0.5*(cmaxRR+cmaxLL)
    cmax = np.log10(snrMax)
    if polar=='both':
        cmin = np.maximum(np.log10(cmin_both),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooLL.amp)+np.asarray(fooRR.amp))/2)))
    elif polar=='LL':
        cmin=np.maximum(np.log10(cminLL),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooLL.amp)+np.asarray(fooLL.amp))/2)))
    elif polar=='RR':
        cmin=np.maximum(np.log10(cminRR),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooRR.amp)+np.asarray(fooRR.amp))/2)))

    plt.figure(figsize=(10,10))
    if polar=='RR':
        plt.scatter(fooRR.u,fooRR.v,c=colors,cmap='jet',s=80,alpha=0.75)
    else:
        plt.scatter(fooLL.u,fooLL.v,c=colors,cmap='jet',s=80,alpha=0.75)
    #sns.lmplot(x='u',y='v',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=1.25)
    plt.axvline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    plt.axhline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    plt.grid()
    plt.xlabel('u [M$\lambda$]',fontsize=16)
    plt.ylabel('v [M$\lambda$]',fontsize=16)
    plt.title(sour,fontsize=16)
    plt.grid()
    cb = plt.colorbar(fraction=0.035, pad=0.04,label='$\log_{10}$(snr)')
    plt.axes().set_aspect('equal')
    plt.axis([-9000,9000,-9000,9000])
    cb.set_label(label='SNR',size=16)
    #print(cmin)

    cb.set_clim(cmin,cmax)
    ticks_all = np.asarray([0.699,1,1.301,1.69897,2,2.301,2.69897,3,3.3005,3.69897])
    desc_all = np.asarray([5,10,20,50,100,200,500,1000,2000,5000])
    ticks = ticks_all[(ticks_all>cmin)&(ticks_all<cmax)]
    desc = desc_all[(ticks_all>cmin)&(ticks_all<cmax)]
    #print(ticks_all)
    #print(desc_all)
    cb.set_ticks(ticks, update_ticks=True)
    cb.set_ticklabels(desc, update_ticks=True)
    #cb.set_ticks([1.69897,2,2.301,2.69897,3,3.301], update_ticks=True)
    #cb.set_ticklabels([50,100,200,500,1000,2000], update_ticks=True)
    


    t= np.linspace(0,2*np.pi,256)
    uas = np.pi/180/60/60/1e6
    w0 = 1/(50*uas)/1e6
    w1 = 1/(25*uas)/1e6

    plt.axis([-9000,9000,-9000,9000])
    
   
    plt.plot(w0*np.sin(t),w0*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    plt.plot(w1*np.sin(t),w1*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    
    r1 = 7400; a1 = np.pi*0.3
    r2=3300; a2=np.pi*0.35
    plt.text(r1*np.cos(a1),r1*np.sin(a1), '25 $\mu$as', fontsize=14,rotation=-42)
    plt.text(r2*np.cos(a2),r2*np.sin(a2), '50 $\mu$as', fontsize=14,rotation=-42)
      
    if savefig==True:
        tit = 'uv_snr_'+sour
        plt.savefig(tit+'.pdf')
    plt.show()


baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']
def plot_uv_snr_ALMA(sour = '3C279',exptL=[3597,3598,3599,3600,3601],baseL = baseL,bandL=['lo','hi'],data=[],polar='both',savefig=False,snrCut=2.,snrMax=1010.,snrMin=0):
   
    data2 = data[data.snr>snrCut].copy()
    
    data2['u'] = -data['u']
    data2['v'] = -data['v']
    
    dataF = pd.concat([data,data2],ignore_index=True)
    dataF = dataF[dataF.snr>snrCut]
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='RR')]
    fooLL = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='LL')]
    cminRR = np.min(fooRR.snr)
    cmaxRR = np.max(fooRR.snr)
    cminLL = np.min(fooLL.snr)
    cmaxLL = np.max(fooLL.snr)
    cmin_both = 0.5*(cminRR+cminLL)
    cmax_both = 0.5*(cmaxRR+cmaxLL)
    cmax = np.log10(snrMax)
    if polar=='both':
        cmin = np.maximum(np.log10(cmin_both),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooLL.snr)+np.asarray(fooRR.snr))/2)))
    elif polar=='LL':
        cmin=np.maximum(np.log10(cminLL),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooLL.snr)+np.asarray(fooLL.snr))/2)))
    elif polar=='RR':
        cmin=np.maximum(np.log10(cminRR),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooRR.snr)+np.asarray(fooRR.snr))/2)))



    fooRRA = fooRR[fooRR.baseline.str.contains('A')]
    fooLLA = fooLL[fooLL.baseline.str.contains('A')]

    plt.figure(figsize=(10,10))
    if polar=='RR':
        plt.scatter(fooRR.u,fooRR.v,c=colors,cmap='jet',s=80,alpha=0.75)
        #plt.scatter(fooRRA.u,fooRRA.v,c=colors,cmap='jet',s=160,alpha=0.75)
    else:
        plt.scatter(fooLLA.u,fooLLA.v,edgecolors='k', linewidths=3., facecolors='none',s=120,alpha=1.)
    
        plt.scatter(fooLL.u,fooLL.v,c=colors,cmap='jet',s=80,alpha=0.75)
        #plt.scatter(fooLLA.u,fooLLA.v,c=colors,cmap='jet',s=160,alpha=0.75)
        

    #sns.lmplot(x='u',y='v',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=1.25)
    plt.axvline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    plt.axhline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    plt.grid()
    plt.xlabel('u [M$\lambda$]',fontsize=16)
    plt.ylabel('v [M$\lambda$]',fontsize=16)
    plt.title(sour,fontsize=16)
    plt.grid()
    cb = plt.colorbar(fraction=0.035, pad=0.04,label='$\log_{10}$(snr)')
    plt.axes().set_aspect('equal')
    plt.axis([-9000,9000,-9000,9000])
    cb.set_label(label='SNR',size=16)
    #print(cmin)

    cb.set_clim(cmin,cmax)
    ticks_all = np.asarray([0,0.301,0.699,1,1.301,1.69897,2,2.301,2.69897,3,3.3005,3.69897])
    desc_all = np.asarray([1,2,5,10,20,50,100,200,500,1000,2000,5000])
    ticks = ticks_all[(ticks_all>cmin)&(ticks_all<cmax)]
    desc = desc_all[(ticks_all>cmin)&(ticks_all<cmax)]
    #print(ticks_all)
    #print(desc_all)
    cb.set_ticks(ticks, update_ticks=True)
    cb.set_ticklabels(desc, update_ticks=True)
    #cb.set_ticks([1.69897,2,2.301,2.69897,3,3.301], update_ticks=True)
    #cb.set_ticklabels([50,100,200,500,1000,2000], update_ticks=True)
    


    t= np.linspace(0,2*np.pi,256)
    uas = np.pi/180/60/60/1e6
    w0 = 1/(50*uas)/1e6
    w1 = 1/(25*uas)/1e6

    plt.axis([-9000,9000,-9000,9000])
    
   
    plt.plot(w0*np.sin(t),w0*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    plt.plot(w1*np.sin(t),w1*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    
    r1 = 7400; a1 = np.pi*0.3
    r2=3300; a2=np.pi*0.35
    plt.text(r1*np.cos(a1),r1*np.sin(a1), '25 $\mu$as', fontsize=14,rotation=-42)
    plt.text(r2*np.cos(a2),r2*np.sin(a2), '50 $\mu$as', fontsize=14,rotation=-42)
      
    ## ADD ALMA
    
    #plt.scatter(fooLLA.u,fooLLA.v,c=colors,cmap='jet',s=160,alpha=0.75)
        
    #plt.scatter(fooLLA.u,fooLLA.v,edgecolors='k', linewidths=1., facecolors='none',s=100,alpha=1.)
    #plt.scatter(fooLLA.u,fooLLA.v,edgecolors='k', facecolors='k',s=10,alpha=0.75)

    if savefig==True:
        tit = 'uv_snr_AA_'+sour
        plt.savefig(tit+'.pdf')
    plt.show()



        

#FIRST TYPE OF PLOT: GMST time vs fractional polarization
def errplot_fracpol_gmst(sour = '3C279',base = 'AS',band='lo',polar='RR',data=[],errscale=1,yrange=[0,1],savefig=False,tshift=0):
    markers=["bo", "rd","go","md","co","cd"]
    if 'Night' not in data.columns:
        data['Night'] =(list(map(lambda x: dict_night[x],data.expt_no)))
    data=data.groupby(['scan_id','band','baseline']).filter(lambda x: len(x) == 4)
    fooRL = data[(data.source==sour)&(data.baseline==base)&(data.band==band)&(data.polarization=='RL')]
    fooLR = data[(data.source==sour)&(data.baseline==base)&(data.band==band)&(data.polarization=='LR')]
    fooRR = data[(data.source==sour)&(data.baseline==base)&(data.band==band)&(data.polarization=='RR')]
    fooLL = data[(data.source==sour)&(data.baseline==base)&(data.band==band)&(data.polarization=='LL')]
    #print('sigma' in data.columns)
    #fooRR['ratio_RL2RR'] = np.asarray(fooRL.amp)/np.asarray(fooRR.amp)
    fooRR['fracpol'] = np.sqrt(np.asarray(fooLR.amp)*np.asarray(fooRL.amp)/np.asarray(fooRR.amp)/np.asarray(fooLL.amp))
    fooRR['ErrFracPol'] = 0.5*np.asarray(fooRL.amp)*np.asarray(fooLR.sigma)/np.asarray(fooRR.amp)/np.asarray(fooLL.amp)
    + 0.5*np.asarray(fooRL.sigma)*np.asarray(fooLR.amp)/np.asarray(fooRR.amp)/np.asarray(fooLL.amp)
    + 0.5*np.asarray(fooRL.amp)*np.asarray(fooLR.amp)/np.asarray(fooRR.amp)/(np.asarray(fooLL.amp)**2)*np.asarray(fooLL.sigma)
    + 0.5*np.asarray(fooRL.amp)*np.asarray(fooLR.amp)/(np.asarray(fooRR.amp)**2)/np.asarray(fooLL.amp)*np.asarray(fooRR.sigma)
    #sns.lmplot(x='fmjd',y='ratio_RL2RR',data=fooRR,hue='expt_no',fit_reg=False,size = 5,aspect = 1.5)
    
    fonts=15
    Nights = sorted(list(fooRR.Night.unique()))
    Ni = len(Nights)
    markers = markers[:len(fooRR.Night.unique())]
    plt.figure(figsize=(10,6))
    for cou in range(Ni):
        LocNight = Nights[cou]
        fooNi = fooRR[(fooRR.Night==LocNight)&(fooRR.fracpol<0.95)]
        fmtloc = markers[cou]
        gtime = np.mod(np.asarray(fooNi.gmst)+tshift,24)-tshift
        plt.errorbar(gtime,fooNi.fracpol,errscale*fooNi.ErrFracPol,fmt=fmtloc,capsize=5,label=LocNight,markersize=8)

    plt.grid()
    plt.xlabel('GMST [h]',fontsize=fonts)
    plt.ylabel('fractional polarization',fontsize=fonts)
    plt.title(sour+', '+Z2SMT[base[0]]+'-'+Z2SMT[base[1]]+', '+band+' band',fontsize=fonts)
    plt.grid()
    [x1,x2,y1,y2] = plt.axis()
    plt.axis([x1,x2]+yrange)
    plt.legend()
    if savefig==True:
        tit = sour+'_'+Z2SMT[base[0]]+'_'+Z2SMT[base[1]]+'_'+band
        plt.savefig(tit+'.pdf')
    plt.show()








#baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']  
#Fourth TYPE OF PLOT: u-v coverage by baseline
def plot_coverage(sour = '3C279',baseL = baseL,bandL=['lo','hi'],polarL=['LL','RR'],data=[],savefig=False,custom_title='',y_range='',red=False,onlyred=False,loglog=False,alphabet_bsl=False,
alpha=0.25):
    #sns.set_context("talk")
    if 'baselength' not in data.columns:
        data = ut.add_baselength(data)
    #baseL0 = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','ZY']
    redundant = ['JS','XL','XZ','XP','XS','JL','ZJ','AJ','XJ','JP','XY','JY']
    baseL0 = ['AX','AZ','AP','AS','AL','AY','ZL','LP','LS','ZP','ZS','PS','LY','PY','SY','ZY']
    if alphabet_bsl==True:
        redundant=[''.join(sorted(x)) for x in redundant]
        baseL0=[''.join(sorted(x)) for x in baseL0]
    #print(red==True)
    if red==True:
        baseL0=baseL0+redundant
        if onlyred==True:
            baseL0=redundant

        #print('baseL0 ', baseL0)
    dataF = data
    #print(dataF.baseline.unique())
    #print('Size to plot0:',np.shape(dataF))
    #fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in polarL,dataF.polarization))&list(map(lambda x: x in bandL,dataF.band))]
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL0,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in polarL,dataF.polarization))]
    
    #print('Size to plot:',np.shape(fooRR))
    #print(fooRR.baseline.unique())
    #print(fooRR)
    #fooRR['ratio_RL2RR'] = np.asarray(fooRL.amp)/np.asarray(fooRR.amp)
    #sns.lmplot(x='mu',y='mv',data=fooRR,hue='expt_no',fit_reg=False,size = 5,aspect = 1)
    #Nb = len(set(fooRR.baseline.unique())&set(baseL))
    #baseL0 = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','ZY']
    Nb = len(set(fooRR.baseline.unique())&set(baseL0))
    #print(set(fooRR.baseline.unique()))
    #print(set(fooRR.baseline.unique())&set(baseL0))
    baseL = ['AX','AZ','AP','AS','AL','AY','ZL','LP','LS','ZP','ZS','PS','LY','PY','SY','ZY']
    if alphabet_bsl==True:
        baseL=[''.join(sorted(x)) for x in baseL]
    #baseL = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','ZY']
    colors=['k','mediumblue','red','darkgreen','lime','magenta','blueviolet','orange','yellow','cyan','olivedrab','salmon','saddlebrown','dodgerblue','tomato','tan']

    #print(baseL0)

    if red==True:
        baseL=baseL+redundant
        colors=colors+['darkgray']*len(redundant)
        if onlyred==True:
            baseL=redundant

    current_palette=dict(zip(baseL,colors))
    #current_palette={'AX':'k','AZ':'b'}
    #current_palette = sns.color_palette("Dark2", Nb)
    #sns.set_palette(current_palette)
    markers=["o"]*8+["o"]*8+["v"]*8
    markers = markers[:Nb]

    #
    g = sns.lmplot(x='baselength',y='amp',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=2,scatter_kws={"s": 30,'alpha':alpha},markers=markers,palette=current_palette)
    plt.grid()
    #sns.lmplot(x='baselength',y='amp',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=2,scatter_kws={"s": 30,'alpha':0.25},markers=markers,palette=current_palette)
    plt.axvline(0,linestyle='--',color= (0.5, 0.5, 0.5))
    plt.axhline(0,linestyle='--',color= (0.5, 0.5, 0.5))
    plt.grid()
    plt.xlabel('uv distance [$\lambda$]',fontsize=13)
    plt.ylabel('flux [Jy]',fontsize=13)
    #plt.title(sour,fontsize=13)
    plt.title(sour+' '+custom_title,fontsize=13)
    for lh in g._legend.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [50] 
    
    
    #plt.axis([0,9000,0,100])
    if y_range=='':
        plt.axis([-1.e9,9e9,-0.02,8.5])
    else:
        plt.axis([-1e8,9e9]+y_range)
    plt.gcf().subplots_adjust(left=0.3,top=0.95)
    if savefig==True:
        tit = 'uvdist_amp_'+sour
        plt.savefig(tit+'.pdf')
    plt.grid()
    plt.grid()
    if loglog==True:
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.axis([4.e8,1e10,1e-3,1e1])
        plt.grid()
    plt.grid()
    plt.grid()
    plt.show()




baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']
def plot_uv_ampcal_ALMA(sour = '3C279',exptL=[3597,3598,3599,3600,3601],baseL = baseL,bandL=['lo','hi'],data=[],polar='both',savefig=False,ampCut=0.,snrMax=1010.,snrMin=0):
   
    data2 = data[data.snr>snrCut].copy()
    
    data2['u'] = -data['u']
    data2['v'] = -data['v']
    
    dataF = pd.concat([data,data2],ignore_index=True)
    dataF = dataF[dataF.snr>snrCut]
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='RR')]
    fooLL = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='LL')]
    
    cminRR = np.min(fooRR.amp)
    cmaxRR = np.max(fooRR.amp)
    cminLL = np.min(fooLL.amp)
    cmaxLL = np.max(fooLL.amp)
    cmin_both = 0.5*(cminRR+cminLL)
    cmax_both = 0.5*(cmaxRR+cmaxLL)
    cmax = np.log10(snrMax)
    if polar=='both':
        cmin = np.maximum(np.log10(cmin_both),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooLL.amp)+np.asarray(fooRR.amp))/2)))
    elif polar=='LL':
        cmin=np.maximum(np.log10(cminLL),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooLL.amp)+np.asarray(fooLL.amp))/2)))
    elif polar=='RR':
        cmin=np.maximum(np.log10(cminRR),np.log10(snrCut))
        if snrMin > 0 : cmin = np.log10(snrMin)
        colors= np.maximum(cmin,np.minimum(cmax,np.log10((np.asarray(fooRR.amp)+np.asarray(fooRR.amp))/2)))



    fooRRA = fooRR[fooRR.baseline.str.contains('A')]
    fooLLA = fooLL[fooLL.baseline.str.contains('A')]

    plt.figure(figsize=(10,10))
    if polar=='RR':
        plt.scatter(fooRR.u,fooRR.v,c=colors,cmap='jet',s=80,alpha=0.75)
        #plt.scatter(fooRRA.u,fooRRA.v,c=colors,cmap='jet',s=160,alpha=0.75)
    else:
        plt.scatter(fooLLA.u,fooLLA.v,edgecolors='k', linewidths=3., facecolors='none',s=120,alpha=1.)
    
        plt.scatter(fooLL.u,fooLL.v,c=colors,cmap='jet',s=80,alpha=0.75)
        #plt.scatter(fooLLA.u,fooLLA.v,c=colors,cmap='jet',s=160,alpha=0.75)
        

    #sns.lmplot(x='u',y='v',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=1.25)
    plt.axvline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    plt.axhline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    plt.grid()
    plt.xlabel('u [M$\lambda$]',fontsize=16)
    plt.ylabel('v [M$\lambda$]',fontsize=16)
    plt.title(sour,fontsize=16)
    plt.grid()
    cb = plt.colorbar(fraction=0.035, pad=0.04,label='$\log_{10}$(snr)')
    plt.axes().set_aspect('equal')
    plt.axis([-9000,9000,-9000,9000])
    cb.set_label(label='SNR',size=16)
    #print(cmin)

    cb.set_clim(cmin,cmax)
    ticks_all = np.asarray([0.699,1,1.301,1.69897,2,2.301,2.69897,3,3.3005,3.69897])
    desc_all = np.asarray([5,10,20,50,100,200,500,1000,2000,5000])
    ticks = ticks_all[(ticks_all>cmin)&(ticks_all<cmax)]
    desc = desc_all[(ticks_all>cmin)&(ticks_all<cmax)]
    #print(ticks_all)
    #print(desc_all)
    cb.set_ticks(ticks, update_ticks=True)
    cb.set_ticklabels(desc, update_ticks=True)
    #cb.set_ticks([1.69897,2,2.301,2.69897,3,3.301], update_ticks=True)
    #cb.set_ticklabels([50,100,200,500,1000,2000], update_ticks=True)
    


    t= np.linspace(0,2*np.pi,256)
    uas = np.pi/180/60/60/1e6
    w0 = 1/(50*uas)/1e6
    w1 = 1/(25*uas)/1e6

    plt.axis([-9000,9000,-9000,9000])
    
   
    plt.plot(w0*np.sin(t),w0*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    plt.plot(w1*np.sin(t),w1*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    
    r1 = 7400; a1 = np.pi*0.3
    r2=3300; a2=np.pi*0.35
    plt.text(r1*np.cos(a1),r1*np.sin(a1), '25 $\mu$as', fontsize=14,rotation=-42)
    plt.text(r2*np.cos(a2),r2*np.sin(a2), '50 $\mu$as', fontsize=14,rotation=-42)
    plt.show()


def err_nights_time_cphase_2x2(data,sour='3C279',triangle='ALX',shift=0,errscale=1,
savefig=False,time_type='gmst',phase_type='cphase_fix_amp',error_type='sigmaCP',snr_treshold=1,conj=False,
ms=7,line=False,show_both_pol=False,y_range=[],custom_title='',tshift=0,timerange='',line0 = False):
    fonts=16
    if 'Night' not in data.columns:
        data['Night'] =list(map(lambda x: dict_night[x],data.expt_no))
    if time_type=='gmst':
        util.add_gmst(data)
    elif time_type=='fmjd':
        data = ut.add_mjd(data)
        data = ut.add_fmjd(data)

    if line==True:   
        markers=["bo-", "rd-","go-","md-","co-","cd-"]
    else:
        markers=["bo", "rd","go","md","co","cd"]
    
    exptL = [3597,3598,3599,3600,3601]
    
    if sour=='any':
        fooG = data[(data.triangle==triangle)&(data.snr>snr_treshold)]
    else:
        fooG = data[(data.source==sour)&(data.triangle==triangle)&(data.snr>snr_treshold)]

    fig, ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(14,10))
    Nights = sorted(list(fooG.Night.unique()))
    Ni = len(Nights)

    bands=['lo','hi']
    pols=['LL','RR']
    for couB in [0,1]:
        for couP in [0,1]:
            if line0==True:
                ax[couB,couP].axhline(0,color='k',linestyle='--',label='_nolegend_')

            for cou in range(Ni):

                LocNight = Nights[cou]
                fooNi = fooG[(fooG.Night==LocNight)&(fooG.band==bands[couB])&(fooG.polarization==pols[couP])]
                fmtloc = markers[cou]
                #cphaseLoc = np.asarray(fooNi[phase_type])[:,1]
                cphaseLoc = np.asarray(fooNi[phase_type])
                #print([np.shape(cphaseLoc),np.shape(fooNi.gmst),np.shape(fooNi.sigmaCP)])
                if conj==True:
                    cphaseLoc= -cphaseLoc
                cphaseLoc = np.mod(cphaseLoc + shift,360) - shift

                gtime = np.mod(np.asarray(fooNi.gmst)+tshift,24)-tshift
                try:
                    ax[couB,couP].errorbar(gtime,cphaseLoc,errscale*fooNi[error_type],fmt=fmtloc,capsize=5,markersize=ms,mfc=dict_night_color[LocNight],ecolor=dict_night_color[LocNight],markeredgecolor=dict_night_color[LocNight],label=LocNight)
                    
                except IndexError: pass
            
            [x1,x2,y1,y2]=ax[couB,couP].axis()
            if y_range==[]:
                ax[couB,couP].axis([x1,x2,y1,y2])
            else:
                ax[couB,couP].axis([x1,x2]+y_range)

            tmax = np.max(gtime)
            tmin = np.min(gtime)
            ax[couB,couP].set_xlim([tmin-0.1,tmax+0.1])

            if timerange!='':
                ax[couB,couP].set_xlim(timerange)

                
    ax[1,1].set_xlabel('gmst time [h]')
    ax[1,0].set_xlabel('gmst time [h]')
    ax[1,0].set_ylabel('closure phase [deg]')
    ax[0,0].set_ylabel('closure phase [deg]')
    ax[0,0].set_title(sour+', '+triangle+', band: LO, polarization: LL')
    ax[0,1].set_title(sour+', '+triangle+', band: LO, polarization: RR')
    ax[1,0].set_title(sour+', '+triangle+', band: HI, polarization: LL')
    ax[1,1].set_title(sour+', '+triangle+', band: HI, polarization: RR')
    
    #ax[0,0].legend()

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], marker='o', color='b', label='04/04/17',markerfacecolor='b', markersize=8),
    Line2D([0], [0], marker='o', color='r', label='04/05/17',markerfacecolor='r', markersize=8),
    Line2D([0], [0], marker='o', color='g', label='04/06/17',markerfacecolor='g', markersize=8),
    Line2D([0], [0], marker='o', color='m', label='04/09/17',markerfacecolor='m', markersize=8),
    Line2D([0], [0], marker='o', color='k', label='04/10/17',markerfacecolor='k', markersize=8)]
    #ax[0,1].legend()
    #ax[1,0].legend()
    ax[0,0].legend(handles=legend_elements,frameon=True,framealpha=0.5)
    if savefig==True:
        tit= sour+'_'+Z2SMT[triangle[0]]+'_'+Z2SMT[triangle[1]]+'_'+Z2SMT[triangle[2]]+'_'+custom_title
        plt.savefig(tit+'.pdf')
    plt.tight_layout()
    plt.show()


def err_nights_time_lcamp_2x2(data,sour='3C279',quadrangle='ALX',errscale=1,
savefig=False,time_type='gmst',error_type='sigmaCA',snr_treshold=1,
ms=7,line=False,show_both_pol=False,y_range=[],custom_title='',tshift=0,timerange='',line0 = False):
    fonts=16
    if 'Night' not in data.columns:
        data['Night'] =list(map(lambda x: dict_night[x],data.expt_no))
    if time_type=='gmst':
        util.add_gmst(data)
    elif time_type=='fmjd':
        data = ut.add_mjd(data)
        data = ut.add_fmjd(data)
    if line==True:   
        markers=["bo-", "rd-","go-","md-","co-","cd-"]
    else:
        markers=["bo", "rd","go","md","co","cd"]
    
    exptL = [3597,3598,3599,3600,3601]
    
    if sour=='any':
        fooG = data[(data.quadrangle==quadrangle)]
    else:
        fooG = data[(data.source==sour)&(data.quadrangle==quadrangle)]

    fig, ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(14,10))
    Nights = sorted(list(fooG.Night.unique()))
    Ni = len(Nights)
    

    bands=['lo','hi']
    pols=['LL','RR']
    for couB in [0,1]:
        for couP in [0,1]:
            if line0==True:
                ax[couB,couP].axhline(0,color='k',linestyle='--',label='_nolegend_')

            for cou in range(Ni):
                

                LocNight = Nights[cou]
                fooNi = fooG[(fooG.Night==LocNight)&(fooG.band==bands[couB])&(fooG.polarization==pols[couP])]
                fmtloc = markers[cou]
                #cphaseLoc = np.asarray(fooNi[phase_type])[:,1]
                lcampLoc = np.asarray(fooNi['camp'])
                #print([np.shape(cphaseLoc),np.shape(fooNi.gmst),np.shape(fooNi.sigmaCP)])
                
                gtime = np.mod(np.asarray(fooNi.gmst)+tshift,24)-tshift
                try:
                    ax[couB,couP].errorbar(gtime,lcampLoc,errscale*fooNi['sigmaCA'],fmt=fmtloc,capsize=5,markersize=ms,mfc=dict_night_color[LocNight],ecolor=dict_night_color[LocNight],markeredgecolor=dict_night_color[LocNight],label=LocNight)

                except IndexError: pass
            
            [x1,x2,y1,y2]=ax[couB,couP].axis()
            if y_range==[]:
                ax[couB,couP].axis([x1,x2,y1,y2])
            else:
                ax[couB,couP].axis([x1,x2]+y_range)

                
    ax[1,1].set_xlabel('gmst time [h]')
    ax[1,0].set_xlabel('gmst time [h]')
    ax[1,0].set_ylabel('log closure amp')
    ax[0,0].set_ylabel('log closure phase amp')
    ax[0,0].set_title(sour+', '+quadrangle+', band: LO, polarization: LL')
    ax[0,1].set_title(sour+', '+quadrangle+', band: LO, polarization: RR')
    ax[1,0].set_title(sour+', '+quadrangle+', band: HI, polarization: LL')
    ax[1,1].set_title(sour+', '+quadrangle+', band: HI, polarization: RR')
    
    #ax[0,0].legend()

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], marker='o', color='b', label='04/04/17',markerfacecolor='b', markersize=8),
    Line2D([0], [0], marker='o', color='r', label='04/05/17',markerfacecolor='r', markersize=8),
    Line2D([0], [0], marker='o', color='g', label='04/06/17',markerfacecolor='g', markersize=8),
    Line2D([0], [0], marker='o', color='m', label='04/09/17',markerfacecolor='m', markersize=8),
    Line2D([0], [0], marker='o', color='k', label='04/10/17',markerfacecolor='k', markersize=8)]
    #ax[0,1].legend()
    #ax[1,0].legend()
    ax[0,0].legend(handles=legend_elements,frameon=True,framealpha=0.5)
    if savefig==True:
        tit= sour+'_'+Z2SMT[triangle[0]]+'_'+Z2SMT[triangle[1]]+'_'+Z2SMT[triangle[2]]+'_'+custom_title
        plt.savefig(tit+'.pdf')
    plt.tight_layout()
    plt.show()


def err_cphase_band_pol(data,expt=3601,sour='3C279',triangle='ALX',shift=0,errscale=1,
savefig=False,time_type='gmst',phase_type='cphase_fix_amp',error_type='sigmaCP',snr_treshold=1,conj=False,
ms=7,line=True,show_both_pol=False,y_range=[],custom_title='',tshift=0,timerange=''):
    fonts=16
    if 'Night' not in data.columns:
        data['Night'] =list(map(lambda x: dict_night[x],data.expt_no))
    if time_type=='gmst':
        util.add_gmst(data)
    elif time_type=='fmjd':
        data = ut.add_mjd(data)
        data = ut.add_fmjd(data)

    if line==True:   
        markers=["bo-", "rd-","go-","md-","co-","cd-"]
    else:
        markers=["bo", "rd","go","md","co","cd"]

    if sour=='any':
        fooG = data[(data.expt_no==expt)&(data.triangle==triangle)&(data.snr>snr_treshold)]
    else:
        fooG = data[(data.expt_no==expt)&(data.source==sour)&(data.triangle==triangle)&(data.snr>snr_treshold)]

    fig, ax = plt.subplots(1,figsize=(10,6))
    foo=fooG
    bandL = ['lo','lo','hi','hi']
    polL = ['LL','RR','LL','RR']

    for cou in range(4):
        band=bandL[cou]
        pol=polL[cou]
        fooNi = foo[(foo.band==band)&(foo.polarization==polL[cou])]
        fmtloc = markers[cou]
        #cphaseLoc = np.asarray(fooNi[phase_type])[:,1]
        cphaseLoc = np.asarray(fooNi[phase_type])
        #print([np.shape(cphaseLoc),np.shape(fooNi.gmst),np.shape(fooNi.sigmaCP)])
        if conj==True:
            cphaseLoc= -cphaseLoc
        cphaseLoc = np.mod(cphaseLoc + shift,360) - shift

        gtime = np.mod(np.asarray(fooNi.gmst)+tshift,24)-tshift
        ax.errorbar(gtime,cphaseLoc,errscale*fooNi[error_type],fmt=fmtloc,capsize=5,label=band+' '+pol,markersize=ms)
        
        if timerange!='':
            xtime = timerange
        try:
            ax.legend()
        except IndexError:
            pass
    plt.grid()
    [x1,x2,y1,y2]=ax.axis()
    if y_range==[]:
        ax.axis([x1,x2,y1,y2])
    else:
        ax.axis([x1,x2]+y_range)
    [x1,x2,y1,y2]=ax.axis()
    if timerange=='':
        pass
    else:
        ax.axis(timerange+[y1,y2])
    ax.axhline(0,linestyle='--',color= (0.25, 0.25, 0.25))
    ax.set_xlabel('GMST [h]',fontsize=fonts)
    ax.set_ylabel('closure phase [deg]',fontsize=fonts)
    #ax.set_title(sour+', '+Z2SMT[triangle[0]]+'-'+Z2SMT[triangle[1]]+'-'+Z2SMT[triangle[2]]+', '+band+' band'+', '+polar[couP],fontsize=13)
    ax.set_title(sour+', '+dict_night[expt]+', '+Z2SMT[triangle[0]]+'-'+Z2SMT[triangle[1]]+'-'+Z2SMT[triangle[2]],fontsize=fonts)
    plt.tick_params(axis='both', labelsize=fonts-1)
    plt.grid()
    try:     
        #ax.legend(fontsize=fonts-1,bbox_to_anchor=(1.0, 1.0))
        ax.legend(fontsize=fonts-1)


    except IndexError:
        pass
    plt.tight_layout()
    if savefig==True:
        tit= sour+'_'+Z2SMT[triangle[0]]+'_'+Z2SMT[triangle[1]]+'_'+Z2SMT[triangle[2]]+'_'+band+'_'+polar[couP]+custom_title
        plt.savefig(tit+'.pdf')
    plt.show()

def plot_uv_coverage_publ(sour = '3C279',baseL = baseL,bandL=['lo','hi'],polarL=['LL','RR'],data=[],savefig=False,custom_title='',snrCut=2.,redundant=False,sc=1e6,all_baselines=False,locsourlabel=[6,8.5]):
    import matplotlib.transforms as mt
    data=data[data.snr>snrCut]
    data2=data[data.snr>snrCut]
    #sns.set_context("talk")
    
    sns.set(font_scale = 1.7)
    sourcelabelsize=17
    sns.set_style('white')
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    t= np.linspace(0,2*np.pi,256)
    uas = (1e3)*np.pi/180/60/60
    w0 = 1/(50*uas)
    w1 = 1/(25*uas)
    
    data2['u'] = -data['u']
    data2['v'] = -data['v']
    dataF = pd.concat([data,data2],ignore_index=True)
    dataF['u'] = dataF['u']/1e3
    dataF['v'] = dataF['v']/1e3
    dataF = dataF[dataF.snr>snrCut].copy()
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in polarL,dataF.polarization))]
    #fooRR['ratio_RL2RR'] = np.asarray(fooRL.amp)/np.asarray(fooRR.amp)
    #sns.lmplot(x='mu',y='mv',data=fooRR,hue='expt_no',fit_reg=False,size = 5,aspect = 1)
    Nb = len(set(fooRR.baseline.unique())&set(baseL))
    fooRR['baseline'] = list(map(lambda x: Z2SMT[x[0]]+'-'+Z2SMT[x[1]],fooRR['baseline']))
    #print('all baselines= ',all_baselines )
    baseL= list(fooRR.baseline.unique())
    if all_baselines==False:
        if redundant==False:
            baseL = ['AX','AZ','AP','AS','AL','AY','ZL','LP','LS','ZP','ZS','PS','LY','PY','SY','ZY']
            baseL = list(map(lambda x: Z2SMT[x[0]]+'-'+Z2SMT[x[1]],baseL))
            baseL = list(set(baseL)&set(fooRR.baseline.unique()))
            baseL=sorted(baseL)
        elif redundant==True:
            baseL = list(fooRR.baseline.unique())
            #baseL = [x for x in baseL if (('X' in x)|('J' in  x))&(x!='AX')]
            baseL = [x for x in baseL if (('APEX' in x)|('JCMT' in  x))&(x!='ALMA-APEX')]
            baseL = list(set(baseL)&set(fooRR.baseline.unique()))
            baseL=sorted(baseL)

    #baseL = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','YZ']  
    
    colors=['k','mediumblue','red','darkgreen','lime','magenta','blueviolet','orange','yellow','cyan','olivedrab','salmon','saddlebrown','dodgerblue','tomato','tan']
    #current_palette=dict(zip(baseL,colors))

    palette_dict = {'ALMA-APEX':'k','JCMT-SMA':'k','SMT-LMT':'lime','ALMA-LMT':'mediumblue','APEX-LMT':'mediumblue',
    'SMT-SMA':'red','SMT-JCMT':'red','LMT-SMA':'cyan','JCMT-LMT':'cyan',
    'ALMA-SMT':'magenta','APEX-SMT':'magenta','ALMA-SPT':'blueviolet','APEX-SPT':'blueviolet',
    'ALMA-IRAM30':'orange','APEX-IRAM30':'orange','ALMA-SMA':'darkgreen','ALMA-JCMT':'darkgreen','APEX-SMA':'darkgreen','APEX-JCMT':'darkgreen',
    'LMT-SPT':'yellow','LMT-IRAM30':'tomato','SMA-SPT':'olivedrab','JCMT-SPT':'olivedrab',
    'SMT-SPT':'salmon', 'IRAM30-SPT':'saddlebrown','IRAM30-SMA':'tan','JCMT-IRAM30':'tan',
    'SMT-IRAM30':'dodgerblue'}
    current_palette=palette_dict
    #current_palette={'AX':'k','AZ':'b'}
    #current_palette = sns.color_palette("Dark2", Nb)
    #sns.set_palette(current_palette)
    markers=["o"]*8+["o"]*8+["v"]*8
    markers = markers[:Nb]
    
    ######PLOTTING
    g = sns.lmplot(x='u',y='v',data=fooRR,hue='baseline',hue_order=tuple(sorted(baseL)),fit_reg=False,size = 6,aspect=1.2,scatter_kws={"s": 40},palette=current_palette,legend=False)#markers=markers
    g.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    g.set(xlim=(-9, 9), ylim=(-9, 9), xticks=range(-8,9,2), yticks=range(-8,9,2))
    ######PLOTTING

    #sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
    plt.axvline(0,linestyle='-',color= (0.5, 0.5, 0.5), zorder=0)
    plt.axhline(0,linestyle='-',color= (0.5, 0.5, 0.5), zorder=0)
    plt.axis([-9,9,-9,9])
    #plt.grid()
    plt.plot(w0*np.sin(t),w0*np.cos(t),'--',color= (0.5, 0.5, 0.5),zorder=0)
    plt.plot(w1*np.sin(t),w1*np.cos(t),'--',color= (0.5, 0.5, 0.5),zorder=0)
    
    ####FONTS SIZES
    labelfontsize=15
    arcfotsize=13

    plt.xlabel('u (G$\lambda$)')
    plt.ylabel('v (G$\lambda$)')#,fontsize=labelfontsize)
    if sour=='SGRA':
        sour='Sgr A*'

    plt.text(locsourlabel[0], locsourlabel[1],sour, fontsize=sourcelabelsize, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2))
    #plt.title(sour,fontsize=13)
    #plt.title(sour+' '+custom_title,fontsize=13)
    
    #sc = 1e6
    plt.axis([-10,10,-10,10])
    '''
    shY = 2
    plt.text(-16.8, 0+shY, 'A = ALMA/APEX', fontsize=13)
    plt.text(-16.8, -0.8+shY, 'L = LMT', fontsize=13)
    plt.text(-16.8, -1.6+shY, 'Z = SMT', fontsize=13)
    plt.text(-16.8, -2.4+shY, 'S = SMA/JCMT', fontsize=13)
    plt.text(-16.8, -3.2+shY, 'Y = SPT', fontsize=13)
    plt.text(-16.8, -4.0+shY, 'P = PICOVEL', fontsize=13)
    plt.text(-16.8, -4.8+shY, 'X = APEX', fontsize=13)
    '''
    r1 = 7.4; a1 = np.pi*0.3
    r2=3.3; a2=np.pi*0.35
    plt.text(r1*np.cos(a1),r1*np.sin(a1), '25 $\mu$as', fontsize=12,rotation=-42)
    plt.text(r2*np.cos(a2),r2*np.sin(a2), '50 $\mu$as', fontsize=12,rotation=-42)
    #plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.3,top=0.95)
    if len(baseL)>16:
        lgd = plt.legend(bbox_to_anchor=(1, 1),fontsize=14,ncol=2,markerscale=1.5)
    else:
        lgd = plt.legend(bbox_to_anchor=(1, 1),fontsize=14,markerscale=1.5) 
    if savefig==True:
        tit = 'uv_cover_'+sour
        g.savefig(tit+'.pdf',bbox_inches='tight')
        #plt.savefig(tit+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()



def plot_snr_hist_publ(data,sour,ymax=250,xmax=15000,deltasource=[0.3,0.93],nbins=100,savefig=False):
    #sourL=['M87','SGRA','3C279','OJ287']
    binmax=xmax
    #ymax=250
    xmin=0.5
    labelfontsize=18
    tickfontsize=17
    sourcelabelsize=17
    sns.set(font_scale = 1.7)
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    #for sour in sourL:
    #bins=np.linspace(0,binmax,128)
    bins=np.logspace(-1,np.log(binmax),nbins)
    if sour!='All sources':
        foo=data[data.source==sour]
    else:
        foo=data
    foo = foo[foo.polarization.str[0]==foo.polarization.str[1]]
    foo['is_alma'] = foo.baseline.str.contains('A')
    plt.figure(figsize=(8,6))
    sns.distplot(foo[foo.is_alma==True].snr,bins=bins,kde=False,label='ALMA baselines',hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1})
    sns.distplot(foo[foo.is_alma==False].snr,bins=bins,kde=False,label='non-ALMA \n baselines',hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1})
    plt.axis([xmin,binmax,0.5,ymax])
    #plt.axvline(6,linestyle='--',color=(0.5,0.5,0.5),label='detection threshold')
    #plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=tickfontsize)
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.xlabel('Signal-to-Noise Ratio',fontsize=labelfontsize)
    plt.ylabel('Number of Detections',fontsize=labelfontsize)
    #plt.title(sour)
    if sour=='SGRA':
        sour='Sgr A*'
    plt.text(xmin+deltasource[0], ymax*deltasource[1],sour, fontsize=sourcelabelsize, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2))
    
    if sour=='Sgr A*':
        sour='SGRA'
    if savefig==True:
        tit = 'snr_vis_'+sour
        plt.savefig(tit+'.pdf',bbox_inches='tight')
        #plt.savefig(tit+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    

def plot_snr_hist_publ_M87_SGRA(data,ymax=250,xmax=15000,deltasource=[0.3,0.93],nbins=100,savefig=False):
    #sourL=['M87','SGRA','3C279','OJ287']
    binmax=xmax
    #ymax=250
    xmin=0.5
    labelfontsize=18
    tickfontsize=17
    sourcelabelsize=17
    sns.set(font_scale = 1.7)    
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    #for sour in sourL:
    #bins=np.linspace(0,binmax,128)
    bins=np.logspace(-1,np.log(binmax),nbins)
    foo=data[(data.source=='M87')|(data.source=='SGRA')]
    foo = foo[foo.polarization.str[0]==foo.polarization.str[1]]
    foo['is_alma'] = foo.baseline.str.contains('A')
    plt.figure(figsize=(8,6))
    sns.distplot(foo[foo.is_alma==True].snr,bins=bins,kde=False,label='ALMA baselines',hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1})
    sns.distplot(foo[foo.is_alma==False].snr,bins=bins,kde=False,label='non-ALMA \n baselines',hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1})
    plt.axis([xmin,binmax,0.5,ymax])
    #plt.axvline(6,linestyle='--',color=(0.5,0.5,0.5),label='detection threshold')
    #plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=tickfontsize)
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.xlabel('Signal-to-Noise Ratio',fontsize=labelfontsize)
    plt.ylabel('Number of Detections',fontsize=labelfontsize)
    #plt.title(sour)
    plt.text(xmin+deltasource[0], ymax*deltasource[1],'Sgr A* + M87', fontsize=sourcelabelsize, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2))

    if savefig==True:
        tit = 'snr_vis_SGRA_M87'
        plt.savefig(tit+'.pdf',bbox_inches='tight')
        #plt.savefig(tit+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()



    



    #baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']  
#Fourth TYPE OF PLOT: u-v coverage by baseline
def plot_coverage_2(sour = '3C279',baseL = baseL,bandL=['lo','hi'],polarL=['LL','RR'],data=[],savefig=False,custom_title='',y_range='',red=False,onlyred=False,loglog=False,alphabet_bsl=False,
alpha=0.25):
    #sns.set_context("talk")
    if 'baselength' not in data.columns:
        data = ut.add_baselength(data)
    #baseL0 = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','ZY']
    #redundant = ['JS','XL','XZ','XP','XS','JL','ZJ','AJ','XJ','JP','XY','JY']
    baseL0 = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','YZ']
    redundant = ['JS','LX','XZ','PX','SX','JL','JZ','AJ','JX','JP','XY','JY']
    if alphabet_bsl==True:
        redundant=[''.join(sorted(x)) for x in redundant]
        baseL0=[''.join(sorted(x)) for x in baseL0]
    #print(red==True)
    if red==True:
        baseL0=baseL0+redundant
        if onlyred==True:
            baseL0=redundant

        #print('baseL0 ', baseL0)
    dataF = data
    #print(dataF.baseline.unique())
    #print('Size to plot0:',np.shape(dataF))
    #fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in polarL,dataF.polarization))&list(map(lambda x: x in bandL,dataF.band))]
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL0,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in polarL,dataF.polarization))]
    
    #print('Size to plot:',np.shape(fooRR))
    #print(fooRR.baseline.unique())
    #print(fooRR)
    #fooRR['ratio_RL2RR'] = np.asarray(fooRL.amp)/np.asarray(fooRR.amp)
    #sns.lmplot(x='mu',y='mv',data=fooRR,hue='expt_no',fit_reg=False,size = 5,aspect = 1)
    #Nb = len(set(fooRR.baseline.unique())&set(baseL))
    #baseL0 = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','ZY']
    Nb = len(set(fooRR.baseline.unique())&set(baseL0))
    #print(set(fooRR.baseline.unique()))
    #print(set(fooRR.baseline.unique())&set(baseL0))
    baseL = ['AX','AZ','AP','AS','AL','AY','ZL','LP','LS','ZP','ZS','PS','LY','PY','SY','ZY']
    if alphabet_bsl==True:
        baseL=[''.join(sorted(x)) for x in baseL]
    #baseL = ['AX','AZ','AP','AS','AL','AY','LZ','LP','LS','PZ','SZ','PS','LY','PY','SY','ZY']
    colors=['k','mediumblue','red','darkgreen','lime','magenta','blueviolet','orange','yellow','cyan','olivedrab','salmon','saddlebrown','dodgerblue','tomato','tan']
    palette_dict = {'AX':'k','JS':'k','LZ':'lime','AL':'mediumblue','LX':'mediumblue',
    'SZ':'red','JZ':'red','LS':'cyan','JL':'cyan',
    'AZ':'magenta','XZ':'magenta','AY':'blueviolet','XY':'blueviolet',
    'AP':'orange','PX':'orange','AS':'darkgreen','AJ':'darkgreen','SX':'darkgreen','JX':'darkgreen',
    'LY':'yellow','LP':'tomato','SY':'olivedrab','JY':'olivedrab',
    'YZ':'salmon', 'PY':'saddlebrown','PS':'tan','JP':'tan',
    'PZ':'dodgerblue'}
    
    #print(baseL0)

    if red==True:
        baseL=baseL+redundant
        colors=colors+['darkgray']*len(redundant)
        if onlyred==True:
            baseL=redundant

    current_palette=dict(zip(baseL,colors))
    #current_palette={'AX':'k','AZ':'b'}
    #current_palette = sns.color_palette("Dark2", Nb)
    #sns.set_palette(current_palette)
    markers=["o"]*8+["o"]*8+["v"]*8+["v"]*8
    markers = markers[:Nb]

    current_palette=palette_dict
    #
    g = sns.lmplot(x='baselength',y='amp',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=2,scatter_kws={"s": 50,'alpha':alpha},markers=markers,palette=current_palette)
    plt.grid()
    #sns.lmplot(x='baselength',y='amp',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=2,scatter_kws={"s": 30,'alpha':0.25},markers=markers,palette=current_palette)
    plt.axvline(0,linestyle='--',color= (0.5, 0.5, 0.5))
    plt.axhline(0,linestyle='--',color= (0.5, 0.5, 0.5))
    plt.grid()
    plt.xlabel('uv distance [$\lambda$]',fontsize=13)
    plt.ylabel('flux [Jy]',fontsize=13)
    #plt.title(sour,fontsize=13)
    plt.title(sour+' '+custom_title,fontsize=13)
    for lh in g._legend.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [50] 
    
    
    #plt.axis([0,9000,0,100])
    if y_range=='':
        plt.axis([-1.e9,9e9,-0.02,8.5])
    else:
        plt.axis([-1e8,9e9]+y_range)
    plt.gcf().subplots_adjust(left=0.3,top=0.95)
    if savefig==True:
        tit = 'uvdist_amp_'+sour
        plt.savefig(tit+'.pdf')
    plt.grid()
    plt.grid()
    if loglog==True:
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.axis([4.e8,1e10,1e-3,1e1])
        plt.grid()
    plt.grid()
    plt.grid()
    plt.show()

