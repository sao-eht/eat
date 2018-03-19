import numpy as np
import pandas as pd
import seaborn as sns
from eat.io import hops, util
from eat.hops import util as hu
import matplotlib.pyplot as plt
import seaborn as sns


dict_night = {3597: '04/04/2017',3598: '04/05/2017',3599: '04/06/2017', 3600: '04/09/2017',3601: '04/10/2017'}
SMT2Z = {'ALMA': 'A', 'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R', 'SMA':'S', 'SMT':'Z', 'PV':'P','SPT':'Y'}
Z2SMT = {v: k for k, v in SMT2Z.items()}
baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']
markers=["o", "d","v","^",'s']
markers_od = ["o", "d","o","d",'o']

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


def err_nights_time_amp(data,sour='3C279',base='AL',band='lo',polar='RR',errscale=1,errcolumn='sigma',time_type='gmst'):
    
    if 'Night' not in data.columns:
        data['Night'] =list(map(lambda x: dict_night[x],data.expt_no))
    if time_type=='gmst':
        util.add_gmst(data)
    elif time_type=='fmjd':
        data = ut.add_mjd(data)
        data = ut.add_fmjd(data)

    foo = data[(data.source==sour)&(data.baseline==base)&(data.band==band)&(data.polarization==polar)]
    Nights = list(foo.Night.unique())
    Ni = len(Nights)
    markers=["o", "d","o","d",'o','d']
    markers = markers[:Ni]
    
    sns.set_style('darkgrid')
    plt.figure(figsize=(10,6))
    for cou in range(Ni):
        LocNight = Nights[cou]
        fooNi = foo[foo.Night==LocNight]
        fmtloc = markers[cou]
        plt.errorbar(fooNi.gmst,fooNi.amp,errscale*fooNi.sigma,fmt=fmtloc,capsize=5,label=LocNight)
    
    plt.grid()
    plt.xlabel(time_type,fontsize=14)
    plt.ylabel('amplitude [Jy]',fontsize=14)
    #plt.title(sour+', '+Z2SMT[base[0]]+'-'+Z2SMT[base[1]]+', '+band+' band'+', '+polar,fontsize=13)
    plt.title(Z2SMT[base[0]]+'-'+Z2SMT[base[1]]+', '+band+' band'+', '+polar,fontsize=13)
    plt.grid()
    plt.legend()
    plt.show()

def err_nights_time_cphase(data,sour='3C279',triangle='ALX',band='lo',polar=['RR'],shift=0,errscale=1,
savefig=False,time_type='gmst',phase_type='cphase_fix_amp',error_type='sigmaCP',snr_treshold=1,conj=False,
ms=7,line=True,show_both_pol=False,y_range=[],custom_title=''):
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
            Nights = list(foo.Night.unique())
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

        Nights = list(foo.Night.unique())
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
            ax.errorbar(fooNi.gmst,cphaseLoc,errscale*fooNi[error_type],fmt=fmtloc,capsize=5,label=LocNight,markersize=ms)
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
        ax.axhline(0,linestyle='--',color= (0.25, 0.25, 0.25))
        ax.set_xlabel('GMST [h]',fontsize=fonts)
        ax.set_ylabel('closure phase [deg]',fontsize=fonts)
        #ax.set_title(sour+', '+Z2SMT[triangle[0]]+'-'+Z2SMT[triangle[1]]+'-'+Z2SMT[triangle[2]]+', '+band+' band'+', '+polar[couP],fontsize=13)
        ax.set_title(Z2SMT[triangle[0]]+'-'+Z2SMT[triangle[1]]+'-'+Z2SMT[triangle[2]]+', '+band+' band'+', '+polar[couP],fontsize=fonts)
        plt.tick_params(axis='both', labelsize=fonts-1)
        plt.grid()
        try:     
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
def plot_coverage(sour = '3C279',baseL = baseL,bandL=['lo','hi'],polarL=['LL','RR'],data=[],savefig=False,custom_title=''):
    sns.set_context("talk")
    t= np.linspace(0,2*np.pi,256)
    uas = np.pi/180/60/60/1e6
    w0 = 1/(50*uas)/1e6
    w1 = 1/(25*uas)/1e6
    data2 = data.copy()
    data2['u'] = -data['u']
    data2['v'] = -data['v']
    dataF = pd.concat([data,data2],ignore_index=True)
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in polarL,dataF.polarization))]
    #fooRR['ratio_RL2RR'] = np.asarray(fooRL.amp)/np.asarray(fooRR.amp)
    #sns.lmplot(x='mu',y='mv',data=fooRR,hue='expt_no',fit_reg=False,size = 5,aspect = 1)
    Nb = len(set(fooRR.baseline.unique())&set(baseL))
    baseL = ['AX','AZ','AP','AS','AL','AY','ZL','LP','LS','ZP','ZS','PS','LY','PY','SY','ZY']
    colors=['k','mediumblue','red','darkgreen','lime','magenta','blueviolet','orange','yellow','cyan','olivedrab','salmon','saddlebrown','dodgerblue','tomato','tan']
    current_palette=dict(zip(baseL,colors))
    #current_palette={'AX':'k','AZ':'b'}
    #current_palette = sns.color_palette("Dark2", Nb)
    #sns.set_palette(current_palette)
    markers=["o"]*8+["o"]*8+["v"]*8
    markers = markers[:Nb]
    sns.lmplot(x='u',y='v',data=fooRR,hue='baseline',fit_reg=False,size = 6,aspect=1.25,scatter_kws={"s": 80},markers=markers,palette=current_palette)
    plt.axvline(0,linestyle='--',color= (0.5, 0.5, 0.5))
    plt.axhline(0,linestyle='--',color= (0.5, 0.5, 0.5))
    plt.grid()
    plt.plot(w0*np.sin(t),w0*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    plt.plot(w1*np.sin(t),w1*np.cos(t),'--',color= (0.5, 0.5, 0.5))
    plt.xlabel('u [M$\lambda$]',fontsize=13)
    plt.ylabel('v [M$\lambda$]',fontsize=13)
    #plt.title(sour,fontsize=13)
    plt.title(sour+', '+custom_title,fontsize=13)
    
    plt.grid()
    plt.axis([-9000,9000,-9000,9000])
    
    shY = 2000
    plt.text(-18000, 0+shY, 'A = ALMA/APEX', fontsize=13)
    plt.text(-18000, -800+shY, 'L = LMT', fontsize=13)
    plt.text(-18000, -1600+shY, 'Z = SMT', fontsize=13)
    plt.text(-18000, -2400+shY, 'S = SMA/JCMT', fontsize=13)
    plt.text(-18000, -3200+shY, 'Y = SPT', fontsize=13)
    plt.text(-18000, -4000+shY, 'P = PICOVEL', fontsize=13)
    plt.text(-18000, -4800+shY, 'X = APEX', fontsize=13)
    
    r1 = 7400; a1 = np.pi*0.3
    r2=3300; a2=np.pi*0.35
    plt.text(r1*np.cos(a1),r1*np.sin(a1), '25 $\mu$as', fontsize=12,rotation=-42)
    plt.text(r2*np.cos(a2),r2*np.sin(a2), '50 $\mu$as', fontsize=12,rotation=-42)
    #plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.3,top=0.95)
    if savefig==True:
        tit = 'uv_cover_'+sour
        plt.savefig(tit+'.pdf')
    plt.show()




    
baseL = ['AL','AZ','AP','AS','AX','ZL','LP','LS','ZP','ZS','PS','AY','LY','PY','SY','ZY']
def plot_uv_snr(sour = '3C279',exptL=[3597,3598,3599,3600,3601],baseL = baseL,bandL=['lo','hi'],data=[],polar='both',savefig=False):

    data2 = data.copy()
    data2['u'] = -data['u']
    data2['v'] = -data['v']
    dataF = pd.concat([data,data2],ignore_index=True)
    fooRR = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='RR')]
    fooLL = dataF[(dataF.source==sour)&list(map(lambda x: x in baseL,dataF.baseline))&list(map(lambda x: x in bandL,dataF.band))&list(map(lambda x: x in exptL,dataF.expt_no))&(dataF.polarization=='LL')]
    if polar=='both':
        colors= np.maximum(1,np.minimum(3,np.log10((np.asarray(fooLL.snr)+np.asarray(fooRR.snr))/2)))
    elif polar=='LL':
        colors= np.maximum(1,np.minimum(3,np.log10((np.asarray(fooLL.snr)+np.asarray(fooLL.snr))/2)))
    elif polar=='RR':
        colors= np.maximum(1,np.minimum(3,np.log10((np.asarray(fooRR.snr)+np.asarray(fooRR.snr))/2)))

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
    cb.set_ticks([1,1.301,1.69897,2,2.301,2.69897,3], update_ticks=True)
    cb.set_ticklabels([10,20,50,100,200,500,1000], update_ticks=True)
    
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
        