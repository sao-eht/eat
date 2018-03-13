import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interact_manual
import seaborn as sns
import statsmodels.stats.stattools as sss

def plot_box_data_amplitude(corr,save_figure):
    global g, goo
    goo = corr; 
    if sour != 'all sources':
        goo = goo[goo.source==sour]
    if expt != 'all expt':
        goo = goo[goo.expt_no==expt]
    if band != 'both bands':
        goo = goo[goo.band==band]
    if polar != 'both pol':
        if polar == 'cross pol':
            goo = goo[list(map(lambda x: x[0]!=x[1],goo.polarization))]
        else:
            goo = goo[goo.polarization==polar]
    else:
        goo = goo[list(map(lambda x: x[0]==x[1],goo.polarization))]
    goo = goo[goo.baseline==basel]
    
    if (band != 'both bands') & (polar != 'both pol'):
        my_hue= None; my_color= 'red'
        if sour == 'all sources':
            my_hue= 'source';my_color= None
        elif expt == 'all expt':
            my_hue= 'expt_no';my_color= None
    elif (band != 'both bands') & (polar == 'both pol'):
        my_hue='polarization';my_color= None
        if sour == 'all sources':
            goo['sourcepolarization'] = list(map(lambda x: x[0] +'-'+x[1], zip(list(goo['source']),list(goo['polarization']))))
            my_hue = 'sourcepolarization';my_color= None
        elif expt == 'all expt':
            goo['exptpolarization'] = list(map(lambda x: x[0] +'-'+str(x[1]), zip(list(goo['expt_no']),list(goo['polarization']))))
            my_hue = 'exptpolarization';my_color= None
    elif (band == 'both bands') & (polar != 'both pol'):
        my_hue='band';my_color= None
        if sour == 'all sources':
            goo['bandsource'] = list(map(lambda x: x[0] +'-'+x[1], zip(list(goo['band']),list(goo['source']))))
            my_hue = 'bandsource';my_color= None
        elif expt == 'all expt':
            goo['exptband'] = list(map(lambda x: str(x[0]) +'-'+(x[1]), zip(list(goo['expt_no']),list(goo['band']))))
            my_hue = 'exptband';my_color= None
    else:
        goo['bandopolarization'] = list(map(lambda x: x[0] +'-'+x[1], zip(list(goo['band']),list(goo['polarization']))))
        my_hue = 'bandopolarization';my_color= None
        if sour == 'all sources':
            goo['bandosourcepolarization'] = list(map(lambda x: x[0] +'-'+x[1]+'-'+x[2], zip(list(goo['band']),list(goo['source']),list(goo['polarization']))))
            my_hue = 'bandosourcepolarization';my_color= None
        elif expt == 'all expt':
            goo['bandoexptpolarization'] = list(map(lambda x: x[0] +'-'+str(x[1])+'-'+x[2], zip(list(goo['band']),list(goo['expt_no']),list(goo['polarization']))))
            my_hue = 'bandoexptpolarization';my_color= None
    try:
        plt.figure(figsize=(10,8))      
        sns_plot = sns.boxplot(x='scan_id',y='amp', data = goo.sort_values('scan_id'),color=my_color,hue=my_hue)
        plt.grid()
        #plt.tight_layout()
        plt.xticks(rotation=45)
        plt.title(basel+' , '+sour+' , '+str(expt)+' , '+band+' , '+polar+' , '+'5s', fontsize=15 )
        plt.ylabel('corr coeff',fontsize=15)
        plt.xlabel('scan_id',fontsize=15)
        plt.grid()
        plt.show()
        #print(save_figure)
        if save_figure == True:
            figtitle=basel+'_'+sour+'_'+str(expt)+'_'+band+'_'+polar+'_'+'5s'
            fig = sns_plot.get_figure()
            fig.savefig(figtitle+'.png')
    except ValueError:
        print('NO DATA!')

def reset_plots_amplitude(corr,save_figure=False):
    global sour, expt, band, polar, basel
    widget_source = widgets.Dropdown(
        options=list(corr.source.unique())+['all sources'],
        value='3C279',description='Source:',disabled=False,)
    widget_experiment = widgets.Dropdown(
        options=[3597, 3598, 3599,3600,3601, 'all expt'],
        value=3597,description='Experiment:',disabled=False,)
    widget_band = widgets.Dropdown(
        options=['lo','hi','both bands'],
        value='lo',description='Band:',disabled=False,)
    widget_polarization = widgets.Dropdown(
        options=list(corr.polarization.unique())+['both pol','cross pol'],
        value='LL',description='Polarization:',disabled=False,)
    widget_baseline = widgets.Dropdown(
        options=sorted(list(corr.baseline.unique())),
        value=list(corr.baseline.unique())[0],
        description='Baseline:',disabled=False,)
    def interact_source(x): global sour; sour = x
    def interact_experiment(x): global expt; expt = x
    def interact_band(x): global band; band = x
    def interact_polarization(x): global polar; polar = x
    def interact_baseline(x): global basel; basel = x
    interact(interact_source, x=widget_source)
    interact(interact_experiment, x=widget_experiment)
    interact(interact_band, x=widget_band)
    interact(interact_polarization, x=widget_polarization)
    interact(interact_baseline, x=widget_baseline)
    button = widgets.Button(description="Add plot")
    display(button)
    def on_button_clicked(b):
        plot_box_data_amplitude(corr,save_figure)
    button.on_click(on_button_clicked)

def plot_scan_data(corr, save_figure):
    global g, goo, goosc
    goo = corr; 
    if sour != 'all sources':
        goo = goo[goo.source==sour]
    if expt != 'all expt':
        goo = goo[goo.expt_no==expt]
    if band != 'both bands':
        goo = goo[goo.band==band]
    if polar != 'both pol':
        if polar == 'cross pol':
            goo = goo[list(map(lambda x: x[0]!=x[1],goo.polarization))]
        else:
            goo = goo[goo.polarization==polar]
    else:
        goo = goo[list(map(lambda x: x[0]==x[1],goo.polarization))]
    goo = goo[goo.baseline==basel]
    goosc = goo[goo.scan_id==scanid]
    #print(goosc.baseline)
    goosc.loc[:,'real']=np.asarray(goosc['amp'])*np.asarray(np.sin(goosc['resid_phas']*np.pi/180))
    goosc.loc[:,'imag']=np.asarray(goosc['amp'])*np.asarray(np.cos(goosc['resid_phas']*np.pi/180))
    
    '''
    try:
        fig, ax = plt.subplots(2,1, figsize=(10,8)) 
        ax[0].hist(goosc.amp)
    except ValueError:
        print('NO DATA!')
    '''
    try:
        plt.figure(figsize=(10,8))
        #print('amp', goosc.amp)
        #sns_plot = sns.distplot(np.asarray(goosc.amp))
        st = np.std(goosc.amp)
        m = np.mean(goosc.amp)
        plt.hist(goosc.amp)
        plt.grid()
        #plt.axis('equal')
        #plt.tight_layout()
        #plt.xticks(rotation=45)
        plt.title(basel+' , '+sour+' , '+str(expt)+' , '+band+' , '+polar+' , '+scanid+' , '+'1s'+' , '+str(len(goosc.amp))+' points', fontsize=15 )
        plt.xlabel('corr coeff',fontsize=15)
        plt.ylabel('counts',fontsize=15)
        plt.grid()
        plt.show()
        #print(save_figure)
        if save_figure == True:
            figtitle=basel+'_'+sour+'_'+str(expt)+'_'+band+'_'+polar+'_'+scanid+' , '+'1s'
            fig = sns_plot.get_figure()
            fig.savefig(figtitle+'.png')
    except ValueError:
        print('NO DATA!')

       
    try:
        plt.figure(figsize=(10,5))      
        #sns_plot = sns.regplot(x='real',y='imag',data=goosc,fit_reg=False)
        #plt.plot(goosc.real,goosc.imag,'*')
        plt.plot(goosc.datetime,goosc.amp,'*',label='amplitudes')
        plt.grid()
        goosc['outlier'] = adj_box_outlier(goosc.amp)
        plt.plot(goosc[goosc['outlier']==True].datetime,goosc[goosc['outlier']==True].amp,'r*',label='adaptive box rule amplitude outliers')
        #plt.tight_layout()
        #plt.xticks(rotation=45)
        plt.title(basel+' , '+sour+' , '+str(expt)+' , '+band+' , '+polar+' , '+scanid+' , '+'1s', fontsize=15 )
        plt.ylabel('corr coeff',fontsize=15)
        plt.xlabel('time UTC',fontsize=15)
        plt.legend()
        plt.show()
        #print(save_figure)
        if save_figure == True:
            figtitle=basel+'_'+sour+'_'+str(expt)+'_'+band+'_'+polar+'_'+scanid+' , '+'1s'
            fig = sns_plot.get_figure()
            fig.savefig(figtitle+'.png')
    except ValueError:
        print('NO DATA!')

    try:
        plt.figure(figsize=(10,5))      
        #sns_plot = sns.regplot(x='real',y='imag',data=goosc,fit_reg=False)
        plt.ylabel('phase [deg]',fontsize=15)
        plt.xlabel('time UTC',fontsize=15)
        plt.plot(goosc.datetime,np.mod(goosc.resid_phas+180,360)-180,'*')
        plt.plot(goosc[goosc['outlier']==True].datetime,np.mod(goosc[goosc['outlier']==True].resid_phas+180,360)-180,'r*',label='adaptive box rule amplitude outliers')
        plt.plot()
        plt.grid()
        #plt.tight_layout()
        #plt.xticks(rotation=45)
        plt.title(basel+' , '+sour+' , '+str(expt)+' , '+band+' , '+polar+' , '+scanid+' , '+'1s', fontsize=15 )
        #plt.ylabel('corr coeff',fontsize=15)
        #plt.xlabel('scan_id',fontsize=15)
        plt.legend()
        plt.show()
        #print(save_figure)
        if save_figure == True:
            figtitle=basel+'_'+sour+'_'+str(expt)+'_'+band+'_'+polar+'_'+scanid+' , '+'1s'
            fig = sns_plot.get_figure()
            fig.savefig(figtitle+'.png')
    except ValueError:
        print('NO DATA!')
    return goosc
    
def reset_plots_scan(corr,save_figure=False):
    global sour, expt, band, polar, basel, scan_id_loc, goo, goosc
    #goosc = 0
    widget_source = widgets.Dropdown(
        options=list(corr.source.unique())+['all sources'],
        value=sour,description='Source:',disabled=False,)
    widget_experiment = widgets.Dropdown(
        options=[3597, 3598, 3599,3600,3601, 'all expt'],
        value=expt,description='Experiment:',disabled=False,)
    widget_band = widgets.Dropdown(
        options=['lo','hi','both bands'],
        value=band,description='Band:',disabled=False,)
    widget_polarization = widgets.Dropdown(
        options=list(corr.polarization.unique())+['both pol','cross pol'],
        value=polar,description='Polarization:',disabled=False,)
    widget_baseline = widgets.Dropdown(
        options=sorted(list(corr.baseline.unique())),
        value=basel,
        description='Baseline:',disabled=False,)
    widget_scan = widgets.Dropdown(
        options=list(goo.scan_id.unique()),
        value=list(goo.scan_id.unique())[0],
        description='Scan ID:',disabled=False,)
    def interact_source(x): global sour; sour = x
    def interact_experiment(x): global expt; expt = x
    def interact_band(x): global band; band = x
    def interact_polarization(x): global polar; polar = x
    def interact_baseline(x): global basel; basel = x
    def interact_scan(x): global scanid; scanid = x

    interact(interact_source, x=widget_source)
    interact(interact_experiment, x=widget_experiment)
    interact(interact_band, x=widget_band)
    interact(interact_polarization, x=widget_polarization)
    interact(interact_baseline, x=widget_baseline)
    interact(interact_scan, x=widget_scan)
    button = widgets.Button(description="Add plot")
    display(button)
    def on_button_clicked(b):
        global goosc
        goosc = plot_scan_data(corr, save_figure)
        #print(goosc.source.unique())
        #return goosc
    button.on_click(on_button_clicked)
    

def range_adjust_box(vec):
    quart3 = np.percentile(vec,75)
    quart1 = np.percentile(vec,25)
    iqr = quart3-quart1
    mc = float(sss.medcouple(vec))
    if mc > 0:
        whisk_plus = 1.5*iqr*np.exp(3*mc)
        whisk_min = 1.5*iqr*np.exp(-4*mc)
    else:
        whisk_plus = 1.5*iqr*np.exp(4*mc)
        whisk_min = 1.5*iqr*np.exp(-3*mc)       
    range_plus = quart3 + whisk_plus
    range_min = quart1 - whisk_min   
    return [range_min, range_plus]
def adj_box_outlier(vec):
    range_box = range_adjust_box(vec)
    is_out = (vec>range_box[1])|(vec<range_box[0])
    return is_out
def adj_box_outlier_minus(vec):
    range_box = range_adjust_box(vec)
    is_out = (vec<range_box[0])
    return is_out
def adj_box_outlier_plus(vec):
    range_box = range_adjust_box(vec)
    is_out = (vec>range_box[1])
    return is_out