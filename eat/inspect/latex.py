import os


def generate_tex(path_figs = 'figs/', title=' ',date=' ',plots_per_page = 8, outname='texout.tex'):

    with open('raw_latex_text','r') as texfile:
        data = texfile.readlines()

    beg_data = data[:11]
    end_data = data[-3:]
    subplot_data = data[11:14]
    titleL = ['\\center{'+FileTitle+'}\\\\\n']
    dateL = ['\\center{'+FileDate+'}\\\\\n']
    beg_data_new = beg_data[:6]+titleL+dateL+beg_data[8:]
    beg_plot_line = '\\includegraphics[width=\\textwidth]{'
    make_new_plot = ['\\end{figure} \n','\\begin{figure}[h!] \n', '\\centering \n']

    listFigs = os.listdir(path_figs)
    listFigs = [x for x in listFigs if x[0]!= '.']
    all_subplots = []
    cou = 0
    for fig in listFigs:
        subplot_fig = subplot_data
        foo = path_figs+fig+'} \n'
        subplot_fig[1] = beg_plot_line+foo
        cou = cou+1
        if cou == plots_per_page:
            cou=0
            subplot_fig = subplot_fig+make_new_plot    
        
        all_subplots = all_subplots+subplot_fig
    
    data_ext = beg_data_new+all_subplots+end_data

    #save tex script in a file to copy to latex editor
    texout = open(outname,'w')
    for x in data_ext:
        texout.write(x)
    texout.close()