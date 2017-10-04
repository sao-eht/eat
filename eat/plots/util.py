"""
plotting utilities
"""
import numpy as np
import matplotlib.pyplot as plt

# plot as separated segments
# dx: distance at which to break up data
# *args, **kwargs sent to plt.plot()
# dbreak keyword to set break distance
def breakplot(x, y, *args, **kwargs):
	if 'yerr' in kwargs:
		yerr = kwargs.pop('yerr')
	else:
		yerr = None
	i2 = np.arange(len(x))
	dx = np.diff(x)
	# some logic to set clustering, balance clustering against number of segments
	if 'dbreak' not in kwargs:
		dbreak=0.5
	isplit = np.nonzero(np.abs(dx) > dbreak)[0]
	if len(isplit) > 0:
		isplit += 1
	for i3 in np.split(i2, isplit):
		if yerr is not None:
			plt.errorbar(x[i3], y[i3], yerr[i3], *args, **kwargs)
		else:
			plt.plot(x[i3], y[i3], *args, **kwargs)

# remove lines across gaps in data
# this appears to stop working for N>=1000, probably due to lines.py:706 in matplotlib 2.0.0
def rmgaps(dbreakx=2., dbreaky=None):
    if dbreaky == None:
        dbreaky = 1e12
    for line in plt.gca().get_lines():
        codes = np.full(len(line._path.vertices[:,0]), 2, dtype=np.uint8)
        dx = np.hstack(([dbreakx+1.], np.abs(np.diff(line._path.vertices[:,0]))))
        dy = np.hstack(([dbreaky+1.], np.abs(np.diff(line._path.vertices[:,1]))))
        codes[(dx > dbreakx) | (dy > dbreaky)] = 1
        line._path.codes = codes
    plt.draw()

# remove lines across gaps in data
def rmgaps2(dbreak=2.):
	for line in plt.gca().get_lines():
        # this is not the right output tuple for cleaned() -- don't remember why I started this 2nd function..
		# v, c = line._path.cleaned()
		codes = np.full(len(line._path.vertices[:,0]), 2, dtype=np.uint8)
		dx = np.hstack(([dbreak+1.], np.abs(np.diff(line._path.vertices[:,0]))))
		codes[dx > dbreak] = 1
		line._path.codes = codes
	plt.draw()

# remove lines across gaps in data -- or this one
def rmgaps2(dbreak=2.):
	for line in plt.gca().get_lines():
		None

# tag a plot using fake "legend"
# note: this is probably unnecessary now, you can use loc with anchored text
# however this is still useful if you want loc="best"
# http://stackoverflow.com/questions/7045729/automatically-position-text-box-in-matplotlib
def tag(tagstr, **kwargs):
    from matplotlib.legend import Legend
    ax = plt.gca()
    h, = ax.plot(np.NaN, np.NaN, '-', color='none') # empty line for legend
    leg = Legend(ax, (h,), (tagstr,), handlelength=0, handletextpad=0, **kwargs)
    for item in leg.legendHandles:
        item.set_visible(False)
    ax.add_artist(leg)

def figsize(width, height):
    plt.setp(plt.gcf(), figwidth=width, figheight=height)

# set y0 to something between 0 and 1 so that ylog plots work, w=weight
def stepcdf(ranks, x0=0, y0=0.1, w=1.0, **kwargs):
    if type(w) is int or type(w) is float:
        w = w * np.ones_like(ranks)
    sidx = np.argsort(ranks)
    srank = ranks[sidx] # ranks from small to big
    wtot = np.sum(w)
    count = np.empty(len(w) + 1)
    count[0] = wtot
    count[1:] = wtot - np.cumsum(w[sidx]) # counts # of events between rank and previous rank
    count[-1] += w[sidx[-1]]*y0 # some slob so that logscale plots do not get eaten instead of zero
    ypt = np.vstack((count, count)).T.ravel()[:-1]
    xpt = np.hstack((x0, np.vstack((srank, srank)).T.ravel()))
    plt.plot(xpt, ypt, **kwargs)

# confidence interval for one-sided Gaussian RV
def normconfidence(maxval=10.):
    from scipy.stats import norm
    xx = np.linspace(0, maxval, 200)
    sf = nomm.sf(xx)
