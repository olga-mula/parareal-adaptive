
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import matplotlib.cm as cmx

from ode import VDP, Brusselator, Oregonator

# Dictionary of available odes
ode_dict = {'VDP': VDP, 'Brusselator': Brusselator, 'Oregonator': Oregonator}

# Receive args
# ------------
parser = argparse.ArgumentParser()
parser.add_argument(
	'-ode', '--ode_name', default='Brusselator',
	help='{VDP,Brusselator,Oregonator}')
parser.add_argument(
	'-fn', '--fn', default='prod-cluster', help='foldername')
args = parser.parse_args()
if args.ode_name not in ode_dict:
	raise Exception('ODE type '+args.ode_name+' not supported')

data = np.load(args.ode_name+'/'+args.fn+'/data.npz')['data']

# Plots
# -----
tl = [100, 300, 500, 700, 900, 1100]
nl = [10, 20, 30, 40, 50, 60, 70, 80, 100]
# neps = [1.0e-06, 1.0e-07, 1.0e-08, 1.0e-09, 1.0e-10]
neps = [1.e-6, 1.e-8]

# Color map
values = range(len(neps))
jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

pltstyle = plt.plot

for i, T in enumerate(tl):

	# Speed-up as a function of N (parameter: eps)
	plt.plot()
	for j, eps in enumerate(neps):

		colorVal = scalarMap.to_rgba(values[j])

		sa = [ [d['N'], d['spa']] for d in data if (d['T']==T and d['eps']==eps)]
		pltstyle([ s[0] for s in sa ], [ s[1] for s in sa ], color=colorVal, marker='o', linestyle = '-', label='$\eta$='+str(eps))

		sna = [ [d['N'], d['spna']] for d in data if (d['T']==T and d['eps']==eps)]
		pltstyle([ s[0] for s in sna ], [ s[1] for s in sna ], color=colorVal, marker='x', linestyle = ':')

	plt.ylim([1., 15])
	plt.title('T='+str(T))
	plt.legend()
	plt.savefig(args.ode_name+'/'+args.fn+'/speedup-T'+str(T)+'.pdf')
	plt.close()

	# Efficiency as a function of eps (parameter: N)
	plt.plot()
	for j, eps in enumerate(neps):

		colorVal = scalarMap.to_rgba(values[j])

		sa = [ [d['N'], d['ea']] for d in data if (d['T']==T and d['eps']==eps)]
		pltstyle([ s[0] for s in sa ], [ s[1] for s in sa ], color=colorVal, marker='o', linestyle = '-', label='$\eta$='+str(eps))

		sna = [ [d['N'], d['ena']] for d in data if (d['T']==T and d['eps']==eps)]
		pltstyle([ s[0] for s in sna ], [ s[1] for s in sna ], color=colorVal, marker='x', linestyle = ':')

	plt.ylim([0.01, 0.6])
	plt.title('T='+str(T))
	plt.legend()
	plt.savefig(args.ode_name+'/'+args.fn+'/efficiency-T'+str(T)+'.pdf')
	plt.close()


