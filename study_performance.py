import argparse
import numpy as np
import re
import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import matplotlib.cm as cmx

from ode import VDP, Brusselator, Oregonator

def read_results(parent_dir):
  """Extract results from folder of type parent_dir+'T_(\d+)-N_(\d+)-eps_(.*)' """
  _, dirnames, _ = next(os.walk(parent_dir))
  experiment_pattern = re.compile('T_(\d+)-N_(\d+)-eps_(.*)')
  Ts, Ns, epss, cas, cnas, cseqs, spas, spnas, eas, enas = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
  data = dict()
  for d in dirnames:
    m = experiment_pattern.match(d)
    if m is None:
      continue
    T = int(m.group(1))
    N = int(m.group(2))
    eps = float(m.group(3))
    
    Ts.append(T)
    Ns.append(N)
    epss.append(eps)

    # Read results
    da  = np.load(parent_dir+'/'+d+'/adaptive/cost.npz')
    dna = np.load(parent_dir+'/'+d+'/non-adaptive/cost.npz')

    ca = da['cost_f'].item()+da['cost_g'].item()
    cna = dna['cost_f'].item()+dna['cost_g'].item()
    cseq = da['cost_seq_fine'].item()
    # Speed-up
    spa = cseq / ca
    spna = cseq / cna
    # Efficiency
    ea = cseq / (ca * N)
    ena = cseq / (cna * N) 

    # Add to lists
    cas.append(ca)
    cnas.append(cna)
    cseqs.append(cseq)
    spas.append(spa)
    spnas.append(spna)
    eas.append(ea)
    enas.append(ena)

  data = {'T': Ts, 'N': Ns, 'eps': epss, 'cseq': cseqs, 'ca': cas, 'cna': cnas, 'spa': spas, 'spna': spnas, 'ea': eas, 'ena': enas}
  dataf = pd.DataFrame(data=data)
  return dataf

def plot_performance(parent_dir, dataf):
  """Make performance plots and save results in parent_dir"""
  Ts = dataf['T'].unique()
  epss = [1.e-6, 1.e-8]
  # Color map
  values = range(len(epss))
  cm = plt.get_cmap('cool') 
  cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
  scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
  pltstyle = plt.plot
  colorList = ['black', 'orange', 'blue', 'magenta']

  for i, T in enumerate(Ts):

    # Speed-up as a function of N (parameter: eps)
    plt.plot()
    for j, eps in enumerate(epss):
      colorVal = scalarMap.to_rgba(values[j])
      colorVal = colorList[j]
      sa = dataf[ (dataf['T']==T) &  (dataf['eps']==eps) ].sort_values(by=['N'])
      Ns = sa['N'].to_numpy()
      spas = sa['spa'].to_numpy()
      spnas = sa['spna'].to_numpy()
      pltstyle(Ns, spas, color=colorVal, marker='o', linestyle = '-', label='$\eta$='+str(eps)+' (Adaptive)')
      pltstyle(Ns, spnas, color=colorVal, marker='x', linestyle = ':', label='$\eta$='+str(eps)+' (Non Adaptive)')

    plt.ylim([0.1, 15])
    plt.xlabel(r'Nb Processors')
    plt.ylabel('Speed-up')
    plt.title('T='+str(T))
    plt.legend()
    plt.tight_layout()
    plt.savefig(parent_dir+'/speedup-T'+str(T)+'.pdf')
    plt.close()

    # Efficiency as a function of eps (parameter: N)
    plt.plot()
    for j, eps in enumerate(epss):
      colorVal = scalarMap.to_rgba(values[j])
      colorVal = colorList[j]
      sa = dataf[ (dataf['T']==T) &  (dataf['eps']==eps) ].sort_values(by=['N'])
      Ns = sa['N'].to_numpy()
      eas = sa['ea'].to_numpy()
      enas = sa['ena'].to_numpy()
      pltstyle(Ns, eas, color=colorVal, marker='o', linestyle = '-', label='$\eta$='+str(eps)+' (Adaptive)')
      pltstyle(Ns, enas, color=colorVal, marker='x', linestyle = ':', label='$\eta$='+str(eps)+' (Non Adaptive)')

    plt.ylim([0.01, 0.6])
    plt.xlabel(r'Nb Processors')
    plt.ylabel('Efficiency')
    plt.title('T='+str(T))
    plt.legend()
    plt.savefig(parent_dir+'/efficiency-T'+str(T)+'.pdf')
    plt.close()

if __name__ == "__main__":

  # Dictionary of available odes
  ode_dict = {'VDP': VDP, 'Brusselator': Brusselator, 'Oregonator': Oregonator}
  # Receive args
  parser = argparse.ArgumentParser()
  parser.add_argument('-ode', default='Brusselator', help='{VDP,Brusselator,Oregonator}')
  parser.add_argument('-fn', default='prod-cluster',help='foldername')
  args = parser.parse_args()
  if args.ode not in ode_dict:
    raise Exception('ODE type '+args.ode+' not supported')

  # Read data from parent_dir containing experiments and make performance plots
  parent_dir = args.ode + '/' + args.fn
  dataf = read_results(parent_dir)
  plot_performance(parent_dir, dataf)