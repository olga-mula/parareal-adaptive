
import argparse
import numpy as np

from ode import VDP, Brusselator, Oregonator

# Dictionary of available odes
ode_dict = {'VDP': VDP, 'Brusselator': Brusselator, 'Oregonator': Oregonator}

# Receive args
# ------------
parser = argparse.ArgumentParser()
parser.add_argument(
	'-ode', '--ode_name', default='Brusselator',
	help='{VDP,Brusselator,Oregonator}')
args = parser.parse_args()
if args.ode_name not in ode_dict:
	raise Exception('ODE type '+args.ode_name+' not supported')

# Read results
# ------------
tl = [100, 300, 500, 700, 900, 1100]
nl = [10, 20, 30, 40, 50, 60, 70, 80, 100]
epsl = [1.0e-06, 1.0e-07, 1.0e-08, 1.0e-09] # 1.e-10 too close to machine precision

d = list()

for T in tl:
	T_formatted = '{:d}'.format(int(T))
	for N in nl:
		N_formatted = '{:d}'.format(N)
		for eps in epsl:
			eps_formatted = '{:.1e}'.format(eps)

			foldername = args.ode_name + '/prod-cluster/T_'+ T_formatted + '-N_'+ N_formatted + '-eps_'+ eps_formatted + '/'

			da = np.load(foldername+'adaptive/cost.npz')
			dna = np.load(foldername+'non-adaptive/cost.npz')

			# Cost
			ca = da['cost_f'].item()+da['cost_g'].item()
			cna = dna['cost_f'].item()+dna['cost_g'].item()
			cseq = da['cost_seq_fine'].item()

			# Speed-up
			spa = cseq / ca
			spna = cseq / cna
			
			# Efficiency
			ea = cseq / (ca * N)
			ena = cseq / (cna * N) 

			d.append({'T': T, 'N': N, 'eps': eps, 'cseq': cseq, 'ca': ca, 'cna': cna, 'spa': spa, 'spna': spna, 'ea': ea, 'ena': ena})

np.savez(args.ode_name+'/prod-cluster/data.npz', data=d)