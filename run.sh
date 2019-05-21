#!/bin/bash

for T in 100 300 500 700 900 1100; do
	for N in 10 20 30 40 50 60 70 80 100; do
		for EPS in 1.0e-06 1.0e-07 1.0e-08 1.0e-09 1.0e-10; do
			qsub -v T=$T,N=$N,EPS=$EPS script.pbs
		done
	done
done
