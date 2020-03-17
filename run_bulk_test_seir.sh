#!/bin/bash

# for T in 100; do
# 	for N in 10 20 30 40 50 60 70 80 100; do
# 		for EPS in 1.0e-06 1.0e-08; do
for T in 100; do
	for N in 10; do
		for EPS in 1.0e-06; do
			qsub -v T=$T,N=$N,EPS=$EPS submit_run_seir.pbs
		done
	done
done
