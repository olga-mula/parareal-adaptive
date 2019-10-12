Description
===========

This folder contains the sources of the paper

  "An adaptive parareal algorithm"

by Y. Maday and O. Mula. [Arxiv link](https://arxiv.org/pdf/1909.08333.pdf)

Required software
=================
Python >= 3.7

Required modules: scipy, numpy, matplotlib, multiprocessing, cvxopt, itertools, os, sys, time, jsonpickle, pickle, argparse

Running the code
=================
The main file is test.py. To reproduce the results of the paper, run the following command in a cluster

  ./run.sh

This is a script command which will compute the dynamics of the Brusselator system for different parameters and using the adaptive parareal algorithm. The parameters are:
- final time T
- number of processors N
- final target accuracy EPS

For each parameter set, the script calls test.py, which is computes the dynamics and the relevant outputs. To perform one single run, you can thus run the command

python3 test.py -T 200 - N 10 -eps 1.e-8 
 
Other optional parameters can be called. Run

python3 test.py -h

for more information.

Results will be stored in the folder Brusselator.

Copyright
=========
Copyright (c) 2019, Olga Mula (Paris Dauphine University).
