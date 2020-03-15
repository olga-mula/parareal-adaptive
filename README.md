Description
===========

This folder contains the sources of the paper

  "An adaptive parareal algorithm"

by Y. Maday and O. Mula. [Arxiv link](https://arxiv.org/pdf/1909.08333.pdf)

Required software
=================
Python >= 3.7

Required modules: scipy, numpy, matplotlib, os, sys, time, pickle, argparse, pandas

Running the adaptive and classical parareal algorithm
=====================================================
You can run the adaptive and classical parareal algorithm by calling

```
python3 run_parareal.py -ode odename -T 200 - N 10 -eps 1.e-8 -eps_g 0.1 -fn foldername
```

The parameters are the following:
- ode: type of ODE. Supported keys={Brusselator, VDP}
- T: final time
- N: Number of processors
- eps: final target accuracy
- eps_g: accuracy of the coarse solver
- fn: data and plots about the results of the run are stored in odename/fn

All parameters are optional and the default values are:

- ode: Brusselator
- T: 10
- N: 10
- eps: $`10^{-6}`$
- eps_g: $`10^{-1}`$
- fn: default

How to reproduce the results of the paper
=========================================
You can run the following command in a cluster using PBS for job scheduling

```
./run_bulk_test_brusselator.sh
./run_bulk_test_vdp.sh
```

This is an intensive operation that calls multiple times the file run_parareal.py for different parameters T, N, eps
and stores each run in a folder, e.g., Brusselator/T_10-N_20-eps_1.0e-08.

Once this is done, you can build the performance plots of the paper by calling

python3 study_performance.py -ode odename -fn foldername

How to test other ODEs
======================
- Add a class `MY_NEW_ODE` in ode.py following the already existing examples.
- In run_parareal.py, import the class `MY_NEW_ODE`, and add a new keyword in argparse.
- In parareal_factory.py, import the class `MY_NEW_ODE`.

Licence
=======
Licensing information can be found in the accompanying file [COPYING.md](COPYING.md).
Copyright (c) 2019, Olga Mula (Paris Dauphine University).
