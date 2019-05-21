#!/bin/bash
rsync -ravz --exclude='Brusselator' --exclude='Oregonator' --exclude='VDP' --exclude='.git' *.py *.pbs *.sh cluster:parareal
