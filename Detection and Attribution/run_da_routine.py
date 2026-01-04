#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## import modules
import argparse
import numpy as np
from load_fil_data import prefilt
from ROF_main import da
import os

INPATH = "/ampha/tenant/fafu/private/user/guanyl/lyg/data/DA-DAMIP/Tunder-yanmo/Asia/"
OUTPATH = "/ampha/tenant/fafu/private/user/guanyl/lyg/data/DA-DAMIP/Tunder-yanmo/out-asia-twoup"

class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)

PARSER = argparse.ArgumentParser(
    description='Run DA method for the entire period 1982-2020',
    usage='use "python %(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter)
PARSER.add_argument("var", help="Variable name")
PARSER.add_argument("-s", "--sigma", help="Length of filter window", type=int, default=5)
PARSER.add_argument("-b", "--base_per", help="Length of base period", type=int, default=5)
PARSER.add_argument('-fil', "--Ctype", help="Type of filter: C0 or C1 (default: C0)", choices=['C0', 'C1'], default='C0')
PARSER.add_argument('-bg', "--background", help="Type of background climate: stationary or non-stationary (default: stationary)", choices=['stat', 'trans'], default='stat')
PARSER.add_argument('-m', "--member", help="Member simulation in Obs: 1, 2 or 3 (default: 1)", default=1, type=int)
PARSER.add_argument('-r', "--reg", help="Type of regression: OLS or TLS (default: OLS)", choices=['OLS', 'TLS'], default='OLS')
PARSER.add_argument('-cons', "--Cons_test", choices=['OLS_AT99', 'OLS_Corr', 'AS03', 'MC'], default='OLS_Corr',
                    help="Cons_test: the null-distribution used in the Residual Consistency Check (p-value calculation)\n"
                         "In OLS, this may be:\n"
                         "\t OLS_AT99: the formula provided by Allen & Tett (1999), \n"
                         "\t OLS_Corr: the formula provided by Ribes et al. (2012), default.\n"
                         "In TLS, this may be:   \n"
                         "\t AS03: the null-distribution parametric formula provided by Allen & Stott (2003),\n"
                         "\t MC: Null-distribution evaluated via Monte-Carlo Simulations (Ribes et al., 2012)")
PARSER.add_argument('-f', "--Formule_IC_TLS", choices=['AS03', 'ODP'], default='ODP',
                    help="Formule_IC_TLS: the formula used for computing confidence intervals in TLS\n"
                         "\t AS03: the formula provided by Allen & Stott (2003)\n"
                         "\t ODP: the formula implemented in Optimal Detection Package (in Nov 2011), default.")
PARSER.add_argument('-sam', "--sample", help="Extracting Z into two samples Z1 and Z2 using regular, random or segment (default: regular)", choices=['regular', 'random', 'segment'], default='random')         
ARGS = PARSER.parse_args()

####
print("DA ANALYSIS FOR " + ARGS.var + " USING " + ARGS.reg)

####
if ARGS.reg == 'TLS' and (ARGS.Cons_test == 'OLS_AT99' or ARGS.Cons_test == 'OLS_Corr'):
    ARGS.Cons_test = 'AS03'
    print("OBS: TLS requires Cons_test AS03 or MC, set to AS03 (default)")

os.makedirs(OUTPATH, exist_ok=True)

if ARGS.member != 1:
    print("Warning: OBS has single member, setting member to 1")
    ARGS.member = 1
    
####
## Set parameters
START_YEAR_FORCING = 1982
END_YEAR_FORCING = 2020
VARNAME = ARGS.var

####
## Preprocessing
YEAR, OBS, FP, NX, CNTL = prefilt(
    ARGS.var, INPATH, ARGS.Ctype, ARGS.sigma, ARGS.base_per,
    START_YEAR_FORCING, ARGS.background, ARGS.member)

print("Debug - OBS values:", OBS)
print("Debug - FP values:", FP)
print("Debug - NX values:", NX)
print("Debug - CNTL mean:", np.mean(CNTL, axis=0))

# Debug prints
print("Debug - Shapes before DA:")
print("YEAR shape:", YEAR.shape)
print("OBS shape:", OBS.shape)
print("FP shape:", FP.shape)
print("NX shape:", NX.shape)
print("CNTL shape:", CNTL.shape)

if ARGS.background == 'trans':
    BETA = da(OBS, FP, NX, CNTL.T, ARGS.reg, ARGS.Cons_test, ARGS.Formule_IC_TLS, ARGS.sample)
else:
    BETA = da(OBS, FP, NX, CNTL.T, ARGS.reg, ARGS.Cons_test, ARGS.Formule_IC_TLS, ARGS.sample)

# Debug print for BETA
print("Debug - BETA shape:", BETA.shape)
print("Debug - BETA values:", BETA) 

# 打印结果
print("DA Analysis Results for 1982-2020:")
print("ANT beta_hat:", BETA[1, 0])
print("ANT beta_hat_inf (5%):", BETA[0, 0])
print("ANT beta_hat_sup (95%):", BETA[2, 0])
print("NAT beta_hat:", BETA[1, 1])
print("NAT beta_hat_inf (5%):", BETA[0, 1])
print("NAT beta_hat_sup (95%):", BETA[2, 1])
print("Consistency test p-value:", BETA[3, 0])

np.savez(os.path.join(OUTPATH, f"{ARGS.reg}_{ARGS.Ctype}_{VARNAME}_{ARGS.background}_s_{ARGS.sigma}_b_{ARGS.base_per}_mem_{ARGS.member}.npz"),
         base_per=ARGS.base_per, mem=ARGS.member, Ctype=ARGS.Ctype, beta=BETA, Year=YEAR, var=VARNAME,
         Cons_test=ARGS.Cons_test, sigma=ARGS.sigma, reg=ARGS.reg, bg=ARGS.background)

print("DA ANALYSIS DONE")
