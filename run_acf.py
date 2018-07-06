#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:18:31 2018

@author: mkayvanrad
"""

import workflow, getopt, sys, subprocess, fileutils

def printhelp():
    print('Usage: run_acf.py -s <subjects file> --ndiscard <n>')
    print('RUN FROM THE DIRECTORY WHERE YOU WANT TO HAVE THE CSV FIlES SAVED')

subjects_file=''
n_discard='0'
# parse command-line arguments
try:
    (opts,args) = getopt.getopt(sys.argv[1:],'hs:',['subjects=', 'help','ndiscard='])
except getopt.GetoptError:
    sys.exit()
for (opt,arg) in opts:
    if opt in ('-s','--subjects'):
        subjects_file=arg
    elif opt in ('--ndiscard'):
        n_discard=arg
    elif opt in ('-h','--help'):
        printhelp()
        
if subjects_file=='':
    printhelp()
    sys.exit()
    
subjects=workflow.getsubjects(subjects_file)

for subj in subjects:
    for session in subj.sessions:
        for run in session.runs:
            p=subprocess.Popen(['sbatch','run_acf.sh',fileutils.addniigzext(run.data.bold), n_discard])
            p.communicate()
            
