#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:18:31 2018

@author: mkayvanrad
"""

import workflow, getopt, sys, subprocess, fileutils, os

def printhelp():
    print('Usage: run_acf.py --subjects <subjects file> [acf.py additional arguments as listed below]')
    p=subprocess.Popen(['acf.py','-h'])
    p.communicate()
    print('---------------------------------')
    print('Additional job scheduler options:')
    print('--mem <amount in GB = 16>')

ifile=''

# the getopt libraray somehow "guesses" the arguments- for example if given
# '--subject' it will automatically produce '--subjects'. This can cause problems
# later when arguments from sys.argv are passed to pipe.py. The following checks
# in advance to avoid such problems
pipe_args = sys.argv[1:]
for arg in pipe_args:
    if '--' in arg:
        if not arg in ['--subjects','--ndiscard', '--help', '--mem']:
            printhelp()
            sys.exit()

mem='16'

# parse command-line arguments
try:
    (opts,args) = getopt.getopt(sys.argv[1:],'h',['help','subjects=', 'ndiscard=','mem='])
except getopt.GetoptError:
    printhelp()
    sys.exit()
for (opt,arg) in opts:
    if opt in ('-h', '--help'):
        printhelp()
        sys.exit()
    elif opt in ('--subjects'):
        ifile=arg

    elif opt in ('--mem'):
        mem=arg

if ifile=='':
    printhelp()
    sys.exit()

base_command = 'acf.py'

count=0

subjects=workflow.getsubjects(ifile)

# get input file name to use for naming temporary files
(directory,filename)=os.path.split(ifile)

for subj in subjects:
    for session in subj.sessions:
        for run in session.runs:
            count+=1
            qbatch_fname = '.temp_acf_job_'+filename+str(count)+'.sh'
            qbatch_file = open(qbatch_fname, 'w')
        
            # write the header stuff
            qbatch_file.write('#!/bin/bash\n\n')
            qbatch_file.write('#SBATCH -c 1\n')
            qbatch_file.write('#SBATCH --mem='+mem+'g\n')
            qbatch_file.write('#SBATCH -t 72:0:0\n')
            qbatch_file.write('#SBATCH -o .temp_acf_job_'+filename+str(count)+'.o'+'\n')
            qbatch_file.write('#SBATCH -e .temp_acf_job_'+filename+str(count)+'.e'+'\n\n')
        
            qbatch_file.write('module load anaconda/3.5.3\n')
            #qbatch_file.write('module load afni\n')
            #qbatch_file.write('module load fsl\n')
            #qbatch_file.write('module load freesurfer\n')
            qbatch_file.write('module load anaconda/3.5.3\n\n')
                              
            qbatch_file.write(base_command + ' ')
            #Just re-use the arguments given here
            pipe_args = sys.argv[1:]
            pipe_args[pipe_args.index('--subjects')+1] = fileutils.addniigzext(run.data.bold)
            pipe_args[pipe_args.index('--subjects')] = '--file'
            del pipe_args[pipe_args.index('--mem')+1]
            del pipe_args[pipe_args.index('--mem')]
            command_str  = ' '.join(pipe_args)
            qbatch_file.write(command_str)
            qbatch_file.write('\n')
            
            qbatch_file.close()
            
            # now submit job
            p=subprocess.Popen(['sbatch',qbatch_fname])
            p.communicate()            


            
