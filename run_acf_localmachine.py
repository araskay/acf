#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:18:31 2018

@author: mkayvanrad
"""

import workflow, getopt, sys, subprocess, fileutils, os

def printhelp():
    print('Usage: run_acf_localmachine.py --subjects <subjects file> [--numpar <number of parallel jobs = 8> [acf.py additional arguments as listed below]')
    print('---------------------------------')
    p=subprocess.Popen(['acf.py','-h'])
    p.communicate()

ifile=''
numpar=8

# the getopt libraray somehow "guesses" the arguments- for example if given
# '--subject' it will automatically produce '--subjects'. This can cause problems
# later when arguments from sys.argv are passed to pipe.py. The following checks
# in advance to avoid such problems
pipe_args = sys.argv[1:]
for arg in pipe_args:
    if '--' in arg:
        if not arg in ['--subjects','--ndiscard', '--help','--numpar']:
            printhelp()
            sys.exit()

# parse command-line arguments
try:
    (opts,args) = getopt.getopt(sys.argv[1:],'h',['help','subjects=', 'ndiscard=','numpar='])
except getopt.GetoptError:
    printhelp()
    sys.exit()
for (opt,arg) in opts:
    if opt in ('-h', '--help'):
        printhelp()
        sys.exit()
    elif opt in ('--subjects'):
        ifile=arg
    elif opt in ('--numpar'):
        numpar=int(arg)

if ifile=='':
    printhelp()
    sys.exit()

base_command = 'acf.py'

count=0

subjects=workflow.getsubjects(ifile)

# get input file name to use for naming temporary files
(directory,filename)=os.path.split(ifile)

processes = []
proccount=0

for subj in subjects:
    for session in subj.sessions:
        for run in session.runs:
            count+=1
            proccount+=1
            
            command=[]
            
            outputfile='.temp_acf_job_'+filename+str(count)+'.o'
            errorfile='.temp_acf_job_'+filename+str(count)+'.e'

            f_o = open(outputfile, 'w')
            f_e = open(errorfile, 'w')
        
            command.append(base_command)
            #Just re-use the arguments given here
            pipe_args = sys.argv[1:]

            pipe_args[pipe_args.index('--subjects')+1] = fileutils.addniigzext(run.data.bold)
            pipe_args[pipe_args.index('--subjects')] = '--file'

            if '--numpar' in pipe_args:
                del pipe_args[pipe_args.index('--numpar')+1]
                del pipe_args[pipe_args.index('--numpar')]

            command = command+ pipe_args
            
            # now submit job
            print('Running',' '.join(command))
            p=subprocess.Popen(command,stdout=f_o,stderr=f_e)
            processes.append(p)

            if proccount==numpar:
                for p in processes:
                    p.wait()
                proccount=0
                processes=[]
                print('Total of',count,'jobs done')

for p in processes:
    p.wait()      

      


            
