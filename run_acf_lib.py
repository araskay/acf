import getopt
import subprocess


def printhelp():
    print('Usage: run_acf.py --subjects <subjects file> [acf.py additional arguments as listed below]')
    p=subprocess.Popen(['acf.py','-h'])
    p.communicate()
    print('---------------------------------')
    print('Additional job scheduler options:')
    print('--mem <amount in GB = 16>')


def argsvalid(args):
    for arg in args:
        if '--' in arg:
            if not arg in ['--subjects','--ndiscard', '--help', '--mem','--fit','--iqrcoef', '--anomalythresh']:
                return(False)
            else:
                return(True)

class Arguments:
    def __init__(self):
        self.help=False #flag indicating help
        self.error=False #flag indicating error in the input arguments
        self.mem='16'
        self.ifile=''

def parseargs(args):
    out=Arguments()
    # parse command-line arguments
    try:
        (opts,_) = getopt.getopt(args,'h',['help','subjects=', 'ndiscard=','mem=','fit','iqrcoef=','anomalythresh='])
    except getopt.GetoptError:
        out.error=True
        return(out)
    for (opt,arg) in opts:
        if opt in ('-h', '--help'):
            out.help=True
        elif opt in ('--subjects'):
            out.ifile=arg
        elif opt in ('--mem'):
            out.mem=arg
    return(out)

def initialize():
    # initialize acfFWHM.csv and write the header
    csv_fwhm = open('acfFWHM.csv','w')
    csv_fwhm.write('sessionID,'+
        'maxFWHMx,minFWHMx,meanFWHMx,medFWHMx,q1FWHMx,q3FWHMx,stdFWHMx,'+
        'maxFWHMy,minFWHMy,meanFWHMy,medFWHMy,q1FWHMy,q3FWHMy,stdFWHMy,'+
        'fracAnomaliesx,fracAnomaliesy,'+
        'meanAnomalyPerVolx,stdAnomalyPerVolx,'+
        'meanAnomalyPerVoly,stdAnomalyPerVoly,'+
        'medAnomalyPerVolx,q1AnomalyPerVolx,q3AnomalyPerVolx,'+
        'medAnomalyPerVoly,q1AnomalyPerVoly,q3AnomalyPerVoly,'+
        'numAnomaly_x,numAnomaly_y'+
        '\n')
    csv_fwhm.close()

    # initialize other csv files
    errorfile = open('error.txt', 'w')
    
    csv_maxFWHMx = open('maxFWHMx.csv', 'w')
    csv_minFWHMx = open('minFWHMx.csv', 'w')
    csv_medFWHMx = open('medFWHMx.csv', 'w')
    csv_q1FWHMx = open('q1FWHMx.csv', 'w')
    csv_q3FWHMx = open('q3FWHMx.csv', 'w')
    csv_meanFWHMx = open('meanFWHMx.csv', 'w')
    csv_stdFWHMx = open('stdFWHMx.csv', 'w')

    csv_maxFWHMy = open('maxFWHMy.csv', 'w')
    csv_minFWHMy = open('minFWHMy.csv', 'w')
    csv_medFWHMy = open('medFWHMy.csv', 'w')
    csv_q1FWHMy = open('q1FWHMy.csv', 'w')
    csv_q3FWHMy = open('q3FWHMy.csv', 'w')
    csv_meanFWHMy = open('meanFWHMy.csv', 'w')
    csv_stdFWHMy = open('stdFWHMy.csv', 'w')

    csv_teq = open('teq.csv','w')

    csv_maxFWHMx.close()
    csv_minFWHMx.close()
    csv_medFWHMx.close()
    csv_q1FWHMx.close()
    csv_q3FWHMx.close()
    csv_meanFWHMx.close()
    csv_stdFWHMx.close()

    csv_maxFWHMy.close()
    csv_minFWHMy.close()
    csv_medFWHMy.close()
    csv_q1FWHMy.close()
    csv_q3FWHMy.close()
    csv_meanFWHMy.close()
    csv_stdFWHMy.close()

    errorfile.close()

    csv_teq.close()
