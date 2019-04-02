import unittest
import subprocess
import os,shutil
import pandas as pd

class TestACF(unittest.TestCase):
    def setUp(self):
        self.mri_file='/home/mkayvanrad/scratch/fbirnqa/detrend/WEU01_PHA_FBN1391_0040_3dDetrend.nii.gz'
        self.csv_files = ['maxFWHMx.csv','minFWHMx.csv','medFWHMx.csv','q1FWHMx.csv','q3FWHMx.csv','meanFWHMx.csv','stdFWHMx.csv',
                          'maxFWHMy.csv','minFWHMy.csv','medFWHMy.csv','q1FWHMy.csv','q3FWHMy.csv','meanFWHMy.csv','stdFWHMy.csv',
                          'teq.csv','acfFWHM.csv']

        # clear files from previous runs
        for f in self.csv_files:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists('error.txt'):
            os.remove('error.txt')
        if os.path.exists('acf'):
            shutil.rmtree('acf')
        if os.path.exists('fwhm'):
            shutil.rmtree('fwhm')

        # run acf.py
        p=subprocess.Popen(['python','acf.py','--file',self.mri_file])
        p.communicate()

    # the following test is redundant
    '''test if the csv files are created on the disk'''
    '''
    def test_csv_create(self):
        
        for f in self.csv_files:
            self.assertTrue(os.path.isfile(f),f+' should exists.')
    '''

    ''' test if the csv files are properly populated.'''
    def test_csv_write(self):
        for f in self.csv_files:
            df = pd.read_csv(f,header=None)
            self.assertEqual(len(df),1,'Length of '+f+' should be 1.')


    def tearDown(self):
        for f in self.csv_files:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists('error.txt'):
            os.remove('error.txt')
        if os.path.exists('acf'):
            shutil.rmtree('acf')
        if os.path.exists('fwhm'):
            shutil.rmtree('fwhm')

if __name__ == '__main__':
    unittest.main()