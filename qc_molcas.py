## Interface to Molcas for PyRAIMD
## Jingbai Li Feb 24 2020
## fix bug in reading nac data Jingbai Li Jun 22 2020

import os,subprocess,shutil,h5py
import numpy as np

from tools import Printcoord,NACpairs,whatistime,S2F

class MOLCAS:
    ## This function run Molcas single point calculation

    def __init__(self,variables_all,id=None):
        self.natom          = 0
        variables           = variables_all['molcas']
        self.ci             = variables['ci']
        self.previous_civec = variables['previous_civec']
        self.previous_movec = variables['previous_movec']
        self.keep_tmp       = variables['keep_tmp']
        self.verbose        = variables['verbose']

        self.project        = variables['molcas_project']
        self.workdir        = variables['molcas_workdir']
        self.molcas         = variables['molcas']
        molcas_nproc        = variables['molcas_nproc']
        molcas_mem          = variables['molcas_mem']
        molcas_print        = variables['molcas_print']
        omp_num_threads     = variables['omp_num_threads']

        if id != None:
            self.workdir    = '%s-%s' % (self.workdir,id)

        ## set environment variables
        os.environ['MOLCAS_PROJECT']  = self.project   # the input name is fixed!
        os.environ['MOLCAS']          = self.molcas
        os.environ['MOLCAS_NPROC']    = molcas_nproc
        os.environ['MOLCAS_MEM']      = molcas_mem
        os.environ['MOLCAS_PRINT']    = molcas_print
        os.environ['MOLCAS_WORKDIR']  = self.workdir
        os.environ['OMP_NUM_THREADS'] = omp_num_threads

        ## Generate gradient and nac part
        alaska = ''
        i=int((self.ci-1)*self.ci/2)
        pairs=NACpairs(self.ci)
        for j in range(i):
            alaska+='&ALASKA\nNAC=%s %s\n' % (pairs[j+1][0],pairs[j+1][1])
        for j in range(self.ci):
            alaska+='&ALASKA\nROOT=%s\n' % (j+1)
#        alaska+='&CASPT2\nSHIFT\n0.1\nXMULTistate\nall\nmaxiter\n1000'

        ## Read input template from current directory
        with open('%s.molcas' % (self.project),'r') as template:
            input=template.read()
        si_input ='%s\n%s' % (input,alaska)

        if os.path.exists(self.workdir) == False:
            os.makedirs(self.workdir)

        with open('%s/%s.inp' % (self.workdir,self.project),'w') as out:
            out.write(si_input)

    def _setup_molcas(self,x):
        ## prepare .xyz .StrOrb files

        self.natom=len(x)
        with open('%s/%s.xyz' % (self.workdir,self.project),'w') as out:
            xyz='%s\n\n%s' % (self.natom,Printcoord(x))
            out.write(xyz)
        if   os.path.exists('%s.StrOrb' % (self.project)) == True and os.path.exists('%s/%s.RasOrb' % (self.workdir,self.project)) == False:
            shutil.copy2('%s.StrOrb' % (self.project),'%s/%s.StrOrb' % (self.workdir,self.project))
        elif os.path.exists('%s.StrOrb' % (self.project)) == True and os.path.exists('%s/%s.RasOrb' % (self.workdir,self.project)) == True:
            shutil.copy2('%s/%s.RasOrb' % (self.workdir,self.project),'%s/%s.StrOrb' % (self.workdir,self.project))
        else:
            print('Molcas: missing guess orbital .StrOrb ')
            exit()

    def _run_molcas(self):
        maindir=os.getcwd()
        os.chdir(self.workdir)
        subprocess.run('%s/bin/pymolcas -f %s/%s.inp -b 1' % (self.molcas,self.workdir,self.project),shell=True)
        subprocess.run('cp %s/%s/*.h5      %s' % (self.workdir,self.project,self.workdir),shell=True)  # sometimes, Molcas doesn't copy these orbital files
       	subprocess.run('cp %s/%s/*Orb*     %s' % (self.workdir,self.project,self.workdir),shell=True)
       	subprocess.run('cp %s/%s/*molden*  %s' % (self.workdir,self.project,self.workdir),shell=True)
        shutil.rmtree(self.project)
        os.chdir(maindir)

    def _read_molcas(self):
        with open('%s/%s.log' % (self.workdir,self.project),'r') as out:
            log  = out.read().splitlines()
        h5data   = h5py.File('%s/%s.rasscf.h5' % (self.workdir,self.project),'r')
        natom    = self.natom
        casscf   = []
        gradient = []
        nac      = []

        civec    = np.array(h5data['CI_VECTORS'][()]) 
       	movec    = np.array(h5data['MO_VECTORS'][()])
        inactive = 0
        active   = 0
        for i,line in enumerate(log):
            if   """::    RASSCF root number""" in line:
                e=float(line.split()[-1])
                casscf.append(e)
            elif """Molecular gradients """ in line:
                g=log[i+8:i+8+natom]
                g=S2F(g)
                gradient.append(g)
            elif """CI derivative coupling""" in line:
                n=log[i+8:i+8+natom]
                n=S2F(n)
                nac.append(n)
            elif """Inactive orbitals""" in line and inactive == 0:
                inactive=int(line.split()[-1])
            elif """Active orbitals""" in line and active == 0:
                active=int(line.split()[-1])

        energy   = np.array(casscf)
        gradient = np.array(gradient)
        nac      = np.array(nac)
        norb     = int(len(movec)**0.5)
        movec    = movec[inactive*norb:(inactive+active)*norb]
        movec    = np.array(movec).reshape([active,norb])

        return energy,gradient,nac,civec,movec

    def _phase_correction(self,x,nac,civec,movec):
        ## decide wheter to follow previous or search in data set

        if   self.previous_civec is None and self.previous_movec is None:
            ## ci and mo of this geometry will be used as reference and do not apply phase correction 
            refmo=movec
            refci=civec
        elif self.previous_civec is not None and self.previous_movec is not None:
       	    ## apply phase correction based on the ci and mo of previous geometry
            refmo=self.previous_movec
            refci=self.previous_civec

        mooverlap,mophase,mofactor=self._mo_sign_correction(refmo,movec)
        cioverlap,ciphase,cifactor=self._ci_sign_correction(refci,civec,mofactor)
        nac   = (nac.T*cifactor).T
        civec = ((civec*mofactor).T*ciphase).T
        movec = (movec.T*mophase).T
         

        return nac,civec,movec

    def _mo_sign_correction(self,ref,mo):
        ## correct mo sign by overlap. assume the mo order is the same

        mooverlap = np.sum(mo*ref,axis=1)
        mophase   = np.sign(mooverlap)
        mofactor  = np.sign([1,mooverlap[0]*mooverlap[1],1])

        return mooverlap,mophase,mofactor

    def _ci_sign_correction(self,ref,ci,mofactor):
        cioverlap = []
        nstate    = len(ref)
        ciphase   = np.ones(nstate)
        S         = np.array([np.sum(x*mofactor*ref,axis=1) for x in ci]) # sum of ci overlap matrix with mo factor
        aS        = np.abs(S)                             # absolute of sum of ci overlap matrix
        for i in range(nstate):                  # loop over nstate
            s=np.amax(aS[i])                     # find maximum overlap
            state=np.argmax(aS[i])               # find maximum overlap
            ciphase[i]=S[i][state]/aS[i][state]  # compute total sign
            aS[:,state]=0                        # clear the overlap of selected state
            cioverlap.append(S[i][state])

        cifactor  = []
        for i in range(nstate):
            for j in range(i+1,nstate):
                cifactor.append(ciphase[i]*ciphase[j])

        return cioverlap,ciphase,np.array(cifactor)


    def appendix(self,addons):
        ## appendix function to add ci and mo vectors
        self.previous_civec = addons['pciv']
        self.previous_movec = addons['pmov']
        return self

    def evaluate(self,x):
        self._setup_molcas(x)
        self._run_molcas()
        energy,gradient,nac,civec,movec=self._read_molcas()
        nac,civec,movec=self._phase_correction(x,nac,civec,movec)
        if self.keep_tmp == 0:
            shutil.rmtree(self.workdir)

        return {
                'energy'   : energy,
                'gradient' : gradient,
                'nac'	   : nac,
                'civec'    : civec,
                'movec'    : movec,
                'err_e'    : None,
                'err_g'    : None,
                'err_n'    : None,
                }

    def train(self):
        ## fake function

        return self

    def load(self):
        ## fake function

        return self

