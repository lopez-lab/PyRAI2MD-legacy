## Interface to BAGEL for PyRAIMD
## Jingbai Li Sep 29 2020

import os,subprocess,shutil,h5py
import numpy as np

from tools import Printcoord,NACpairs,whatistime,S2F,NACpairs

class BAGEL:
    ## This function run BAGEL single point calculation

    def __init__(self,variables_all,id=None):
        self.natom          = 0
        variables           = variables_all['bagel']
        self.keep_tmp       = variables['keep_tmp']
        self.verbose        = variables['verbose']
        self.ci             = variables['ci']
        self.project        = variables['bagel_project']
        self.workdir        = variables['bagel_workdir']
        self.bagel          = variables['bagel']
        self.nproc          = variables['bagel_nproc']
        self.mpi            = variables['mpi']
        self.blas      	    = variables['blas']
        self.lapack         = variables['lapack']
        self.boost          = variables['boost']
        self.mkl            = variables['mkl']
       	self.arch           = variables['arch']
        self.read_nac       = variables['read_nac']
        self.threads        = variables['omp_num_threads']
        self.use_mpi        = variables['use_mpi']
        self.use_hpc        = variables['use_hpc']

        if id != None:
            self.workdir    = '%s/tmp_BAGEL-%s' % (self.workdir,id)
        else:
            self.workdir    = '%s/tmp_BAGEL' % (self.workdir)

        ## set environment variables
        os.environ['BAGEL_PROJECT']       = self.project   # the input name is fixed!
        os.environ['BAGEL']               = self.bagel
        os.environ['BLAS']                = self.blas
        os.environ['LAPACK']              = self.lapack
        os.environ['BOOST']               = self.boost
        os.environ['BAGEL_WORKDIR']       = self.workdir
        os.environ['OMP_NUM_THREADS']     = self.threads
        os.environ['MKL_NUM_THREADS']     = self.threads
        os.environ['BAGEL_NUM_THREADS']   = self.threads
        os.environ['MV2_ENABLE_AFFINITY'] = '0'
        os.environ['LD_LIBRARY_PATH']     = '%s/lib:%s/lib:%s:%s:%s/lib:%s' % (self.mpi,self.bagel,self.blas,self.lapack,self.boost,os.environ['LD_LIBRARY_PATH'])
        os.environ['PATH']                = '%s/bin:%s' % (self.mpi,os.environ['PATH'])

    def _xyz2json(self,natom,coord):
        ## convert xyz from array to bagel format
        a2b=1.88973   # angstrom to bohr
        jxyz=''
        comma=','

        for n,line in enumerate(coord):
            e,x,y,z=line
            if n == natom-1:
                comma=''
            jxyz+="""{ "atom" : "%s", "xyz" : [%f, %f, %f]}%s\n""" % (e,float(x)*a2b,float(y)*a2b,float(z)*a2b,comma)

        return jxyz

    def _setup_hpc(self):
        if os.path.exists('%s.slurm' % (self.project)) == True:
            with open('%s.slurm' % (self.project)) as template:
                submission=template.read()
        else:
            submission=''

        submission+="""
export BAGEL_PROJECT=%s
export BAGEL=%s
export BLAS=%s
export LAPACK=%s
export BOOST=%s
export MPI=%s
export BAGEL_WORKDIR=%s
export OMP_NUM_THREADS=%s
export MKL_NUM_THREADS=%s
export BAGEL_NUM_THREADS=%s
export MV2_ENABLE_AFFINITY=0
export LD_LIBRARY_PATH=$MPI/lib:$BAGEL/lib:$BALS:$LAPACK:$BOOST/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH

source %s %s

cd $BAGEL_WORKDIR
""" % (self.project,\
               self.bagel,\
               self.blas,\
               self.lapack,\
               self.boost,\
               self.mpi,\
               self.workdir,\
               self.threads,\
               self.threads,\
               self.threads,\
               self.mkl,\
               self.arch)

        if self.use_mpi == 0:
            submission+='%s/bin/BAGEL %s/%s.json > %s/%s.log\n' % (self.bagel,self.workdir,self.project,self.workdir,self.project)
        else:
            submission+='mpirun -np %s %s/bin/BAGEL %s/%s.json > %s/%s.log\n' % (self.nproc,self.bagel,self.workdir,self.project,self.workdir,self.project)

        with open('%s/%s.sbatch' % (self.workdir,self.project),'w') as out:
            out.write(submission)

    def _setup_bagel(self,x):
        ## Read input template from current directory

        self.natom=len(x)
        with open('%s.bagel' % (self.project),'r') as template:
            input=template.read().splitlines()

        part1=''
        part2=''
        breaker=0
        for line in input:
            if '******' in line:
                breaker = 1
                continue
            if breaker == 0:
                part1+='%s\n' % line
            else:
                part2+='%s\n' % line

        coord=self._xyz2json(self.natom,x)

        si_input = part1+coord+part2

        if os.path.exists(self.workdir) == False:
            os.makedirs(self.workdir)

        with open('%s/%s.json' % (self.workdir,self.project),'w') as out:
            out.write(si_input)

        ## prepare .archive files
        if   os.path.exists('%s.archive' % (self.project)) == False:
            print('BAGEL: missing guess orbital .archive ')
            exit()

        if os.path.exists('%s/%s.archive' % (self.workdir,self.project)) == False:
            shutil.copy2('%s.archive' % (self.project),'%s/%s.archive' % (self.workdir,self.project))

    def _run_bagel(self):
        maindir=os.getcwd()
        os.chdir(self.workdir)
        if self.use_hpc == 1:
            subprocess.run('sbatch -W %s/%s.sbatch' % (self.workdir,self.project),shell=True)
        else:
            if self.use_mpi == 1:
                subprocess.run('source %s %s;mpirun -np %s %s/bin/BAGEL %s/%s.json > %s/%s.log' % (self.mkl,self.arch,self.nproc,self.bagel,self.workdir,self.project,self.workdir,self.project),shell=True)
            else:
                subprocess.run('source %s %s;%s/bin/BAGEL %s/%s.json > %s/%s.log' % (self.mkl,self.arch,self.bagel,self.workdir,self.project,self.workdir,self.project),shell=True)
        os.chdir(maindir)

    def _read_bagel(self):
        with open('%s/%s.log' % (self.workdir,self.project),'r') as out:
            log  = out.read().splitlines()
                
        energy   = np.loadtxt('%s/ENERGY.out' % (self.workdir))[0:self.ci]
        gradient = []
        for i in range(self.ci):


            with open('%s/FORCE_%s.out' % (self.workdir,i)) as force:
                g=force.read().splitlines()[1:self.natom+1]
                g=S2F(g)
                gradient.append(g)

        gradient = np.array(gradient)

        if self.read_nac == 1:
            coupling = []
            npair    = int(self.ci*(self.ci-1)/2)
            pairs    = NACpairs(self.ci).copy()
            for i in range(npair):
                pa,pb=pairs[i+1]
                with open('%s/NACME_%s_%s.out' % (self.workdir,pa-1,pb-1)) as nacme:
                    cp=nacme.read().splitlines()[1:self.natom+1]
                    cp=S2F(cp)
                    coupling.append(cp)
            nac  = np.array(coupling) 
        else:
            nac  = np.zeros([1,self.natom,3])

        civec    = np.zeros(0)
        movec    = np.zeros(0)

        return energy,gradient,nac,civec,movec

    def appendix(self,addons):
        ## fake function

        return self

    def evaluate(self,x):
        self._setup_bagel(x)
        if self.use_hpc == 1:
            self._setup_hpc()
        self._run_bagel()
        energy,gradient,nac,civec,movec=self._read_bagel()
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

