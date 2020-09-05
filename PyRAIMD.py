## The main function of PyRAIMD
## the first version  Jingbai Li Feb 16 2020
## the second version Jingbai Li May 14 2020
## the class version  Jingbai Li Jul 15 2020

import sys,os,time,datetime
import numpy as np
from tools import whatistime,howlong
from tools import Readcoord,Printcoord,Checkpoint,Readinitcond
from entrance import ReadInput,StartInfo
from methods import QM
from aimd import AIMD
from dynamixsampling import Sampling
from adaptive_sampling import AdaptiveSampling


class PYRAIMD:

    def __init__(self,input,version):

        input_dict=open(input,'r').read().split('&')
        self.variables_all=ReadInput(input_dict)
        input_info=StartInfo(self.variables_all)
        self.variables_all['version']                  = self._version_info(version,input_info)

    def _version_info(self,x,y):

        ## x: float
        ##    Version 
        ## y: str
        ##    Input information

        info="""
-------------------------------------------------------

                           *
                          /+\\
                         /+++\\
                        /+++++\\
                       /PyRAIMD\\
                      /+++++++++\\
                     *===========*

                      Python Rapid
                 Artificial Intelligence
              Ab Initio Molecular Dynamics

                   Developer@Jingbai Li
               Northeastern University, USA

                      version:   %s


  With contributions from(in alphabetic order):
    Andre Eberhard	 - Gaussian process regression
    Patrick Reiser	 - Neural networks

  Special acknowledgement to:
    Steven A. Lopez	 - Project directorship
    Pascal Friederich    - ML directorship

%s

""" % (x,y)
        return info

    def _machine_learning(self):
        jobtype  = self.variables_all['control']['jobtype']
        qm       = self.variables_all['control']['qm'] 

        model=QM(qm,self.variables_all,id=None)

        if   jobtype == 'train':
            model.train()
        elif jobtype == 'prediction':
            model.load()
            model.evaluate(None)  # None will use json file

    def _dynamics(self):
        title    = self.variables_all['control']['title']
        qm       = self.variables_all['control']['qm']
        md       = self.variables_all['md']
        initcond = md['initcond']
        nesmb    = md['nesmb']
        method   = md['method']
        format   = md['format']
        gl_seed  = md['gl_seed']
        temp     = md['temp']
        if initcond == 0:
            ## load initial condition from .xyz and .velo
            xyz,M=Readcoord(title)
            velo=np.loadtxt('%s.velo' % (title))
        else:
            ## use sampling method to generate intial condition
            trvm=Sampling(title,nesmb,gl_seed,temp,method,format)[-1]
            xyz,mass,velo=Readinitcond(trvm)
            ## save sampled geometry and velocity
            initxyz_info='%d\n%s\n%s' % (len(xyz),'%s sampled geom %s at %s K' % (method,nesmb,temp),Printcoord(xyz))
            initxyz=open('%s.xyz' % (title),'w')
            initxyz.write(initxyz_info)
            initxyz.close()
            initvelo=open('%s.velo' % (title),'w')
            np.savetxt(initvelo,velo,fmt='%30s%30s%30s')
            initvelo.close()
        method=QM(qm,self.variables_all,id=None)
        method.load()
        traj=AIMD(self.variables_all,QM=method,id=None,dir=None)
        traj.run(xyz,velo)

    def	_active_search(self):
        sampling=AdaptiveSampling(self.variables_all)
        sampling.search()

    def run(self):
        jobtype = self.variables_all['control']['jobtype']
        job_func={
        'md'         : self._dynamics,
        'adaptive'   : self._active_search,
        'train'      : self._machine_learning,
        'prediction' : self._machine_learning,
        ## 'search'  : self._machine_learning,  ## not implemented
        }

        job_func[jobtype]()


if __name__ == '__main__':

    version = 0.5
    usage="""
  --------------------------------------------------------------
                               *
                              /+\\
                             /+++\\
                            /+++++\\
                           /PyRAIMD\\
                          /+++++++++\\
                         *===========*

                          Python Rapid
                     Artificial Intelligence
                  Ab Initio Molecular Dynamics

                       Developer@Jingbai Li
                   Northeastern University, USA

                          version:   %s


  With contriutions from(in alphabetic order):
    Andre Eberhard	 - Gaussian process regression
    Patrick Reiser	 - Neural networks

  Special acknowledgement to:
    Steven A. Lopez	 - Project directorship
    Pascal Friederich    - ML directoriship

  Usage:
      python3 PyRAIMD.py input

""" % (version)

    if len(sys.argv) < 2:
        print(usage)
    else:
        pmd=PYRAIMD(sys.argv[1],version)
        pmd.run()
