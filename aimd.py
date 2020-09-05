## The Ab Inito Molecular Dynamics for PyQDynamics
## Jingbai Li Jun 9 2020

import time,datetime
import os
import numpy as np
from periodic_table import Element
from verlet import NoseHoover, VerletI, VerletII
from surfacehopping import FSSH
from tools import Printcoord,NACpairs
class AIMD:
    ## This class propagate nuclear position based on Velocity Verlet algorithm

    def __init__(self,variables_all,QM=None,id=None,dir=None):
        ## T     : list 
        ##         Atom list
        ## R     : np.array
        ##         Coordinates in angstrom
        ## M     : np.array
        ##         Masses in amu
        ## V     : np.array
        ##         Velocity in Bohr/au
        ## Vs    : np.array 
        ##         Thermostat array
        ## E     : np.array
        ##         Energy in Eh
        ## G     : np.array
        ##         Gradient in Eh/Bohr
        ## N     : np.array
        ##         Non-adiabatic coupling in 1/Bohr
        ## M     : np.array
        ##         Nuclear mass in atomic unit
        ## t     : int
        ##         Dynamics time step
        ## maxh  : int
        ##         Maximum number of hoping between states
        ## delt  : float
        ##         Probability integration time step in atomic unit
        ## A     : np.array
        ##         Previous state denesity matrix
        ## H     : np.array
        ##         Previous energy matrix
        ## D     : np.array
        ##         Previous non-adiabatic matrix
        ## At    : np.array
        ##         Current state denesity matrix
        ## Ht    : np.array
        ##         Current energy matrix
        ## Dt    : np.array
        ##         Current non-adiabatic matrix
        ## Ekin  : float
        ##         Current kinetic energy
        ## state : int
        ##         Current state number
        ## deco  : float
        ##         Decoherance energy
        ## 1 au  = 2.4188843265857 * 10**-2 fs
        ## 1 kb  = 3.16881 * 10**-6 Eh/K

        title          = variables_all['control']['title']

        self.fs_to_au=2.4188843265857*10**-2
        self.kb=3.16881*10**-6

        self.variables = variables_all
        self.version   = variables_all['version']
        self.maxerr_e  = variables_all['control']['maxenergy']
       	self.maxerr_g  = variables_all['control']['maxgradient']
       	self.maxerr_n  = variables_all['control']['maxnac']
        self.stop      = 0      ## stop aimd once error exceed maxerr
        self.traj      = variables_all['md'].copy()
        self.QM        = QM
        self.traj.update({
        'title'   : title,      ## name of calculation       
        'dir'     : None,       ## output directory
        'natom'   : 0,          ## number of atoms
        'A'       :[],          ## previous state denesity matrix
        'H'       :[],          ## previous energy matrix
        'D'       :[],          ## previous non-adiabatic matrix
        'At'      :[],          ## current state denesity matrix
        'Ht'      :[],          ## current energy matrix
        'Dt'      :[],          ## current non-adiabatic matrix
        'pciv'    : None,       ## previous ci vectors
        'pmov'    : None,       ## previous mo vectors
        'old'     : 0,          ## previous state number
        'state'   : 0,	        ## current state number
        'T'       :[],          ## atom list
        'R'       :[],          ## coordinates in angstrom
        'V'       :[],          ## velocity in Bohr/au
        'Ekin'    : 0,          ## kinetic energy in Eh
        'E'       :[],          ## potential energy in Eh
        'G'       :[],          ## gradient in Eh/Bohr
        'G0'      :[],          ## previous gradient
        'N'       :[],          ## non-adiabatic coupling in 1/Bohr
        'Vs'      :[],          ## thermostat array
        'iter'    : 0,          ## current iteration
        'hoped'   : 0,          ## surface hopping type
        'err_e'   : None,       ## error of energy in adaptive sampling
        'err_g'   : None,       ## error of gradient in adaptive sampling
        'err_n'   : None,       ## error of nac in adaptive sampling
        'MD_hist' :[],          ## md history
                         })

        self.traj['old']   = self.traj['root']
        self.traj['state'] = self.traj['root']

        if id != None:
            self.traj['title']  = '%s-%s' % (title,id)

        if dir != None:
            self.traj['dir']    = '%s/%s' % (os.getcwd(),self.traj['title'])
            if os.path.exists(self.traj['dir']) == False:
                os.makedirs(self.traj['dir'])

        if self.traj['substep'] == 0:
            self.traj['delt']   = 0.2 
            self.traj['substep']= int(self.traj['size']/self.traj['delt'])
        else:
            self.traj['delt']   = self.traj['size']/self.traj['substep']

    def _propagate(self,init):
        self.traj['R'] = VerletI(self.traj,init)
        ##print('verlet',time.time())
        xyz=self._write_coord(self.traj['T'],self.traj['R'])
        ##print('write_xyz',time.time())
        self._compute_properties(xyz)
        ##print('compute_egn',time.time())
        self.traj['V'] = VerletII(self.traj,init)
        ##print('verlet_2',time.time())
        self.traj['Ekin'] = np.sum(0.5*(self.traj['M']*self.traj['V']**2))

    def _compute_properties(self,xyz):
        addons={
        'pciv' : self.traj['pciv'],
        'pmov' : self.traj['pmov'],
        }
        qm = self.QM
        qm.appendix(addons)
        self.traj['G0'] = self.traj['G']
        results = qm.evaluate(xyz)
        self.traj['E']    = results['energy']
        self.traj['G']    = results['gradient']
        self.traj['N']    = results['nac']
        self.traj['pciv'] = results['civec'] 
        self.traj['pmov'] = results['movec']
        self.traj['err_e']= results['err_e']
        self.traj['err_g']= results['err_g']
        self.traj['err_n']= results['err_n']
        self.traj['MD_hist'].append([xyz,results['energy'].tolist(),results['gradient'].tolist(),results['nac'].tolist(),\
                                         results['err_e'].tolist(), results['err_g'].tolist(),   results['err_n'].tolist()]) # convert all to list

    def _thermostat(self,init):
        if self.traj['thermo'] == 1:
            V,Vs,Ekin = NoseHoover(self.traj,init=init)
            self.traj['V'] = V
            self.traj['Vs'] = Vs
            self.traj['Ekin'] = Ekin

    def _surfacehop(self,init):
        if self.traj['fssh'] == 1:
            self.traj['A'] = np.copy(self.traj['At'])
            self.traj['H'] = np.copy(self.traj['Ht'])
            self.traj['D'] = np.copy(self.traj['Dt'])
            At,Ht,Dt,V,hoped,old_state,state=FSSH(self.traj,init=init)
            self.traj['At'] = At
            self.traj['Ht'] = Ht
            self.traj['Dt'] = Dt
            self.traj['V']  = V
            self.traj['hoped'] = hoped
            self.traj['old'] = old_state
            self.traj['state'] = state

    def _read_coord(self,xyz):
        xyz = np.array(xyz)
        natom = len(xyz)
        T = xyz[:,0]
        R = xyz[:,1:].astype(float)
        M = np.array([Element(x).getMass()*1822.8852 for x in T]).reshape([-1,1])
        return T,R,M

    def _write_coord(self,T,R):
        xyz = []
        for n,i in enumerate(R):
            exyz = [T[n]] + i.tolist()
            xyz.append(exyz)
        return xyz

    def _heading(self):

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |          Nonadiabatic Molecular Dynamics          |
 |                                                   |
 *---------------------------------------------------*

""" % (self.version)

        return headline

    def _whatistime(self):
        return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    def _howlong(self,start,end):
        walltime=end-start
        walltime='%5d days %5d hours %5d minutes %5d seconds' % (int(walltime/86400),int((walltime%86400)/3600),int(((walltime%86400)%3600)/60),int(((walltime%86400)%3600)%60))
        return walltime

    def	_chkpoint(self):
        ## This function print current information
        ## This function append output to .log, .md.energies and .md.xyz

        Chk       = self.traj.copy() ## copy the dict in case I will change the data type for saving in the future
        title     = Chk['title']              ## title
        dir       = Chk['dir']                ## output directory
        temp      = Chk['temp']               ## temperature
        t         = Chk['size']               ## time step size
        ci        = Chk['ci']                 ## ci dimension
        old_state = Chk['old']                ## the previous state or the current state before surface hopping
        state     = Chk['state']              ## the current state or the new state after surface hopping
        iter      = Chk['iter']               ## the current iteration
        T         = Chk['T'].reshape([-1,1])  ## atom list
        R         = Chk['R']                  ## coordiantes
        V         = Chk['V']                  ## velocity
        Ekin      = Chk['Ekin']               ## kinetic energy
        E         = Chk['E']                  ## potential energy
        G         = Chk['G']                  ## gradient
        N         = Chk['N']                  ## non-adiabatic coupling
        At        = Chk['At']                 ## population (complex array)
        hoped     = Chk['hoped']              ## surface hopping detector
        natom     = len(T)                    ## number of atoms
        err_e     = Chk['err_e']              ## error of energy in adaptive sampling
        err_g     = Chk['err_g']              ## error of gradient in adaptive sampling
        err_n     = Chk['err_n']              ## error of nac in adaptive sampling
        verbose   = Chk['verbose']            ## print level

        ## prepare a comment line for xyz file
        cmmt='%s coord %d state %d' % (title,iter,old_state)

        ## prepare the surface hopping detection section according to Molcas output format
        if   hoped == 0:
            hop_info=' A surface hopping is not allowed\n  **\n At state: %3d\n' % (state)
        elif hoped == 1:
       	    hop_info=' A surface hopping event happened\n  **\n From state: %3d to state: %3d\n' % (old_state,state)
            cmmt+=' to %d CI' % (state)
        elif hoped == 2:
            hop_info=' A surface hopping is frustrated\n  **\n At state: %3d\n' % (state)

        ## prepare population and potential energy info
        pop=' '.join(['%28.16f' % (x) for x in np.real(np.diag(At))])
        pot=' '.join(['%28.16f' % (x) for x in E])

        ## prepare non-adiabatic coupling pairs
        pairs=NACpairs(ci)

        ## start to output
        log_info=' Iter: %8d  Ekin = %28.16f au T = %8.2f K dt = %10d CI: %3d\n Root chosen for geometry opt %3d\n' % (iter,Ekin,temp,t,ci,old_state)
        log_info+='\n Gnuplot: %s %s %28.16f\n  **\n  **\n  **\n%s\n' % (pop,pot,E[old_state-1],hop_info)

        if verbose >= 1:
            xyz=np.concatenate((T,R),axis=1)
            log_info+="""
  &coordinates in Angstrom
-------------------------------------------------------
%s-------------------------------------------------------
""" % (Printcoord(xyz))
            velo=np.concatenate((T,V),axis=1)
            log_info+="""
  &velocities in Bohr/au
-------------------------------------------------------
%s-------------------------------------------------------
""" % (Printcoord(velo))
            for n,g in enumerate(G):
                grad=np.concatenate((T,g),axis=1)
                log_info+="""
  &gradient %3d in Eh/Bohr
-------------------------------------------------------
%s-------------------------------------------------------
""" % (n+1,Printcoord(grad))
            for m,n in enumerate(N):
                nac=np.concatenate((T,n),axis=1)
       	        log_info+="""
  &non-adiabatic coupling %3d - %3d in 1/Bohr
-------------------------------------------------------
%s-------------------------------------------------------
""" % (pairs[m+1][0],pairs[m+1][1],Printcoord(nac))

        if err_e != None and err_g != None and err_n != None:
            log_info+="""
  &error iter %-10s
-------------------------------------------------------
  Energy   error:             %-10.4f
  Gradient error:             %-10.4f
  Nac      error:             %-10.4f
-------------------------------------------------------

""" % (iter,err_e,err_g,err_n)
       	    if err_e > self.maxerr_e or err_g > self.maxerr_g or err_n > self.maxerr_n:
                self.stop = 1

        if dir == None:
            logpath=os.getcwd()
        else:
            logpath=dir

        #print(log_info)
        mdlog=open('%s/%s.log' % (logpath,title),'a')
        mdlog.write(log_info)
        mdlog.close() 

        energy_info='%8.2f%28.16f%28.16f%28.16f%s\n' % (iter*t,E[old_state-1],Ekin,E[old_state-1]+Ekin,pot)
        mdenergy=open('%s/%s.md.energies' % (logpath,title),'a')
        mdenergy.write(energy_info)
        mdenergy.close()

        xyz_info='%d\n%s\n%s' % (natom,cmmt,Printcoord(np.concatenate((T,R),axis=1)))
        mdxyz=open('%s/%s.md.xyz' % (logpath,title),'a')
        mdxyz.write(xyz_info)
        mdxyz.close()

        if Chk['silent'] == 0:
            print(log_info)

    def run(self,xyz,velo):
        ## xyz  : list
        ##        Coordinates list of [atom x y z] in angstrom
        ## velo : np.array
        ##        Nuclear velocities in Bohr/au

        title    = self.traj['title']
        dir	 = self.traj['dir']
        warning  = ''

        if dir == None:
            logpath=os.getcwd()
        else:
            logpath=dir

        start=time.time()
        heading='Nonadiabatic Molecular Dynamics Start: %20s\n%s' % (self._whatistime(),self._heading())

        if self.traj['silent'] == 0:
            print(heading)

        mdlog=open('%s/%s.log' % (logpath,title),'w')
        mdlog.write(heading)
        mdlog.close()

        mdenergy=open('%s/%s.md.energies' % (logpath,title),'w')
        mdenergy.close()

        mdxyz=open('%s/%s.md.xyz' % (logpath,title),'w')
        mdxyz.close()

        natom = len(xyz)
        T,R,M = self._read_coord(xyz)

        self.traj['natom'] = natom
        self.traj['T'] = T
        self.traj['R'] = R
        self.traj['M'] = M
        self.traj['V'] = velo
        self._heading()
        for iter in range(self.traj['step']):
            self.traj['iter'] = iter+1
            ##print('start', time.time())
            self._propagate(iter+1)    # update E,G,N,R,V,Ekin
            ##print('propagate',time.time())
            self._thermostat(iter+1)   # update Ekin,V,Vs
            ##print('thermostat',time.time())
            self._surfacehop(iter+1)   # update A,H,D,V,state
            ##print('surfacehop',time.time())
            self._chkpoint()
            ##print('save',time.time())
            if self.stop == 1:
#                if len(self.traj['MD_hist']) > 1:
#                    self.traj['MD_hist'] = self.traj['MD_hist'][:-1] # revert one step back if trajectory has more than one step, since the large error
                warning='Errors are too large'
                break


        end=time.time()
        walltime=self._howlong(start,end)
        tailing='%s\nNonadiabatic Molecular Dynamics End: %20s Total: %20s\n' % (warning,self._whatistime(),walltime)

        if self.traj['silent'] == 0:
            print(tailing)

        mdlog=open('%s/%s.log' % (logpath,title),'a')
        mdlog.write(tailing)
        mdlog.close()

        return self.traj['MD_hist']


