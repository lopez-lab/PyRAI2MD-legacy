## verlocity verlet for PyRAIMD
## Jingbai Li Feb 13 2020

import numpy as np

def NoseHoover(traj,init=1):
    ## This function calculate velocity scale factor by Nose Hoover thermo stat from t to t/2

    natom    = traj['natom']
    V        = traj['V']
    Ekin     = traj['Ekin']
    Vs       = traj['Vs']
    temp     = traj['temp']
    t        = traj['size']
    kb       = 3.16881*10**-6
    fs_to_au = 2.4188843265857*10**-2

    if init == 1:
        freq=1/(22/fs_to_au) ## 22 fs to au Hz
        Q1=3*natom*temp*kb/freq**2
        Q2=temp*kb/freq**2
        Vs=[Q1,Q2,0,0]
    else:
        Q1,Q2,V1,V2=Vs
        G2=(Q1*V1**2-temp*kb)/Q2
        V2+=G2*t/4
        V1*=np.exp(-V2*t/8)
        G1=(2*Ekin-3*natom*temp*kb)/Q1
        V1+=G1*t/4
        V1*=np.exp(-V2*t/8)
        s=np.exp(-V1*t/2)

        Ekin*=s**2

        V1*=np.exp(-V2*t/8)
        G1=(2*Ekin-3*natom*temp*kb)/Q1
        V1+=G1*t/4
        V1*=np.exp(-V2*t/8)
        G2=(Q1*V1**2-temp*kb)/Q2
        V2+=G2*t/4
        Vs[2]=V1
        Vs[3]=V2
        V*=s

    return V,Vs,Ekin

def VerletI(traj,init=1):
    ## This function update nuclear position
    ## R in Angstrom, 1 Bohr = 0.529177 Angstrom
    ## V in Bohr/au
    ## G in Eh/Bohr
    ## M in atomic unit

    R     = traj['R']
    V     = traj['V']
    G     = traj['G']
    M  	  = traj['M']
    t     = traj['size']
    state = traj['state']

    if init > 1:
        G = G[state-1]
        R+= (V*t-0.5*G/M*t**2)*0.529177    
    return R

def VerletII(traj,init=1):
    ## This function update velocity
    M     = traj['M']
    G     = traj['G']
    G0    = traj['G0']
    V     = traj['V']
    t     = traj['size']
    state = traj['state']

    if init > 1:
        G0= G0[state-1]
        G = G[state-1]
        V-= 0.5*(G0+G)/M*t
    return V

