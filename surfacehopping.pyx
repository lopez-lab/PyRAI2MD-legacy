## Surface hopping for PyRAIMD
## Cython module
## Jingbai Li Sept 1 2020 

import sys
import numpy as np
cimport numpy as np

from tools import NACpairs

cdef avoid_singularity(float v_i,float v_j,int i,int j):
    ## This fuction avoid singularity of v_i-v_j for i < j 
    ## i < j assumes v_i <= v_j, thus assumes the sign of v_i-v_j is -1

    cdef float cutoff=1e-16
    cdef float sign, diff

    if i < j:
        sign=-1.0
    else:
        sign=1.0

    if   v_i == v_j:
        diff=sign*cutoff
    elif v_i != v_j and np.abs(v_i-v_j) < cutoff:
        diff=sign*cutoff
    elif v_i != v_j and np.abs(v_i-v_j) >= cutoff:
        #diff=v_i-v_j
        diff=sign*(v_i-v_j) # force v_i < v_j
    return diff

cdef Reflect(np.ndarray V,np.ndarray N,int reflect):
    ## This function refects velocity when frustrated hopping happens

    if   reflect == 1:
        V=-V
    elif reflect == 2:
        V-=2*np.sum(V*N)/np.sum(N*N)*N

    return V

cdef Adjust(float Ea, float Eb,np.ndarray V,np.ndarray M,np.ndarray N,int adjust,int reflect):
    ## This function adjust velocity when surface hopping detected
    ## This function call Reflect if frustrated hopping happens

    cdef float Ekin=np.sum(0.5*M*V**2)
    cdef frustrated=0
    cdef float dT,a,b,f

    if   adjust == 0:
        dT=Ea-Eb+Ekin
        if dT >= 0:
            f=1.0
        else:
            V=Reflect(V,N,reflect)
            frustrated=1

    elif adjust == 1:
        dT=Ea-Eb+Ekin
        if dT >= 0:
            f=(dT/Ekin)**0.5
            V=f*V
        else:
            V=Reflect(V,N,reflect)
            frustrated=1

    elif adjust == 2:
        a=np.sum(N*N/M)
        b=np.sum(V*N)
        dT=Ea-Eb
        dT=4*a*dT+b**2
        if dT >= 0:
           if b < 0:
               f=(b+dT**0.5)/(2*a)
           else:
               f=(b-dT**0.5)/(2*a)
           V-=f*N/M
        else:
            V=Reflect(V,N,reflect)
            frustrated=1

    return V,frustrated

cdef dPdt(np.ndarray A, np.ndarray H, np.ndarray D):
    ## This function calculate the gradient of state population
    ## The algorithm is based on Tully's method.John C. Tully, J. Chem. Phys. 93, 1061 (1990)

    ## State density A
    ## Hamiltonian H
    ## Non-adiabatic coupling D, D(i,j) = velo * nac(i,j)
    
    cdef int ci=len(A)
    cdef int k,j,l
    cdef np.ndarray dA=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray b=np.zeros((ci,ci))

    for k in range(ci):
        for j in range(ci):
            b[k,j]=2*np.imag(np.conj(A[k,j])*H[k,j])-2*np.real(np.conj(A[k,j])*D[k,j])
            for l in range(ci):
                dA[k,j]+=A[l,j]*(-1j*H[k,l]-D[k,l])-A[k,l]*(-1j*H[l,j]-D[l,j])

    return dA,b

cpdef FSSH(dict traj,int init=1):
    ## This function integrate the hopping posibility during a time step
    ## This function call dPdt to compute gradient of state population

    cdef np.ndarray A         = traj['A']
    cdef np.ndarray H         = traj['H']
    cdef np.ndarray D         = traj['D']
    cdef np.ndarray N         = traj['N']
    cdef int        substep   = traj['substep']
    cdef float      delt      = traj['delt']
    cdef int        ci        = traj['ci']
    cdef int        state     = traj['state']
    cdef int        maxhop    = traj['maxh']
    cdef str        usedeco   = traj['deco']
    cdef int        adjust    = traj['adjust']
    cdef int        reflect   = traj['reflect']
    cdef int        verbose   = traj['verbose']
    cdef int        old_state = traj['state']
    cdef int        new_state = traj['state']
    cdef np.ndarray V         = traj['V']
    cdef np.ndarray M         = traj['M']
    cdef np.ndarray E         = traj['E']
    cdef float      Ekin      = traj['Ekin']

    cdef np.ndarray At=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray Ht=np.diag(E).astype(complex)
    cdef np.ndarray Dt=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray dAdt=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray dHdt=np.zeros((ci,ci),dtype=complex)
    cdef np.ndarray dDdt=np.zeros((ci,ci),dtype=complex)

    cdef int n, i, j, k, p, stop, hoped, nhop, event, pairs
    cdef float deco, z, gsum, Asum, Amm
    cdef np.ndarray Vt, g, tau, b, NAC
    cdef dict pairs_dict

    Vt=V
    hoped=0
    n=0
    stop=0
    for i in range(ci):
        for j in range(i+1,ci):
            n+=1
            Dt[i,j]=np.sum(V*N[n-1])/avoid_singularity(E[i],E[j],i,j)
            Dt[j,i]=-Dt[i,j]

    if init == 1:
        At[state-1,state-1]=1
    else:
        dHdt=(Ht-H)/substep
        dDdt=(Dt-D)/substep
        g=np.zeros(ci)
        nhop=0
        
        if verbose == 3:
            print('One step')
            print(dPdt(A,H,D)[0]*delt*substep)
            print('Integral')

        for i in range(substep):
            event=0
            frustrated=0
            dAdt,b=dPdt(A,H,D)
            dAdt*=delt
            A+=dAdt
            for p in range(ci):
                if np.real(A[p,p])>1 or np.real(A[p,p])<0:
                    A-=dAdt  # revert A
                    stop=1   # stop if population exceed 1 or less than 0            

            if stop == 1:
                break

            H+=dHdt
            D+=dDdt

            for j in range(ci):
                g[j]=+np.amax([0,b[j,state-1]*delt/np.real(A[state-1,state-1])])

            z=np.random.uniform(0,1)

            gsum=0
            for j in range(ci):
                gsum+=g[j]
                nhop=np.abs(j+1-state)
                if gsum > z and nhop <= maxhop:
                    new_state=j+1
                    nhop=np.abs(j+1-state)
                    event=1
                    break

            if verbose >= 2:
                print('\nSubIter: %5d' % (i+1))
                print('NAC')
                print(Dt)
                print('A')
                print(A)
                print('Probabality')
                print(' '.join(['%12.8f' % (x) for x in g]))
                print('Population')
                print(' '.join(['%12.8f' % (np.real(x)) for x in np.diag(A)]))
                print('Random: %s' % (z))
                print('old state/new state: %s / %s' % (state, new_state))

            ## detect frustrated hopping and adjust velocity
            if event == 1:
                pairs_dict=NACpairs(ci)
                pairs=pairs_dict[str([state,new_state])]
                NAC=N[pairs-1] # pick up non-adiabatic coupling between state and new_state from the full array

                V,frustrated=Adjust(E[state-1],E[new_state-1],V,M,NAC,adjust,reflect)
                if frustrated == 0:
                    state=new_state

            ## decoherance of the propagation 
            if usedeco != 'OFF':
                deco=float(usedeco)
                tau=np.zeros(ci)

                ## matrix tau
                for k in range(ci):
                    if k != state-1:
                        tau[k]=np.abs(1/avoid_singularity(np.real(H[state-1,state-1]),np.real(H[k,k]),state-1,k))*(1+deco/Ekin) 

                ## update diagonal of A except for current state
                for k in range(ci):
                    for j in range(ci):
                        if k != state-1 and j != state-1:
                            A[k,j]*=np.exp(-delt/tau[k])*np.exp(-delt/tau[j])

                ## update diagonal of A for current state
                Asum=0.0
                for k in range(ci):
                    if k != state-1:
                        Asum+=np.real(A[k,k])
                Amm=np.real(A[state-1,state-1])
                A[state-1,state-1]=1-Asum

                ## update off-diagonal of A
                for k in range(ci):
                    for j in range(ci):
                        if   k == state-1 and j != state-1:
                            A[k,j]*=np.exp(-delt/tau[j])*(A[state-1,state-1]/Amm)**0.5
                        elif k != state-1 and j == state-1:
                            A[k,j]*=np.exp(-delt/tau[k])*(A[state-1,state-1]/Amm)**0.5

        ## final decision on velocity
        if state == old_state:   # not hoped
            Vt=V                 # revert scaled velocity
            if frustrated == 0:   # not frustrated but hoped back at the end
                hoped=0
            else:                # frustrated hopping
                hoped=2
        else:                    # hoped
            hoped=1

        At=A

    return At,Ht,Dt,Vt,hoped,old_state,state

