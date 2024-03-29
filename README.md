# PyRAI<sup>2</sup>MD
Python Rapid Artificial Intelligence Ab Initio Molecular Dynamics

Citation:
Jingbai Li, Patrick Reiser, Benjamin R. Boswell, André Eberhard, Noah Z. Burns, Pascal Friederich, and Steven A. Lopez, "Automatic discovery of photoisomerization mechanisms with nanosecond machine learning photodynamics simulations", Chem. Sci. 2021. DOI: 10.1039/D0SC05610C

-------------------------------------------------------

                           *
                          /+\
                         /+++\
                        /+++++\
                       /PyRAIMD\
                      /+++++++++\
                     *===========*

                      Python Rapid
                 Artificial Intelligence
              Ab Initio Molecular Dynamics

                   Developer@Jingbai Li
               Northeastern University, USA

                      version:   0.9
                      
    With contributions from(in alphabetic order):
    Andre Eberhard	           - Gaussian process regression
    Jingbai Li/Daniel Susman   - Zhu-Nakamura surface hopping
    Jingbai Li                 - Fewest switches surface hopping
                                 Velocity Verlet
                                 Interface to OpenMolcas/BAGEL
                                 Adaptive sampling (with enforcement)
                                 QC/ML non-adiabatic molecular dynamics
    Patrick Reiser	           - Neural networks (pyNNsMD)

    Special acknowledgment to:
      Steven A. Lopez	 - Project directorship
      Pascal Friederich    - ML directorship
      
-------------------------------------------------------

Python Rapid Artificial Intelligence Ab Initio Molecular Dynamics (PyRAI<sup>2</sup>MD) is a package for nonadiabatic molecular dynamics simulation using potential energy surface predicted by machine-learning models. The primary aim of this project is to leverage the present nonadiabatic molecular dynamics techniques enabling nano-second scale simulations in practical molecular systems, which cost unaffordable computational resources using high-level quantum chemical methods like complete active space self-consistent field (CASSCF) with extended multistate second-order perturbative corrections (XMS-CASPT2).

This is the legacy version. The latest version is 2.0 alpha.
