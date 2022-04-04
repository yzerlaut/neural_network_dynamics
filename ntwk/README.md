## Network Dynamics

### simulations

Organized into different parts:

- [./cells/](cells)
  The list of cells available. See [./cells/README.md](cells)
  
- [./build/](build)
  Build the network elements and architecture (equations, connectivity, ...)
  
- [./stim/](stim)
  Stimulate the network with afferent activity.
  
- [./recording/](scan)
  The recording module. See [./recording/README.md](cells)

- [./scan/](scan)
  Perform parameter scans
  
- [./analysis/](analysis)
  Tools to analyze network simulations

- [./plots/](plots)
  Plot network simulation results.


### theory

- [./theory/](theory).
  Implementation of the mean-field approach to network dynamics characterization.
  Based on the Markovian approach formulated in El Boustani & Destexhe, Neural Comp (2009)

- [./transfer_functions/](transfer_functions).
  The core of the mean-field approach.
  The function that accounts for the statistical "rate-behavior" of a population of neurons in the network.
  Procedure to make semi-analytical characterizations of transfer functions. See Zerlaut et al. (2017) JCNS
  

### configs

- [./configs/](configs).
  Stores some cellular and configurations to be easily re-used.
