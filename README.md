# Large-scale Disaster Response Modeling and Simulation
This repository contains dataset and source code for modeling and simulation of large-scale disaster response modeling and simulations.

Before running the source/main.py, add new folder named results structured as follows (in bold):
- results
  - cf
  - tv
  - uv

The simulation will run for all combinations of the following parameters
    - victim_list = [100, 250, 500, 750, 1000, 1250, 1500]
    - disaster_site_list = [1,2,3,4,5,6,7]
    - com_duration_list = [5, 15, 30, 45, 60, 75, 90]
    - scenario_list = ['PHO', 'DHO']

Results of the simulation are stored as numpy array in results folder.
