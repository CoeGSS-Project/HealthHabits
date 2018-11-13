# This file is part of the abm prototype implementation of the CoeGSS project

import os
import sys

from system.world import World
from system.logger import Logger

from configuration import communityHierarchyLevel, cutHierarchyLevel, demographyGeocodeLevel

#TODO run numbering
import pickle

from mpi4py import MPI

MPIcomm = MPI.COMM_WORLD
MPIrank = MPIcomm.Get_rank()
MPIsize = MPIcomm.Get_size()

Log = Logger()

# Load the run number and broadcast it to everyone...
run_number = None
if MPIrank == 0:
    run_number = pickle.load(open("run_number.pkl", "rb"))
    pickle.dump(run_number+1, open("run_number.pkl", "wb"))
    Log.log("Rank %d Size %d, run number %d\n" %(MPIrank, MPIsize, run_number))
run_number = MPIcomm.bcast(run_number, root=0)


# Initialize an instance of World.
# All the needed configuration is read from the default configuration file.
# TODO: specify one single configuration file here for all the modules.
myWorld = World(run_number=run_number, logger=Log,
                MPIsize=MPIsize, MPIrank=MPIrank, MPIcomm=MPIcomm)

# Load the population from the `datasetFile` dataset and prints some examples of the
# read entities.
myWorld.loadPopulation()

if MPIrank == 0:
    Log.log("Agent: %r\nHH: %r\nWP: %r\nComm: %r\nDemog:\n\tMales 35: %r\n\tFemales 35 %r\n\n" %
    (
        myWorld.getEntity("agent", 1), myWorld.getEntity("hh", 1),
        myWorld.getEntity("wp", 1), myWorld.getEntity("comm", tuple(0 for i in range(communityHierarchyLevel))),
        myWorld.getEntity("demo", (0, tuple(0 for i in range(demographyGeocodeLevel)), 0, 35)),
        myWorld.getEntity("demo", (0, tuple(0 for i in range(demographyGeocodeLevel)), 1, 35)),
    ))


# Calls the method of the World entity to initialize the infection.
# We infect one person at random in the community (1,3,3).
# Then lets the system evolve and terminates the simulation afterwards.
myWorld.initializeInfection(nPeople=5,
        seedComm=[(0,0,1), (3,0,0)])
myWorld.evolve()
#TODO myWorld.terminate()
#TODO del myWorld

if MPIrank == 0:
    Log.log("\nDone!\n\n")

