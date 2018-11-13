# This file is part of the abm prototype implementation of the CoeGSS project

import sys
import numpy as np
import h5py
import datetime

from configuration import communityHierarchyLevel
from configuration import outputFile, nStepsToSimulate, serializeResolution
from configuration import initialDate, initialDateFormat, hoursPerStep, stepsBetweenDemography
from configuration import ageUpdateTag, infectionUpdateTag, recoverUpdatesTag
from configuration import travelStartedUpdateTag, travelIncomingAgentsTag
from population import Population

class World(object):

    def __init__(self, run_number, MPIsize, MPIrank, MPIcomm, logger, **kwargs):
        '''
        Main entity containing the population and providing the interface for the
        population evolution and the system dynamics.

        Attributes
        ----------

        `_run_number` int:
            The number of the run (will be used as the dataset name in the hdf5
            output file).

        `_stepCounter` int:
            The step counter, starts from 0 and gets increased after every day,
            night, travel step.

        `_serializeResolution` int:
            Save system in the hdf5 every `_serializeResolution` number of steps.

        `_totalStepsToSimulate` int:
            Run this number of steps for a simulation run.

        `_population` Population:
            The instance of the synthetic population. Many of the world methods are
            just wrappers around the methods of the Population class.


        Usage
        -----

        Initialize your world instance passing the `run_number` to the
        `__init__(run_number)` method. Then you can use `loadPopulation()` to read
        the population from the file indicated in the configuration file.
        You can then initialize the infection in the population with the
        `initializeInfection()` method (see the method help for schemes and
        strategies of infection) and let it evolve with the `evolve()` method.

        The evolution consists in the call of the `stepDay`, `stepNight` and
        `stepTravel` methods and then in the increment of `_stepCounter` and the
        subsequent check for the need of serialization. Customize the things to do
        for each step in the corresponding method.
        '''
        self._run_number = run_number
        self._MPIsize = MPIsize
        self._MPIrank = MPIrank
        self._MPIcomm = MPIcomm

        self._stepCounter = 0
        #TODO offset in time self._timeZero = 0
        self._serializeResolution = serializeResolution
        self._stepsBetweenDemography = stepsBetweenDemography
        self._totalStepsToSimulate = nStepsToSimulate
        self._population = None
        self._demographyTable = None

        self.logger = logger

        # Saving info on initial date and stuff.
        self._initialDate = datetime.datetime.strptime(initialDate, initialDateFormat)
        self._hoursPerStep = datetime.timedelta(hours=hoursPerStep)
        self._currentDate = self._initialDate

    def loadPopulation(self):
        '''
        Initialize the instance of Population to `self._population`.
        '''
        # TODO: pass here the configuration instance to the population class
        self._population = Population(MPIrank=self._MPIrank,
                MPIsize=self._MPIsize, logger=self.logger)
        self._syncPopAge()
        self._syncResidentPop()

    def _syncResidentPop(self):
        '''
        Since we must know how many agents each node is going to write in the hdf5
        file we synchronize the number of agents RESIDING on each node (i.e. all
        the agents that have `agent.homeNode == self._MPIrank`.
        We then synchronize this number with all the nodes and save it in the
        `self._nResidentsPerNode` array.
        '''

        nResidentsHere = self._population._getNresidents()
        self._nResidentsPerNode = self._MPIcomm.allgather(nResidentsHere)
        if self._MPIrank == 0:
            self.logger.log("N agents per node: %r\n\n" % self._nResidentsPerNode)

    def _syncPopAge(self):
        '''
        Since agents randomly add a float in [0,1) to their age we sync them (the
        work node has an age different from the home node (if on different nodes).

        We collect the age updates for all the agents residing on this node and we
        scatter them to the others.
        '''
        updatesToSend = self._population._getAgeUpdates()
        nUpdatesToSend = np.array([l.shape[0] for l in updatesToSend], dtype="i8")
        nODupdates = self._computeODupdates(nUpdatesToSend)
        updatesReceived = self._sendRecvUpdates(nODupdates, updatesToSend,
                                nCols=2, dtype="f8", tag=ageUpdateTag)
        self._population._applyAgeUpdates(updatesReceived)
        if self._MPIrank == 0:
            self.logger.log("Age updates received: %r\n\n" % updatesReceived)

    def _sendRecvOBJUpdates(self, ODmatrix, localpayload, tag):
        received = [None for i in xrange(self._MPIsize)]
        for target in xrange(self._MPIsize):
            sendBuff = localpayload[target]
            received[target] = self._MPIcomm.sendrecv(sendBuff, target, tag)
        return received

    def _sendRecvUpdates(self, ODmatrix, localpayload, nCols, dtype, tag):
        received = [None for i in xrange(self._MPIsize)]
        for target in xrange(self._MPIsize):
            if target == self._MPIrank: continue
            sendBuff = localpayload[target]

            recvNrows = ODmatrix[target, self._MPIrank]
            recvNcols = nCols
            recvBuff = np.empty((recvNrows, recvNcols), dtype=dtype)
            self._MPIcomm.Sendrecv(sendBuff, target, tag,
                                   recvBuff, target, tag)
            received[target] = recvBuff
        return received

    def _computeODupdates(self, toSend):
        '''
        Given the array of the number of updates to send returns the "origin
        destination" matrix of the updates.
        '''
        assert toSend.shape == (self._MPIsize,)

        nUpdatesToReceive = np.empty([self._MPIsize, self._MPIsize], dtype=toSend.dtype)
        self._MPIcomm.Allgather(toSend, nUpdatesToReceive)
        return nUpdatesToReceive

    def getEntity(self, *args, **kwargs):
        '''
        Calls the `Population.getEntity()` method passing args and kwargs to it.
        '''
        return self._population.getEntity(*args, **kwargs)

    def evolve(self):
        '''
        For each step to simulate call the day, night and travel steps and then check
        for serialization.
        '''
        #TODO: allow for arbitrary number of steps (like for thermalization in the
        # preprocess).

        # Save initial configuration...
        self.serialize()
        while self._stepCounter < self._totalStepsToSimulate:
            self.stepDay()
            self.stepNight()
            self.stepTravel()

            self._stepCounter += 1
            self._currentDate = self.getCurrentDate()

            #TODO if self._stepCounter % self._stepsBetweenDemography == 0:
            #TODO    self.stepDemography()

            if self._stepCounter % self._serializeResolution == 0\
                    or self._stepCounter == self._totalStepsToSimulate:
                self.serialize()

    def serialize(self):
        '''
        A wrapper around the saveStep for the agents table.
        Will include also the saving of the household and workplaces structures
        once they evolve in time.
        '''
        #TODO decent logging
        if self._MPIrank == 0:
            self.logger.log("Serializing at step %05d / %05d..." %
                (self._stepCounter, self._totalStepsToSimulate))

        saveTimeSteps = self._stepCounter == self._totalStepsToSimulate
        self.saveStep(saveTimeSteps=saveTimeSteps)
        if self._MPIrank == 0:
            self.logger.log(" done!\n")

    def terminate(self):
        '''
        Call at the end of the simulation.

        Currently a pass.
        '''
        # TODO in the parallel version close the mpi here or in the main script?
        pass

    def initializeInfection(self, *args, **kwargs):
        '''
        Write here the steps needed to initialize infection.
        So far a wrapper around the `Population.initializeInfection` method to which
        all the *args and **kwargs are passed.

        Once we get the updates from the local population we broadcast them to the
        other nodes.
        '''
        updatesToSend = self._population.initializeInfection(*args, **kwargs)
        self._getNsetInfectionUpdates(updatesToSend)

    def _getNsetInfectionUpdates(self, updatesToSend):
        nUpdatesToSend = np.array([l.shape[0] for l in updatesToSend], dtype="i8")
        nODupdates = self._computeODupdates(nUpdatesToSend)
        updatesReceived = self._sendRecvUpdates(nODupdates, updatesToSend,
                                nCols=5, dtype="i8", tag=infectionUpdateTag)
        self._population._applyInfectionUpdates(updatesReceived)
        if self._MPIrank == 0:
            self.logger.log("Infection updates received: %r\n\n" % updatesReceived)

    def _getNsetRecoverUpdates(self, updatesToSend):
        nUpdatesToSend = np.array([l.shape[0] for l in updatesToSend], dtype="i8")
        nODupdates = self._computeODupdates(nUpdatesToSend)
        updatesReceived = self._sendRecvUpdates(nODupdates, updatesToSend,
                                nCols=1, dtype="i8", tag=recoverUpdatesTag)
        self._population._applyRecoverUpdates(updatesReceived)
        if self._MPIrank == 0:
            self.logger.log("Relapse updates received: %r\n\n" % updatesReceived)

    def _getNsetStartedTravelUpdates(self, updatesToSend):
        nUpdatesToSend = np.array([l.shape[0] for l in updatesToSend], dtype="i8")
        nODupdates = self._computeODupdates(nUpdatesToSend)
        updatesReceived = self._sendRecvUpdates(nODupdates, updatesToSend,
                                nCols=4+communityHierarchyLevel, dtype="i8", tag=travelStartedUpdateTag)
        self._population._applyStartTravelUpdates(updatesReceived)
        if self._MPIrank == 0:
            self.logger.log("Started travels updates received: %r\n\n" % updatesReceived)

    def _checkinTravellingAgents(self, updatesToSend):
        nUpdatesToSend = np.array([len(l) for l in updatesToSend], dtype="i8")
        nODupdates = self._computeODupdates(nUpdatesToSend)
        updatesReceived = self._sendRecvOBJUpdates(nODupdates, updatesToSend,
                                                    tag=travelIncomingAgentsTag)
        self._population._checkinIncomingAgents(updatesReceived)
        if self._MPIrank == 0:
            self.logger.log("Checked in agents travelling updates received: %r\n\n" % updatesReceived)

    def stepDay(self):
        '''
        Write here the steps to be taken in the day step.
        '''
        infectionUpdatesToSend, recoverUpdatesToSend = self._population.stepDay(time=self._stepCounter)
        self._getNsetInfectionUpdates(infectionUpdatesToSend)
        self._getNsetRecoverUpdates(recoverUpdatesToSend)


    def stepNight(self):
        '''
        Write here the steps to be taken in the night step.
        '''
        infectionUpdatesToSend, recoverUpdatesToSend = self._population.stepNight(time=self._stepCounter)
        self._getNsetInfectionUpdates(infectionUpdatesToSend)
        self._getNsetRecoverUpdates(recoverUpdatesToSend)

    def stepTravel(self):
        '''
        Write here the steps needed for the handling of the travelling agents.
        '''
        travelUpdatesToSend, travelAgentsToSend = self._population.stepTravel(time=self._stepCounter)
        # Here send updates to work nodes to tell them that an agent left for a
        # journey then we deploy the agents copies on the destination node...
        # NOTE that here we also collect updates from a node to itself as the
        # destination node may be the same as the home/work one.
        self._getNsetStartedTravelUpdates(travelUpdatesToSend)
        self._checkinTravellingAgents(travelAgentsToSend)

    def stepDemography(self):
        '''
        Write here the steps needed for the handling of the demography.
        '''
        self._population.stepDemography(time=self._stepCounter, date=self._currentDate)

    def saveStep(self, saveTimeSteps=False):
        '''
        Opens the output file and writes the table of the agent for this step.

        The format is like this:
        dataset name: processNumber/_run_number/_stepCounter
        dataset content:
        | idx | sex | age | hh | wp | infection_status | infection_context |
        infection_source | infection_time | treatment |

        You can specify the agent's attributes to be saved in the `typeNames`
        variable and their type in the `typeFormats` one (respect the order!).
        The value will be read using the `agent.getattr(name)` method.

        Note: if a dataset with the specified name is already in the file we delete
        it and write the current value.
        '''
        #TODO Here we rely on the knowledge of the agent's attributes and we also
        # rely on the population structure (calling e.g.
        # `_population._agents.itervalues()`): shall we emancipate from this for
        # maintainability and let the population return the table for us?

        #TODO define the output structure in the configuration file.

        agent_dataset_name_base = "{}/%d/%d/agents/agent" %\
                                    (self._run_number, self._stepCounter)
        # Agents
        agentTypeNames =   ["idx",  "sex",  "age",  "hh",  "wp", "infection_status",\
                       "infection_context", "infection_source",\
                       "infection_time", "treatment", "demo_status", "death_cause"]
        agentTypeFormats = ["<i8", "<i8", "<i8", "<i8", "<i8", "<i8", "<i8", "<i8", "<i8",\
                       "<i8", "<i8", "<i8"]
        agentDtype = np.dtype({"names": agentTypeNames, "formats": agentTypeFormats})

        with h5py.File(outputFile, 'a', driver='mpio', comm=self._MPIcomm) as fOut:
            # All the files must perform all the operations on the file. This also
            # means that we must know how many agents every node is going to
            # write. We are going to use `._nResidentsPerNode` as the reference.
            numProcess = self._MPIrank
            for tmp_rank in xrange(self._MPIsize):
                tmp_name = agent_dataset_name_base.format("%d" % tmp_rank)
                tmp_shape = (self._nResidentsPerNode[tmp_rank],)
                fOut.create_dataset(name=tmp_name, dtype=agentDtype, shape=tmp_shape)

            target_name = agent_dataset_name_base.format(numProcess)
            target_dataset = fOut[target_name]
            nAgentsSaved = 0
            for tmp_agent in self._population._agents.itervalues():
                if tmp_agent.homeNode == numProcess:
                    target_dataset[nAgentsSaved] = tuple(getattr(tmp_agent, k)
                                                      for k in agentTypeNames)
                    nAgentsSaved += 1
            # Saving the dates corresponding to the steps...
            if saveTimeSteps:
                dataset = "timeSteps"
                typeNames =   ["stepNumber",  "date"]
                typeFormats = ["<i8", "S8"]
                tmp_dtype = np.dtype({"names": typeNames, "formats": typeFormats})
                if dataset in fOut: del fOut[dataset]
                tmp_shape = (self._totalStepsToSimulate+1,)
                target_dataset = fOut.create_dataset(name=dataset, dtype=tmp_dtype, shape=tmp_shape)
                if numProcess == 0:
                    for tmp_step in range(self._totalStepsToSimulate+1):
                        target_dataset[tmp_step] = tuple([tmp_step,
                                                    self.step2date(tmp_step).strftime("%Y%m%d")])


    def getCurrentDate(self):
        '''
        Returns the current date.
        '''
        return self.step2date()

    def step2date(self, stepNumber=None):
        '''
        Returns the date corresponding to a step.
        If `stepNumber` is `None` returns the current date.
        '''
        if stepNumber is None:
            stepNumber = self._stepCounter
        return self._initialDate + self._hoursPerStep*stepNumber


