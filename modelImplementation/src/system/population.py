# This file is part of the abm prototype implementation of the CoeGSS project

#from communication.nodesCommunication import Synchronize
import sys
import numpy as np
import datetime

from configuration import *

from entities import Agent, Household, Workplace, Community
from populationIO import synpopReadDataset, synpopReadDemographyTable

assert cutHierarchyLevel > 0
assert communityHierarchyLevel >= cutHierarchyLevel
assert communityHierarchyLevel <= hierarchyLevels


class Population(object):
    _agents = {}
    _workplaces = {}
    _households = {}
    _communities = {}
    _code2node = {}
    _node2codes = {}
    _codesOnNode = set()
    _overallCommunities = []
    _demographyTable = {}

    def __init__(self, MPIrank, MPIsize, logger):
        '''
        The class containing the population and providing the interface to evolve it.
        When initialized saves the table of agents, hh and wp in the corresponding
        attributes.

        This class serves as a container of agents/hh/wps and to provide the methods
        to manipulate the population during the epidemic/system evolution and the
        everyday travelling. Other evolution like demography will be handled
        separately by the demography module.

        Attributes
        ----------
        _agents (dict)
            A dictionary with {idx: Agent()}.

        _workplaces (dict)
            A dictionary with {idx: Workplace()}

        _households (dict)
            A dictionary with {idx: Household()}

        _communities (dict)
            A dictionary with {idx: Community()}

        '''
        # TODO: read the configuration passed as argument to the __init__ method.

        # We read the number of nodes and the rank, then we load first the
        # workplaces and households from which we derive the set of codes...
        # We then split the codes among the nodes IN-PLACE.
        # TODO: we may implement different partitioning schemes depending on the situation:
        # for example here we automatically load the same amount of agents on
        # every node since the last levels (and the community level) of the
        # population have (on average) the same amount of agents per code, so that
        # putting a comparable number of codes in a node will result in a
        # comparable number of agents on the nodes.
        # On the other hand this does not yield a minimization of the intra-nodes
        # communication (that is why we may want to port the procedure to
        # optimally split the code-to-code commuting network on the nodes).

        self._MPIsize = MPIsize
        self._MPIrank = MPIrank
        self.logger = logger

        self.loadDatasetToDict(householdsDatasetName, self._households)
        self.loadDatasetToDict(workplacesDatasetName, self._workplaces)

        # Now compute the codes and split them amongst nodes. Then we clean the
        # workplaces and households not belonging to this node...
        self._splitGeoCodes()
        hhID2node, wpID2node = self._cleanLocations()
        self.loadAgentsTable(hhID2node=hhID2node, wpID2node=wpID2node)
        del hhID2node, wpID2node
        synpopReadDemographyTable(tableName=demographyDatasetName, targetObj=self._demographyTable)


    def _splitGeoCodes(self):
        '''
        Given the codes found in `self._workplaces` and `self._households` we
        split them accordingly to the world size and rank.
        '''

        self._overallCommunities = [wp.getCommunityCode() for wp in self._workplaces.itervalues()]\
                  + [hh.getCommunityCode() for hh in self._households.itervalues()]
        listOfCodes = set([wp.getCutLevelCode() for wp in self._workplaces.itervalues()]\
                  + [hh.getCutLevelCode() for hh in self._households.itervalues()])
        listOfCodes = sorted(list(listOfCodes))
        numberOfCodes = len(listOfCodes)
        splitLevels = np.linspace(0, numberOfCodes, self._MPIsize+1, dtype=int)

        for node in xrange(self._MPIsize):
            tmp_from, tmp_to = splitLevels[node], splitLevels[node+1]
            self._node2codes[node] = set(listOfCodes[tmp_from:tmp_to])
            self._code2node.update({cd: node for cd in listOfCodes[tmp_from:tmp_to]})

            if node == self._MPIrank:
                self._codesOnNode.update(listOfCodes[tmp_from:tmp_to])
        if self._MPIrank == 0:
            self.logger.log("Codes on node 0: %r\n" % self._codesOnNode)
            self.logger.log("node2codes: %r\n" % self._node2codes)
            self.logger.log("code2node: %r\n" % self._code2node)

    def _cleanLocations(self):
        '''
        Keeps only local hh and wp and sets the next ids to use for households and
        workplaces...
        '''
        if self._MPIrank == 0:
            self.logger.log("Before: len wps %d  - len hhs %d\n" %
                            (len(self._workplaces), len(self._households)))

        max_hh_idx = -1
        hhID2node = {}
        hhToKeep = {}
        for k, v in self._households.iteritems():
            max_hh_idx = max(max_hh_idx, k)
            tmp_node = self._code2node[v.getCutLevelCode()]
            hhID2node[k] = tmp_node
            if tmp_node == self._MPIrank:
                hhToKeep[k] = v
        self._households = hhToKeep
        self._nextHouseholdIDX = max_hh_idx + self._MPIrank + 1

        max_wp_idx = -1
        wpID2node = {}
        wpToKeep = {}
        for k, v in self._workplaces.iteritems():
            max_wp_idx = max(max_wp_idx, k)
            tmp_node = self._code2node[v.getCutLevelCode()]
            wpID2node[k] = tmp_node
            if tmp_node == self._MPIrank:
                wpToKeep[k] = v
        self._workplaces = wpToKeep
        self._nextWorkplaceIDX = max_wp_idx + self._MPIrank + 1

        if self._MPIrank == 0:
            self.logger.log("After: len wps %d  - len hhs %d\n" %
                            (len(self._workplaces), len(self._households)))

        return hhID2node, wpID2node

    def loadDatasetToDict(self, datasetName, destination):
        '''
        Parameters
        ----------

        datasetName (str)
            Name of the dataset in the synthetic population file containing the table
            to be load.

        destination (dict)
            Object where the table will be imported. For each row we create an entity
            accordingly to the `datasetName` (i.e. if `datasetName` is the name of
            the agents table we will create an `Agent` from that row.
        '''
        if self._MPIrank == 0:
            self.logger.log("Importing `%s` dataset...\n" % datasetName)
        if datasetName == householdsDatasetName:
            referenceClass = Household
        elif datasetName == workplacesDatasetName:
            referenceClass = Workplace
        else:
            raise KeyError, "Unkwown table `%s` to load" % datasetName

        destination.update({obj["idx"]: referenceClass(obj)
                                for obj in synpopReadDataset(datasetName)
                           })
        if self._MPIrank == 0:
            self.logger.log("\n Done!\n")

    def loadAgentsTable(self, hhID2node, wpID2node):
        '''
        Loads the agent table keeping only the agents that either have a home or
        the workplace in the current node (or both). For the every agent belonging
        to the node we save in the agent the home and work node id.

        For every agent we also annotate it into the corresponding
        household/workplace and corresponding community if these are found on this
        node.

        Function that appends the `idx` of each agent to its household,
        community, and workplace in their components attribute.

        We append the workplace and the household ids to the set of the corresponding
        community. Then, at the end, we annotate the agent to the residing members of
        its home community and, if he works in a different community, to the visiting
        members of the community where he works.

        Note that we have to provide the workplace ID when adding the agent to the
        home community because if he works in the same community he will be counted
        also as a daily member.
        '''
        datasetName = agentsDatasetName
        if self._MPIrank == 0:
            self.logger.log("Importing `%s` dataset...\n" % datasetName)
        referenceClass = Agent

        MPIrank = self._MPIrank
        targetDictionary = self._agents

        max_agent_idx = -1
        for obj in synpopReadDataset(datasetName):
            tmp_agent_hhi = obj["hh"]
            tmp_agent_wpi = obj["wp"]
            tmp_agent_idx = obj["idx"]
            max_agent_idx = max(max_agent_idx, tmp_agent_idx)

            home_node = hhID2node[tmp_agent_hhi]
            work_node = home_node if tmp_agent_wpi < 0 else wpID2node[tmp_agent_wpi]

            if home_node == MPIrank or work_node == MPIrank:
                tmp_agent_obj = referenceClass(obj)
                # Log the home and work nodes
                tmp_agent_obj.homeNode = home_node
                tmp_agent_obj.workNode = work_node
                tmp_agent_obj.setUpdatesTargetNodes()

                targetDictionary[tmp_agent_idx] = tmp_agent_obj

                daily_comm, night_comm = None, None
                if home_node == MPIrank:
                    tmp_hh = self._households[tmp_agent_hhi]
                    tmp_hh.addComponent(tmp_agent_idx)

                    # We get the night geocode (full hierarchy code) and the night community
                    # code (code cut at the selected geolevel) from the hh. If it is the
                    # first time we see this community we add a new key/value to the
                    # corresponding dictionary
                    night_geocode = tmp_hh.getGeocode()
                    night_comm = tmp_hh.getCommunityCode()
                    if night_comm not in self._communities:
                        new_comm = Community({hierarchyCodeAttributeName: night_comm})
                        self._communities[night_comm] = new_comm
                    self._communities[night_comm].add_hhID(tmp_agent_hhi)
                    tmp_agent_obj.homeGeocode, tmp_agent_obj.homeCommunity = night_geocode, night_comm
                    self._communities[night_comm].addResidingComponent(tmp_agent_idx, tmp_agent_wpi)

                if work_node == MPIrank:
                    # We do the same for the daily community: if the agent is unemployed the
                    # daily community corresponds to the night one.
                    if tmp_agent_wpi < 0:
                        # If unemployed he works and lives for sure on the same
                        # node so we already have these two
                        daily_geocode = night_geocode
                        daily_comm = night_comm
                    else:
                        tmp_wp = self._workplaces[tmp_agent_wpi]
                        tmp_wp.addComponent(tmp_agent_idx)
                        daily_geocode = tmp_wp.getGeocode()
                        daily_comm = tmp_wp.getCommunityCode()

                    if daily_comm not in self._communities:
                        new_comm = Community({hierarchyCodeAttributeName: daily_comm})
                        self._communities[daily_comm] = new_comm
                    if tmp_agent_wpi >= 0:
                        self._communities[daily_comm].add_wpID(tmp_agent_wpi)

                    tmp_agent_obj.workGeocode, tmp_agent_obj.workCommunity = daily_geocode, daily_comm

                    # Update the agent's codes then save the agent as a residing component in
                    # the night community and as a visitor in the day components if the
                    # community is different from the daily one.
                    if daily_comm != night_comm:
                        self._communities[daily_comm].addVisitingComponent(tmp_agent_idx)

        if self._MPIrank == 0:
            self.logger.log("\n Done!\n")

        # We will use and increment this counter for new agents.
        self._nextAgentIDX = max_agent_idx + MPIrank + 1

    def _getNresidents(self):
        MPIrank = self._MPIrank
        counter = 0
        for _, tmp_agent in self._agents.iteritems():
            if tmp_agent.homeNode == MPIrank:
                counter += 1
        return counter

    def _getAgeUpdates(self):
        updates = [[] for k in xrange(self._MPIsize)]
        MPIrank = self._MPIrank
        for agent_idx, agent_obj in self._agents.iteritems():
            home_node = agent_obj.homeNode
            work_node = agent_obj.workNode

            if home_node == MPIrank and home_node != work_node:
                updates[work_node].append(agent_obj.getAgeUpdate())
        return [np.array(up, dtype="f8") for up in updates]

    def _applyAgeUpdates(self, updatesReceived):
        targetDict = self._agents
        for target_rank, target_data in enumerate(updatesReceived):
            if target_rank == self._MPIrank: continue
            for update in target_data:
                targetDict[int(update[0])].applyAgeUpdate(update)

    def _applyInfectionUpdates(self, updatesReceived):
        targetDict = self._agents
        for target_rank, target_data in enumerate(updatesReceived):
            if target_rank == self._MPIrank: continue
            for update in target_data:
                targetDict[update[0]].applyInfectionUpdate(update)

    def _applyRecoverUpdates(self, updatesReceived):
        targetDict = self._agents
        for target_rank, target_data in enumerate(updatesReceived):
            if target_rank == self._MPIrank: continue
            for update in target_data:
                targetDict[update[0]].applyRecoverUpdate(update)

    def getEntity(self, which, idx):
        '''
        Returns the entity with id `idx` from the selected table `which`.

        Parameters
        ----------

        which (str)
            The kind of entity you want. Can be one of ["agent", "hh", "wp", "comm"]

        idx (int, tuple)
            The idx of the selected entity.


        Behaviour
        --------

        Raises a `KeyError` if an unknown entity or idx are given.

        '''
        if which == "agent":
            return self._agents[idx]
        elif which == "hh":
            return self._households[idx]
        elif which == "wp":
            return self._workplaces[idx]
        elif which == "comm":
            return self._communities[idx]
        elif which == "demo":
            # Recover the date at the idx[0] position and pass the rest
            assert len(idx) == 4
            date = sorted(self._demographyTable.keys())[idx[0]]
            return self._demographyTable[date][idx[1]][idx[2]][idx[3]]
        else:
            raise KeyError, "Unkwown entity `%s` to return!" % which

    def initializeInfection(self, nPeople, seedComm=None):
        '''
        Parameters
        ----------

        nPeople (int, float)
            Infects agents depending on `nPeople`.
            - if 0 < `nPeople` < 1 we infect this fraction of people in each
              community;
            - if `nPeople` >= 1 we infect this number of people in each community;

        seedComm (iterable, None)
            Specifies the communities to be infected in keys of a dict, elements of a list/tuple in
              `seedComm` if you want the epidemics to be seeded in some communities only.
              If `None` (default) then `nPeople` will get infected in ALL the
              communities.

        Raises
        --------

        Raises a `AssertionError` if `nPeople` <= 0.

        Behavior
        --------

        All the agents with home node on this node are infected if the community
        filter applies to their home (night) community.
        '''
        #TODO insert a seedComm dict {comm: value} for global values of prevalence.
        assert nPeople > 0

        updatesToSend = [[] for i in xrange(self._MPIsize)]
        agentsDict = self._agents
        MPIrank = self._MPIrank
        for tmp_comm_id, tmp_comm in self._communities.iteritems():
            if seedComm is None or tmp_comm_id in seedComm:
                # Here we infect members of this community, using the community
                # only as a fast proxy for membership to retrieve the id of the
                # agents to infect.
                comm_ids = np.array(list(tmp_comm.residingMembers))
                len_comm_ids = len(comm_ids)
                ids_to_infect = []
                num_to_infect = int(np.ceil(len_comm_ids*nPeople))\
                                if nPeople < 1 else int(np.ceil(nPeople))
                ids_to_infect = np.random.choice(comm_ids,\
                                   size=min(len_comm_ids, num_to_infect),\
                                   replace=False)

                for agent_idx in ids_to_infect:
                    agent_obj = agentsDict[agent_idx]
                    agent_obj.infect(context=infectionInSeed,\
                                     source=infectionFromSeed, time=0)
                    work_node = agent_obj.workNode
                    if work_node != MPIrank:
                        updatesToSend[work_node].append(agent_obj.getInfectionUpdate())
        return [np.array(u) for u in updatesToSend]

    def stepDay(self, time):
        '''
        Customize here the daily step.

        Parameters
        ----------

        time (int)
            The timestep value to be saved as infection time.

        Synopsis
        --------
        We annotate the updates (change of infection state) in the updates variable,
        since we have to keep the infection status frozen until the end of the step.

        Note that we use the `Community.getDailyComponents` method that hides the
        complexity of evaluating the possible members of the community during the
        day. However, we still have to check that a contact between two potential
        daily member of the community can happen.

        For each community in this world and for each infected source we try to
        infect susceptible targets. The contact between two agents is possible if:
        - they live in the same community and are unemployed;
        - one or two of them work in this community and the other lives here.
        '''
        # We save the updates for each node (including the local one) in this list
        infectionUpdates = []
        recoverUpdates = []
        for comm_id, comm in self._communities.iteritems():
            # Doing communities...
            # We cycle over the infected agents possibly in the community during the
            # day: visitors, agents travelling here, residing agents retired at home
            # for infection or unemployed agents residing here.
            for source_id, source_agent, source_travelling, source_atHome\
                    in comm.getDailyComponents(self._agents,\
                                       infection_status=infectionInfected):

                # If the source is travelling we use the hosting details for hh and
                # wp, otherwise we use the original ones. Then we check if the agent
                # is in the community or employed (an agent could be constrained at
                # home if in quarantine).
                if source_travelling:
                    source_hh = source_agent.hostingHH
                    source_wp = source_agent.hostingWP
                else:
                    source_hh = source_agent.hh
                    source_wp = source_agent.wp

                source_inComm = source_agent.inCommunity()
                source_employed = source_agent.isEmployed()

                # We cycle over the susceptible agents potentially found in the
                # community during the day.
                for target_id, target_agent, target_travelling, target_atHome\
                    in comm.getDailyComponents(self._agents,\
                                infection_status=infectionSusceptible):
                    # During the day if the agent is in not at home he can
                    # interact with co-workers and community, otherwise he can
                    # interact with household members and, if he is at home
                    # because of unemployment, with the community.
                    if target_id == source_id: continue

                    if target_travelling:
                        target_hh = target_agent.hostingHH
                        target_wp = target_agent.hostingWP
                    else:
                        target_hh = target_agent.hh
                        target_wp = target_agent.wp

                    # Here we compute at runtime the network of interactions between
                    # the agents checking their respective household, workplace and
                    # community id. Moreover, we check if an agent is at home because
                    # of unemployment or quarantine.
                    # Checking household...
                    if (source_atHome and target_atHome)\
                            and (source_hh == target_hh):
                        if np.random.rand() < betaHH:
                            # Annotate the target, context, from_source
                            infectionUpdates.append((target_id, infectionInHousehold, source_agent.role))
                            continue

                    # Checking community...
                    if source_inComm and target_agent.inCommunity():
                        if np.random.rand() < betaCC:
                            # Annotate the target, what, context, from_source
                            infectionUpdates.append((target_id, infectionInCommunity, source_agent.role))
                            continue

                    # Checking workplace...
                    if source_employed and (not (source_atHome or target_atHome))\
                            and (source_wp == target_wp):
                        if np.random.rand() < betaWP:
                            # Annotate the target, what, context, from_source
                            infectionUpdates.append((target_id, infectionInWorkplace, source_agent.role))
                            continue

                # Now that we tried to infect others with this agent we can try
                # to recover him with half rate since we have two steps per day...
                # TODO generic number of days per step...
                if np.random.rand() < muRate/2.:
                    recoverUpdates.append(source_id)

        # Deploy the updates...
        infectionUpdatesToSend = [[] for n in xrange(self._MPIsize)]
        sourceDictionary = self._agents
        MPIrank = self._MPIrank
        for agent_id, context, source in infectionUpdates:
            tmp_agent = sourceDictionary[agent_id]
            tmp_agent.infect(context=context, source=source, time=time)
            payload = tmp_agent.getInfectionUpdate()
            for targetNode in tmp_agent.updatesTargetNodes:
                if targetNode != MPIrank:
                    infectionUpdatesToSend[targetNode].append(payload)
        infectionUpdatesToSend = [np.array(u) for u in infectionUpdatesToSend]

        recoverUpdatesToSend = [[] for n in xrange(self._MPIsize)]
        for agent_id in recoverUpdates:
            tmp_agent = sourceDictionary[agent_id]
            tmp_agent.recover()
            payload = tmp_agent.getRecoverUpdate()
            for targetNode in tmp_agent.updatesTargetNodes:
                if targetNode != MPIrank:
                    recoverUpdatesToSend[targetNode].append(payload)
        recoverUpdatesToSend = [np.array(u) for u in recoverUpdatesToSend]

        return infectionUpdatesToSend, recoverUpdatesToSend


    def stepNight(self, time):
        '''
        Customize here the nightly step.

        Parameters
        ----------

        time (int)
            The timestep value to be saved as infection time.

        Synopsis
        --------
        We annotate the updates (change of infection state) in the updates variable,
        since we have to keep the infection status frozen until the end of the step.

        For each community in this world and for each infected source we try to
        infect susceptible targets. The contact between two agents is possible if:
        - they live in the same community;
        - one or two of them is travelling in this community and/or the other lives here.
        '''
        infectionUpdates = []
        recoverUpdates = []
        for comm_id, comm in self._communities.iteritems():
            # Doing communities...
            for source_id, source_agent, source_travelling, source_inComm\
                    in comm.getNightComponents(self._agents,\
                                       infection_status=infectionInfected):
                if source_travelling:
                    source_hh = source_agent.hostingHH
                else:
                    source_hh = source_agent.hh

                for target_id, target_agent, target_travelling, target_inComm\
                        in comm.getNightComponents(self._agents,\
                                infection_status=infectionSusceptible):

                    if target_travelling:
                        target_hh = target_agent.hostingHH
                    else:
                        target_hh = target_agent.hh

                    # Checking household...
                    if source_hh == target_hh:
                        if np.random.rand() < betaHH:
                            # Annotate the target, context, from_source
                            infectionUpdates.append([target_id, infectionInHousehold,\
                                            source_agent.role])
                            continue

                    # Checking community...
                    if source_inComm and target_inComm:
                        if np.random.rand() < betaCC:
                            # Annotate the target, what, context, from_source and
                            # continue to next agent!
                            infectionUpdates.append([target_id, infectionInCommunity,\
                                            source_agent.role])
                            continue

                # Now that we tried to infect others with this agent as source
                # we can try to recover him with half rate since we have two steps per day...
                if np.random.rand() < muRate/2.:
                    recoverUpdates.append(source_id)

        # Deploy the updates...
        infectionUpdatesToSend = [[] for n in xrange(self._MPIsize)]
        sourceDictionary = self._agents
        MPIrank = self._MPIrank
        for agent_id, context, source in infectionUpdates:
            tmp_agent = sourceDictionary[agent_id]
            tmp_agent.infect(context=context, source=source, time=time)
            payload = tmp_agent.getInfectionUpdate()
            for targetNode in tmp_agent.updatesTargetNodes:
                if targetNode != MPIrank:
                    infectionUpdatesToSend[targetNode].append(payload)
        infectionUpdatesToSend = [np.array(u) for u in infectionUpdatesToSend]

        recoverUpdatesToSend = [[] for n in xrange(self._MPIsize)]
        for agent_id in recoverUpdates:
            tmp_agent = sourceDictionary[agent_id]
            tmp_agent.recover()
            payload = tmp_agent.getRecoverUpdate()
            for targetNode in tmp_agent.updatesTargetNodes:
                if targetNode != MPIrank:
                    recoverUpdatesToSend[targetNode].append(payload)
        recoverUpdatesToSend = [np.array(u) for u in recoverUpdatesToSend]

        return infectionUpdatesToSend, recoverUpdatesToSend

    def stepTravel(self, time):
        '''
        Write here the travel management.

        Parameters
        ----------

        time (int)
            The time step to be used as initial time of the travel. The end of the
            travel  will be at `time + travelLength`.


        Synopsis
        --------
        First, for each travelling agent we check if she finished the travel.
        We do this step also in the blocked travel situation, i.e., we allow for the
        agents to come back home even when travels are blocked.

        We perform this step in parallel on all the nodes since this step can only
        produce predictable outcomes (end of travel and travel back home). In other
        words we do not have to wait for or send updates in this step.

        Then, if travelling is enabled, we let the agents try to leave for a journey.
        In this case we operate 
        '''
        # TODO manage the length of the travels from the configuration by providing a
        # travel length distribution and an origin-destination matrix for realistic
        # modelling.
        MPIrank = self._MPIrank
        travelsToStop = []
        updatesToSend = [[] for i in xrange(self._MPIsize)]
        agentsToSend = [[] for i in xrange(self._MPIsize)]
        for agent_id, tmp_agent in self._agents.iteritems():
            if tmp_agent.isTravelling:
                if tmp_agent.travelUntil <= time:
                    travelsToStop.append(tmp_agent)
            elif allowOccasionalTravels:
                # All the details of the travel handling are stored in one place in
                # the `self._sampleAgentTravel` method.
                if tmp_agent.homeNode == MPIrank and self._startTravelOrNot(tmp_agent):
                    destinationNode, destinationCC, travelKind, travelLength =\
                            self._sampleAgentTravel(agent=tmp_agent)
                    travelUntil = time + travelLength
                    tmp_agent.startTravel(mode=travelKind, until=travelUntil,
                                    hostCC=destinationCC, hostNode = destinationNode,
                            )
                    if tmp_agent.workNode != MPIrank:
                        print MPIrank, tmp_agent.idx, tmp_agent.getStartTravelUpdate()
                        updatesToSend[tmp_agent.workNode].append(tmp_agent.getStartTravelUpdate())
                    agentsToSend[destinationNode].append(tmp_agent)

        # We now stop the travels and then send around the agents leaving for a
        # travel now to the work node and the destination node.
        for tmp_agent in travelsToStop:
            self._stopAgentTravel(agent=tmp_agent)

        updatesToSend = [np.array(u) for u in updatesToSend]
        return updatesToSend, agentsToSend

    def _sampleAgentTravel(self, agent):
        '''
        Parameters
        ----------

        agent (Agent)
            The instance of the agent trying to leave.

        For agents with age > 18 we leave with a certain probability for a trip with a
        random and uniform origin destination matrix and a travel length extracted from a normal
        distribution.
        '''
        # Take off for a travel
        travelKind, travelDuration = self._getTravelKindLength()
        destinationNode, destinationComm = self._getTravelDestination(agent)
        return destinationNode, destinationComm, travelKind, travelDuration

    def _startTravelOrNot(self, agent):
        return (
                (agent.age >= 18)
                and (np.random.rand() < .001)
                and (len(self._communities) > 1)
               )

    def _getTravelKindLength(self):
        travelKind = leisureTravel if np.random.rand() < .1 else businessTravel
        duration = max(4, int(np.ceil(np.random.normal(7., 2.))))
        return travelKind, duration

    def _getTravelDestination(self, agent):
        destination_community = comm_id = agent.homeCommunity
        work_comm_id = agent.workCommunity
        toAvoidComm = (comm_id, work_comm_id)  # since only 2 elements avoid overhead of set
        while destination_community in toAvoidComm:
            dest_id = np.random.randint(len(self._overallCommunities))
            destination_community = self._overallCommunities[dest_id]
        destination_node = self._code2node[destination_community[:cutHierarchyLevel]]
        return destination_node, destination_community

    def _applyStartTravelUpdates(self, updatesReceived):
        targetDict = self._agents
        for target_rank, target_data in enumerate(updatesReceived):
            if target_rank == self._MPIrank: continue
            for update in target_data:
                targetDict[int(update[0])].applyStartTravelUpdate(update)

    def _checkinIncomingAgents(self, incomingAgents):
        '''
        Parameters
        ----------

        incomingAgents (list of list of Agent objs)
            For each source node we have a list of the incoming agents from that
            node.
        '''
        MPIrank = self._MPIrank
        targetDict = self._agents

        for source_node, source_agents in enumerate(incomingAgents):
            appendAgent = False if source_node == MPIrank else True

            for agent in source_agents:
                agent_idx = agent.idx
                if appendAgent:
                    targetDict[agent_idx] = agent
                real_agent = targetDict[agent_idx]

                destination_community = real_agent.hostingCC
                travelKind = real_agent.isTravelling
                destination_hh = self._communities[destination_community].sampleHH()
                if travelKind == businessTravel:
                    destination_wp = self._communities[destination_community].sampleWP()
                else:
                    destination_wp = -1
                #Update the agent's hosting wp and hh
                real_agent.checkin(hostHH=destination_hh, hostWP=destination_wp)
                self._communities[destination_community].addTraveler(real_agent)
                self._households[destination_hh].addComponent(agent_idx)
                if destination_wp >= 0:
                    self._workplaces[destination_wp].addComponent(agent_idx)

    def _stopAgentTravel(self, agent):
        '''
        Parameters
        ----------

        agent (Agent)
            The instance of the agent finishing the travel.

        We have to carefully remove the agent as the traveler from the destination
        community and household/workplace then stopping the travel of the agent.

        Also, we have to perform first the step on the destination node, then the one
        on home and workplace (this is because if the agent is travelling to the same
        home and/or work node the first step on the destination node would erase the
        information on the destination cc, hh and wp!).
        '''
        # Remove agent from hosting community, hh and wp...
        homeWorkNodes = (agent.homeNode, agent.workNode)
        destinationNode = agent.hostingNode
        MPIrank = self._MPIrank
        if destinationNode == MPIrank:
            self._communities[agent.hostingCC].delTraveler(agent)
            self._households[agent.hostingHH].delComponent(agent.idx)
            hostingWP = agent.hostingWP
            if hostingWP >= 0:
                self._workplaces[hostingWP].delComponent(agent.idx)
            if MPIrank not in homeWorkNodes:
                # If home and work are on other nodes we do not want this agent
                # anymore...
                del self._agents[agent.idx]
        agent.stopTravel()

    def stepDemography(self, time, date):
        '''
        Makes a step of demography: check for death and birth and then (if not dead)
        increases the ages of agents.

        Parameters
        ----------

        time - int
            The time step number of the simulation.

        date - datetime
            The current datetime of the simulation (will be used to determine the table to
            read for demography).
        '''
        # TODO households or workplaces left empty by the death of one agent are left in
        # the system and they can be visited by travelling agents. We should specify how
        # to deal with it (i.e. always remove empty groups or leave them as containers).

        # Select the demography table as the most recent in the past-present.
        datetimeToUse = max([d for d in self._demographyTable.keys() if d <= date])
        tmp_demo_table = self._demographyTable[datetimeToUse]
        if self._MPIrank == 0:
            self.logger.log("Demo step for %r %r using %r" %
                                (time, date, datetimeToUse))

        # The fraction of the year covered and the lists where we log the updates to be
        # made (we cannot modify the `_agents` set while traversing it).
        fractionOfYear = stepsBetweenDemography*hoursPerStep/24./365.
        deaths = []
        births = []
        for tmp_idx, tmp_agent in self._agents.iteritems():
            # First check for death then if not dead and if it is a female try to give birth.
            if tmp_agent.demo_status != agentStatusAlive:
                continue

            tmp_sex, tmp_age, tmp_comm = \
                    tmp_agent.sex, tmp_agent.age, tmp_agent.homeGeocode[:demographyGeocodeLevel]

            # Read the age and the death-birth values in the demography.
            tmp_age = int(np.floor(min(max(1, tmp_age), maxAgentAge)))
            tmp_vals = tmp_demo_table[tmp_comm][tmp_sex][tmp_age]
            death_proba = tmp_vals["d"]*fractionOfYear

            if tmp_age >= maxAgentAge or np.random.rand() < death_proba:
                deaths.append([tmp_agent, agentDeathCauseNatural])
            elif tmp_sex == agentSexF and tmp_agent.role == agentRoleParent\
                    and np.random.rand() < tmp_vals["b"]:
                births.append(self._giveBirth(mother=tmp_agent))
            tmp_agent.age += fractionOfYear

        for tmp_agent, death_cause in deaths:
            self._killAgent(tmp_agent, cause=death_cause)
        for son in births:
            self._registerNewAgent(son)

    def _registerNewAgent(self, agent):
        '''
        Register the new agent into the home and work community, to her workplace and
        household and to the agents list.

        Parameters
        ----------

        agent - Agent
            The instance of the agent to be inserted in the `self._agents` dictionary and
            to the respective communities. The agent is added as a residing component in
            the home community. If the work community differs we register the agent also
            there.
        '''
        self._communities[agent.homeCommunity].addResidingComponent(agent.idx, agent.wp)
        if agent.homeCommunity != agent.workCommunity:
            self._communities[agent.workCommunity].addVisitingComponent(agent.idx)

        self._households[agent.hh].addComponent(agent.idx)
        if agent.wp >= 0:
            self._workplaces[agent.wp].addComponent(agent.idx)

        if agent.idx in self._agents:
            raise RuntimeError, "Agent %d already in the system!" % agent.idx
        self._agents[agent.idx] = agent


    def _giveBirth(self, mother):
        '''
        We create a new agent, and assign it to a kindergarten in either the home or work
        community of the mother. We then increment the `_nextAgentIDX` counter for the next baby.

        Parameters
        ----------

        mother - Agent
            The instance of the mother giving birth to the children. The `mother.makeSon`
            method will be used to generate the children.

        Returns
        -------

        son - Agent
            The instance of the son with home and work community and geocode assigned and
            with age=1 and with susceptible infection status.

        '''
        # TODO here we assign the children to a random kindergarten in either the
        # home or work community of the mother, if there is no kindergarten in the
        # two he stays at home!.

        # Create new agent
        son = mother.makeSon()
        tmp_kindergartens = [wpidx for wpidx, wp in self._workplaces.iteritems()
                                if wp.kind == kindergartenCode
                                and wp.geocode in [mother.homeGeocode, mother.workGeocode]]
        son.wp = -1
        if tmp_kindergartens:
            selectedKG = tmp_kindergartens[np.random.randint(len(tmp_kindergartens))]
            son.startWorkingAt(self._workplaces[selectedKG])

        son.idx = self._nextAgentIDX
        self._nextAgentIDX += 1
        return son

    def _killAgent(self, agent, cause):
        '''
        Removes the agent from the home/working community household and workplace (if employed). Marks the
        agent as dead with the specified cause of death. Note that the agent is kept in
        the system and keeps being serialized.

        Parameters
        ----------

        agent - Agent
            The instance of the agent to kill.

        cause - enumerator of death causes
            The death cause to log in the agent.
        '''
        # If travelling call back the agent.
        # Let agent die then remove him from household, workplace and day/night communities.
        #TODO parallel these updates needs to be ported outside for
        # workplace/day-communities outside of the node!
        if agent.isTravelling:
            self._stopAgentTravel(agent=agent)

        agent.die(deathKind=cause)
        self._communities[agent.homeCommunity].delResidingComponent(agent.idx,\
                                                                    agent.wp)
        if agent.homeCommunity != agent.workCommunity:
            self._communities[agent.workCommunity].delVisitingComponent(agent.idx)

        self._households[agent.hh].delComponent(agent.idx)
        WP = agent.wp
        if WP >= 0:
            self._workplaces[WP].delComponent(agent.idx)

    def _getNextAgentid(self):
        res = self._nextAgentIDX
        self._nextAgentIDX += self._MPIsize
        return res

    def _getNextHHid(self):
        res = self._nextHouseholdIDX
        self._nextHouseholdIDX += self._MPIsize
        return res

    def _getNextWPid(self):
        res = self._nextWorkplaceIDX
        self._nextWorkplaceIDX += self._MPIsize
        return res


