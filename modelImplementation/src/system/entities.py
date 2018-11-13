# This file is part of the abm prototype implementation of the CoeGSS project

import numpy as np
from random import sample
from copy import deepcopy

from configuration import *

class genericEntity(object):

    def __init__(self, dictionary):
        '''
        The parent class from which the synpop entities will inherit.  The class is
        initialized from a dictionary and an attribute <-> value couple is created
        for each key <-> value in the dictionary.
        '''
        for k, v in dictionary.iteritems():
            setattr(self, k, v)

    def __str__(self):
        out_str = ""
        for k in dir(self):
            if k.startswith("__"): continue
            out_str += "%s: %r\n" % (k, getattr(self, k))
        return out_str


class Agent(genericEntity):
    def __init__(self, dictionary):
        '''
        Inherits from `genericEntity`

        Customizable in the `_customInit` method.
        '''
        super(Agent, self).__init__(dictionary)
        self.homeGeocode = None
        self.homeCommunity = None
        self.homeNode = None

        self.workGeocode = None
        self.workCommunity = None
        self.workNode = None

        self.hostingNode = None
        self.hostingCC = None
        self.hostingHH = None
        self.hostingWP = None

        self._customInit()

    def _customInit(self):
        '''
        Sets all the attributes relevant for the epidemic simulation and the agent
        status.
        '''
        # The demography status is assumed to be alive
        # The death code is set to -1 (default) and we initialize the age of the
        # agent to be somewhere uniformly distributed in his year
        self.demo_status = agentStatusAlive
        self.death_cause = -1
        self.age += np.random.rand()

        self.infection_status = infectionSusceptible
        self.infection_context = -1
        self.infection_source = -1
        self.infection_time = -1

        self.treatment = -1
        self.response = -1

        self.isTravelling = 0
        self.travelUntil = -1

    def setUpdatesTargetNodes(self):
        '''
        Sets `self.updatesTargetNodes` to the set of node between home, work and
        travel. Must be called after home/work assignation and after travel
        start/stop
        '''
        #TODO use this method as decorator every time a self.*Node gets accessed for
        # readability and maintainability.
        target_nodes = (n
                for n in (self.homeNode, self.workNode, self.hostingNode)
                if n is not None
            )
        self.updatesTargetNodes = set(target_nodes)

    def isEmployed(self):
        return not self.wp < 0

    def isInfected(self):
        return self.infection_status == infectionInfected

    def isSusceptible(self):
        return self.infection_status == infectionSusceptible

    def isVaccinated(self):
        return self.treatment == treatmentVaccinated

    def inCommunity(self):
        return not ( (self.response == responseQuarantine)\
               or (self.response == responseRetireHome) )

    def atHome(self):
        return (not self.isEmployed()) or (self.response == responseQuarantine)\
               or (self.response == responseRetireHome)

    def inQuarantine(self):
        return self.response == responseQuarantine

    def infect(self, context, source, time):
        '''
        When infected the agent stores the source, context and time of first
        infection.
        '''
        self.infection_status = infectionInfected
        self.infection_context = context
        self.infection_source = source
        self.infection_time = time

    def recover(self):
        '''
        Recovering means no more response and status set to `infectionRecovered`.
        '''
        self.infection_status = infectionRecovered
        self.response = -1

    def checkin(self, hostHH, hostWP):
        self.hostingHH = hostHH
        self.hostingWP = hostWP

    def startTravel(self, mode, until, hostCC, hostNode):
        '''
        Set the travelling status and the target time until agent will be travelling.
        '''
        self.isTravelling = mode
        self.travelUntil = until
        self.hostingCC = hostCC
        self.hostingNode = hostNode
        self.setUpdatesTargetNodes()

    def stopTravel(self):
        '''
        Stops the traveling status.
        '''
        self.isTravelling = notTravelling
        self.travelUntil = -1
        self.hostingCC = None
        self.hostingHH = None
        self.hostingWP = None
        self.hostingNode = None
        self.setUpdatesTargetNodes()

    def startWorkingAt(self, wp):
        self.wp = wp.idx
        self.workGeocode = wp.getGeocode()
        self.workCommunity = wp.getCommunityCode()

    def die(self, deathKind):
        self.demo_status = agentStatusDead
        self.death_cause = deathKind

    def makeSon(self):
        son = deepcopy(self)
        son.age = 1
        son.income = .0
        son.education = 0
        son.sex = np.random.randint(2)
        son.workCommunity = son.homeCommunity
        son.workGeocode = son.homeGeocode
        son.role = agentRoleChildren
        son._customInit()
        return son

    def getAgeUpdate(self):
        return (self.idx, self.age)

    def applyAgeUpdate(self, update):
        self.age = update[1]

    def getInfectionUpdate(self):
        return (self.idx, self.infection_status,
                self.infection_context, self.infection_source, self.infection_time)

    def applyInfectionUpdate(self, update):
        self.infect(context=update[2],
                    source=update[3],
                    time=update[4],
                )

    def getRecoverUpdate(self):
        return (self.idx,)

    def applyRecoverUpdate(self, update):
        self.recover()

    def getStartTravelUpdate(self):
        return (self.idx, self.isTravelling,
                     self.travelUntil, self.hostingNode,)\
               + tuple((c for c in self.hostingCC))

    def applyStartTravelUpdate(self, update):
        self.startTravel(mode=update[1], until=update[2],
                hostNode=update[3], hostCC=tuple(update[4:]))

class genericGroup(genericEntity):
    def __init__(self, *args, **kwargs):
        '''
        Inherits from `genericEntity`.

        Customize the `_customInit` method.

        Attributes
        ----------

        members (set)
            The agent.idx collection of the agents belonging to this entity.

        Usage
        -----

        Add and remove agents from the entity with the `self.addComponent` and
        `self.delComponent` methods. Query for the members via the
        `self.getComponents` method.

        '''
        super(genericGroup, self).__init__(*args, **kwargs)
        self._customInit()
        self.nComponents = 0

    def _customInit(self):
        self.members = set()

    def getGeocode(self):
        return self.geocode

    def getCutLevelCode(self):
        return self.geocode[:cutHierarchyLevel]

    def getCommunityCode(self):
        return self.geocode[:communityHierarchyLevel]

    def getComponents(self):
        return self.members

    def addComponent(self, component):
        self.members.add(component)
        self.nComponents = len(self.members)

    def delComponent(self, component):
        self.members.remove(component)
        self.nComponents = len(self.members)

class Workplace(genericGroup):
    def __init__(self, *args, **kwargs):
        super(Workplace, self).__init__(*args, **kwargs)

class Household(genericGroup):
    def __init__(self, *args, **kwargs):
        super(Household, self).__init__(*args, **kwargs)

class Community(genericEntity):
    def __init__(self, *args, **kwargs):
        super(Community, self).__init__(*args, **kwargs)
        self._customInit()

    def getCommunityCode(self):
        return self.geocode

    def getCutLevelCode(self):
        return self.geocode[:cutHierarchyLevel]

    def _customInit(self):
        self.visitingMembers =  set()
        self.residingMembers =  set()
        self.potentialDailyMembers = set()

        self.hhIDs =  set()
        self.wpIDs =  set()
        self.nDailyComponents = 0
        self.nNightComponents = 0

    def add_hhID(self, hhID):
        self.hhIDs.add(hhID)

    def add_wpID(self, wpID):
        self.wpIDs.add(wpID)

    def del_hhID(self, hhID):
        self.hhIDs.remove(hhID)

    def del_wpID(self, wpID):
        self.wpIDs.remove(wpID)

    def sampleHH(self):
        '''
        Draws an household id uniformly from the community.

        Returns
        -------

        hhID - int
            The `Household.idx` of the selected household.

        Raises
        ------
        RuntimeError
            If no households are found in the community.
        '''
        if len(self.hhIDs) > 0:
            return sample(self.hhIDs, 1)[0]
        else:
            raise RuntimeError("Agent sent to comm %r with no households!" %\
                                (self.geocode))

    def sampleWP(self):
        '''
        Draws a workplace idx uniformly from the community.

        Returns
        -------

        wpID - int
            The `Workplace.idx` of the selected workplace. If no workplaces are found
            in the community return `-1` (default for unemployed).
        '''
        # TODO Here we can pass schools as workplaces, this is fine but we may insert
        # a filter for workplace kind in the parameters.
        if len(self.wpIDs) > 0:
            return sample(self.wpIDs, 1)[0]
        else:
            # If no work here act like an unemployed
            return -1

    def getDailyComponents(self, agents, infection_status):
        '''
        Generator: it yields the
        `(Agent.idx, Agent, Agent.isTravelling, Agent.atHome())`
        tuple.

        Parameters
        ----------

        agents - iterable
            The list of ids from which to determine the actual daily agents
            found in the community during the day.

        infection_status - enumerator for infection status.
            Filter only the agents featuring this infection status.

        Yields
        ------
        (Agent.idx, Agent, Agent.isTravelling, Agent.atHome())
            The tuple containing the agent id, the instance of the agent, her
            travelling status and whether the agent is at home during the day
            (`False` if the agent is working, `True` if unemployed/at home because of
            sickness).
        '''

        tmp_comm_code = self.geocode
        for agent_id in self.potentialDailyMembers:
            tmp_agent = agents[agent_id]
            # Filter out agents with different infection status and those who are not
            # here because of travelling.
            if tmp_agent.infection_status != infection_status:
                continue
            elif tmp_agent.isTravelling:
                if tmp_agent.hostingCC != tmp_comm_code:
                    continue
                isTravelling = True
                atHome = tmp_agent.atHome()
            else:
                isTravelling = False
                if tmp_agent.atHome():
                    if tmp_agent.homeCommunity != tmp_comm_code:
                        continue
                    atHome = True
                else:
                    if tmp_agent.workCommunity != tmp_comm_code:
                        continue
                    atHome = False

            yield agent_id, tmp_agent, isTravelling, atHome

    def getNightComponents(self, agents, infection_status):
        '''
        Generator: it yields the
        `(Agent.idx, Agent, Agent.isTravelling, Agent.inCommunity())`
        tuple.

        Parameters
        ----------

        agents - iterable
            The list of ids from which to determine the actual nightly agents found in the
            community during the night step.

        infection_status - enumerator for infection status.
            Filter only the agents featuring this infection status.

        Yields
        ------
        (Agent.idx, Agent, Agent.isTravelling, Agent.inCommunity())
            The tuple containing the agent id, the instance of the agent, her
            travelling status and whether the agent is in the community during the
            night (`False` if the agent is retired home because of sickness, `True`
            otherwise).
        '''
        tmp_comm_code = self.geocode

        for agent_id in self.residingMembers:
            tmp_agent = agents[agent_id]

            if tmp_agent.infection_status != infection_status:
                continue
            elif tmp_agent.isTravelling:
                if tmp_agent.hostingCC != tmp_comm_code:
                    continue
                isTravelling = True
            else:
                isTravelling = False
            inComm = tmp_agent.inCommunity()
            yield agent_id, tmp_agent, isTravelling, inComm

    def addVisitingComponent(self, component):
        '''
        Adds the component to the set of commuters visiting this community during the
        day to work.

        Parameters
        ----------

        component - Agent.idx
            The idx of the agent to be added. The idx will be added to the
            `self.visitingMembers` and `self.potentialDailyMembers` sets, if not
            already there. Also, `self.nDailyComponents` is increased by one if we
            have a new agent.
        '''
        if component not in self.visitingMembers:
            self.visitingMembers.add(component)
            self.potentialDailyMembers.add(component)
            self.nDailyComponents += 1

    def addResidingComponent(self, component, workplaceID):
        '''
        Adds the component to the set of agents living in this community. If the
        agent also works here we add one to the `self.nDailyComponents` counter.

        Parameters
        ----------

        component - Agent.idx
            The idx of the agent to be added. The idx will be added to the
            `self.residingMembers` and `self.potentialDailyMembers` sets, if not
            already there. Also, `self.nNightComponents` is increased by one if we
            have a new agent and if the agent is also working here we also increase
            `self.nDailyComponents`.

        workplaceID - Workplace.idx
            The idx of the Workplace where the agent works.
        '''
        if component not in self.residingMembers:
            self.residingMembers.add(component)
            self.potentialDailyMembers.add(component)
            self.nNightComponents += 1
            if workplaceID < 0 or workplaceID in self.wpIDs:
                self.nDailyComponents += 1

    def delVisitingComponent(self, component):
        '''
        Removes the specified agent id from the `self.visitingMembers` and
        `self.potentialDailyMembers`,

        Parameters
        ----------

        component - Agent.idx
            The idx of the agent to delete as visitor.

        Raises
        ------
        Error
            If `component` is not found in the `self.visitingMembers` and/or
            `self.potentialDailyMembers`.
        '''
        self.visitingMembers.remove(component)
        self.potentialDailyMembers.remove(component)
        self.nDailyComponents -= 1

    def delResidingComponent(self, component, workplaceID):
        '''
        Provide the `workplaceID` of the ex-workplace the agent is leaving.
        If he is unemployed or he worked in this community we remove him from
        the possible daily container.

        Parameters
        ----------

        component - Agent.idx
            The idx of the agent to delete from the `self.residingMembers`.

        workplaceID - Workplace.idx
            The idx of the workplace where the agent is working

        Raises
        ------
        Error
            If `component` is not found in the `self.residingMembers`.
        '''
        self.residingMembers.remove(component)
        self.nNightComponents -= 1
        if workplaceID < 0 or workplaceID in self.wpIDs:
            self.potentialDailyMembers.remove(component)
            self.nDailyComponents -= 1

    def addTraveler(self, agent):
        '''
        Adds the agent as if she was residing in this community until the end of the
        journey.

        Parameters
        ----------

        agent- Agent
            The instance of the agent to add as a traveler.
        '''
        # TODO uniform the interface to add/delete members to always accept the agent
        # instance as in this case.
        self.addResidingComponent(component=agent.idx,
                                  workplaceID=agent.hostingWP)

    def delTraveler(self, agent):
        '''
        Deletes the agent from the residents of this at the end of the journey.

        Parameters
        ----------

        agent- Agent
            The instance of the agent to delete as a traveler.
        '''
        self.delResidingComponent(component=agent.idx,\
                                  workplaceID=agent.hostingWP)


