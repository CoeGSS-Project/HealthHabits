# This file is part of the abm prototype implementation of the CoeGSS project

#############
# I/O files #
#############

datasetFile = "data/synthPop_5levels_prova_small.h5"

outputFile = "data/simulation_oneSeed_Out.h5"


#############
# Hierarchy #
#############

# Set here the number of levels that we will have in the synthpop
hierarchyLevels = 5

# Set the level at which you want a community to be defined, e.g. with
# `communityHierarchyLevel = 3` all the nodes with the first three terms of the
# code equals are treated to be in the same community since we assign to the
# community the `geoCode[:communityHierarchyLevel]` code.
communityHierarchyLevel = 3

# Set the columns indicating the levels (in their order from top
# to bottom, i.e. coarse to fine graines) in the HDF5 file household
# and workplaces columns.
hierarchyColumns = ["l0", "l1", "l2", "l3", "l4"]


##########
# Agents #
##########

# Set the dataset name of people in the file
agentsDatasetName = "agent"

# Set the column names of the id, age and sex
idxAgentsColumn = "id"
sexAgentsColumn = "sex"
ageAgentsColumn = "age"
roleAgentsColumn = "role"

# Set the column names and the dtype of the households
# and workplaces/schools ids for the agents table
householdAgentsColumn = "hh"
workplaceAgentsColumn = "wp"

# Set the number of additional attributes, their {columns
# name: name of corresponding attribute of agent}.
additionalAgentsAttributesNumber = 3
additionalAgentsAttributesColumns = {\
                                        "employed": "employed",\
                                        "edu": "education",
                                        "income": "income",\
                                    }

##############
# Households #
##############

# Set the dataset name of the household table
householdsDatasetName = "household"

# Set the columns name of the household id, size
# and lat, lon coordinates
idHouseholdsColumn = "id"
sizeHouseholdsColumn = "size"
kindHouseholdsColumn = "kind"
latHouseholdsColumn = "lat"
lonHouseholdsColumn = "lon"

# Set the number of additional attributes, their {columns
# name: name of corresponding attribute of hh}.
additionalHouseholdsAttributesNumber = 0
additionalHouseholdsAttributesColumns = {}


######################
# Workplaces/Schools #
######################

# Set the dataset name of the household table
workplacesDatasetName = "workplace"

# Set the columns name of the workplace id, size,
# lat, lon coordinates and workplace type
idWorkplacesColumn = "id"
sizeWorkplacesColumn = "size"
kindWorkplacesColumn = "kind"
latWorkplacesColumn = "lat"
lonWorkplacesColumn = "lon"

# Set the number of additional attributes, their {columns
# name: name of corresponding attribute of wp}.
additionalWorkplacesAttributesNumber = 0
additionalWorkplacesAttributesColumns = {}

##############
# Demography #
##############

# Set the dataset name of the demography table
demographyDatasetName = "demography"

# Set the level at which demography is set (columns of the code MUST have the same
# labels and names as in the geocode section), the sex, age, mortality and
# natality columns names and the date column with the format to expect.
demographyGeocodeLevel = 3
demographyDateColumn = "date"
demographyDateFormat = "%Y%m%d"
demographySexColumn = "sex"
demographyAgeColumn = "age"
demographyMortalityColumn = "mortality"
demographyNatalityColumn = "natality"


############
# Dynamics #
############

# The infection rate at home, community, workplace and school
betaHH = .003250
betaCC=  .0003125
betaWP = .0015325
betaSC = .0042865


# The recovery rate
muRate = .05

# Allow travelling?
# Set to false to stop occasional travel.
# TODO: develop more structured and status dependent travel policies (e.g. stop
# travelling if prevalence > 20% or after 20 days from first infection)
allowOccasionalTravels = True

# Allow commuting?
#TODO to be implemented
allowCommuting = True

# Serialize resolution and the total number of steps
# we want to simulate. Also, set the number of steps between two demography
# updates.
nStepsToSimulate = 120
serializeResolution = 2
stepsBetweenDemography = 10
maxAgentAge = 100

# Initial date a and its format, we assume that we start at the midnight of this
# day.
initialDate = "20150101"
initialDateFormat = "%Y%m%d"
hoursPerStep = 24


###################
# Custom settings #
###################

#TODO

##############################
# Default fields translation #
##############################

# Do not touch from here on unless you know what you are doing!

# Here we set the default attributes names that will be bound to the different
# required columns for the agents, households and workplaces datasets.
agentsAttributesTranslation = {\
            idxAgentsColumn: "idx",\
            sexAgentsColumn: "sex",\
            ageAgentsColumn: "age",\
            roleAgentsColumn: "role",\
            householdAgentsColumn: "hh",\
            workplaceAgentsColumn: "wp",\
        }
householdsAttributesTranslation = {\
            idHouseholdsColumn: "idx",\
            sizeHouseholdsColumn: "npeople",\
            kindHouseholdsColumn: "kind",\
            latHouseholdsColumn: "lat",\
            lonHouseholdsColumn: "lon",\
        }
workplacesAttributesTranslation = {\
            idWorkplacesColumn: "idx",\
            sizeWorkplacesColumn: "npeople",\
            kindWorkplacesColumn: "kind",\
            latWorkplacesColumn: "lat",\
            lonWorkplacesColumn: "lon",\
        }

# Where to store the code of the hierarchy (the name of the corresponding
# attribute in the generic group and community classes
hierarchyCodeAttributeName = "geocode"


# The enum types

# The demography status
agentStatusAlive = 0
agentStatusDead = 1
agentDeathCauseNatural = 1
agentDeathCauseIllness = 2
agentSexM = 0
agentSexF = 1
agentRoleChildren = 0
agentRoleParent = 1
agentRoleElderly = 2 #TODO

# The school as workplaces codes
kindergartenCode = 5
elementaryCode = 6
secondaryCode = 7

# Infection status
infectionSusceptible = 0
infectionInfected = 1
infectionAsymptomatic = 2
infectionRecovered = 4

# The agent travel state
notTravelling = 0
businessTravel = 1
leisureTravel = 2

# Treatments and responses codes
treatmentVaccinated = 1

responseRetireHome = 1
responseQuarantine = 2

# The context of infection
infectionInSeed = -1
infectionInHousehold = 0
infectionInWorkplace = 1
infectionInCommunity = 2
infectionInTravelling = 3

# The source of infection from seed
# The other code are given by the role attribute of the Agent.
infectionFromSeed  = -1


