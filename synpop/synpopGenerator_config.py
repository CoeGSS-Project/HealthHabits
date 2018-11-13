from scipy.stats import norm
import numpy as np
import datetime

from synpopGenerator_tools import age2schoolKind, age2defaultEdu

universityBins = np.arange(1000, 4100, 100)


# Population-dependent...
# Geodataframes
# The geodataframe of the Piedimont area...
cfg = {
"geoDataFrame_file": "resources/Italy/boundaries/Piemonte_NUTS3_to_LAU2_gdf.pkl.gz",
"geoDFid2nuts_file": "resources/Italy/boundaries/Piemonte_NUTS3_to_LAU2_id2NUTS.pkl",
"geoDFnuts2id_file": "resources/Italy/boundaries/Piemonte_NUTS3_to_LAU2_NUTS2id.pkl",
"referenceName": "ITC14",
"selectedNUTS": set(["ITC14"]),
"populationFileName": "resources/Italy/synthPop_ITC4_010pc_2011.h5",
# Scale of the population to save
# Fraction between 0 and 1
"popScale": .2,

# The levels for the communities and other levels to add to the administrative
# ones...
"levelsTargetSize": [5000, 800, 90],

##############################################
##############################################

# Population structure
# The sex strings that will be found in the files
"sexDict": {"male": 0, "female": 1},
# The values of the ages to be simulated, extremes included!
"ages": range(0, 101, 1),

# The dataframes containing the age distribution (PDF and CDF) of the population per
# sex for all the NUTS0-3
"ageBySex_PDF_file": "resources/Europe/population/structure/dataframes/2011_ageBySex_PDF.pkl",
"ageBySex_CDF_file": "resources/Europe/population/structure/dataframes/2011_ageBySex_CDF.pkl",
"popBroadAgeBySex_NUTS3_file": "resources/Europe/population/structure/dataframes/1990-2016_broadAgeBySexNUTS3.pkl",

# The df containing the distribution of household kind per NUTS2
"householdKind_PDF_file": "resources/Europe/population/structure/dataframes/2011_hhType_PDF.pkl",
"householdKind_CDF_file": "resources/Europe/population/structure/dataframes/2011_hhType_CDF.pkl",

# The df containing the size distribution for each kind of household
"householdSizeByKind_PDF_file": "resources/Europe/population/structure/dataframes/2011_hhsizeByType_PDF.pkl",
"householdSizeByKind_CDF_file": "resources/Europe/population/structure/dataframes/2011_hhsizeByType_CDF.pkl",

# The df containing the age distribution of the population per household type (i.e. which age have the components of households)...
"ageByHHrole_PDF_file": "resources/Europe/population/structure/dataframes/2011_ageByHHstatus_PDF.pkl",
"ageByHHrole_CDF_file": "resources/Europe/population/structure/dataframes/2011_ageByHHstatus_CDF.pkl",
"ageByHHrole_RAW_file": "resources/Europe/population/structure/dataframes/2011_ageByHHstatus_RAW.pkl",

# Work and education
# The commuting probabilities for each NUTS3
"studyCommuting_df_file": "resources/Europe/population/structure/dataframes/2011_studyCommuting_ISTAT_NUTS3.pkl",
"workCommuting_df_file": "resources/Europe/population/structure/dataframes/2011_workCommuting_ISTAT_NUTS3.pkl",

# Education indicator...
"educationLevelByAge_PDF_file": "resources/Europe/population/structure/dataframes/educationLevel_NUTS2_PDF.pkl",
"educationLevelByAge_CDF_file": "resources/Europe/population/structure/dataframes/educationLevel_NUTS2_CDF.pkl",

# School attendance per age...
"schoolAttendanceRate_df_file": "resources/Europe/population/structure/dataframes/educationParticipationRate_NUTS2.pkl",

# Employment rate given education...
"employmentBySexAgeEdu_df_file": "resources/Europe/population/structure/dataframes/employmentSexAgeEdu_NUTS2.pkl",

# Schools and wp size...
"schoolSize_df_file": "resources/Europe/population/structure/dataframes/school_PISA_sizeDistribution_NUTS0.pkl",
"workplSize_df_file": "resources/Europe/population/structure/dataframes/wpSizeDistribution_NUTS1.pkl",

"age2schoolKind": age2schoolKind,
"age2defaultEdu": age2defaultEdu,

# Commuting
"minimumPopulationPerWPkind": {0: 1, 1: 1, 2: 15000, 3: 100000, 10: 10},
"maximumDistancePerWPkind": {0: 100, 1: 100, 2: 100, 3: 1000, 10: 1000},

# Gravity model:
"tau_f": .28,
"tau_t": .66,
"rho_exp": 2.95,

# The index position of the attributes...

# Agents
"agentIndex_id": 0,
"agentIndex_hhid": 1,
"agentIndex_role": 2,
"agentIndex_sex": 3,
"agentIndex_age": 4,
"agentIndex_edu": 5,
"agentIndex_wpkind": 6,
"agentIndex_income": 7,
"agentIndex_wpid": 8,

# Households
"hhIndex_id": 0,
"hhIndex_kind": 1,
"hhIndex_size": 2,
"hhIndex_lon": 3,
"hhIndex_lat": 4,
"hhIndex_geocode": 5,

# Workplaces
"wpIndex_id": 0,
"wpIndex_kind": 1,
"wpIndex_size": 2,
"wpIndex_lon": 3,
"wpIndex_lat": 4,
"wpIndex_geocode": 5,

"universityBins": universityBins,
"universityPDF": norm.pdf(universityBins[:-1], loc=2500, scale=200),

# Agents table
"agentTypeNames":   ["id",  "hh",  "role", "sex", "age", "edu", "employed", "income", "wp"],
"agentTypeFormats": ["<i8", "<i8", "<i8",  "<i8", "<i8", "<i8", "<i8",      "<f8",    "<i8"],
"agentDatasetName": "agent",
# Household table
"hhTypeNames": ["id",  "kind", "size", "lon", "lat", "l0",  "l1",  "l2",  "l3",  "l4", "l5"],
"hhTypeFormats": ["<i8", "<i8",  "<i8",  "<f8", "<f8", "<i8", "<i8", "<i8", "<i8", "<i8", "<i8"],
"hhDatasetName": "household",
# Workplaces table
"wpTypeNames":  ["id",  "kind", "size", "lon", "lat", "l0",  "l1",  "l2",  "l3",  "l4", "l5"],
"wpTypeFormats": ["<i8", "<i8",  "<i8",  "<f8", "<f8", "<i8", "<i8", "<i8", "<i8", "<i8", "<i8"],
"wpDatasetName": "workplace",
"typeDatasetName": "types",

# ## Demography
# 
# We load the natality and mortality tables and we save the data for the NUTS codes in the synthetic population.
# Also, we convert the dates of the original dataframe to be in a given time range for simulations.

"natality_df_file": "resources/Europe/population/demography/birthRates_1990-2015_age_nuts2_PANDAS.pkl",
"mortality_df_file": "resources/Europe/population/demography/deathRates_1990-2015_sexAge_nuts2_PANDAS.pkl",

"demographyLevel": 1,
"dateFormat": "%Y%m%d",
"dateZero": datetime.datetime(2015,1,1,0,0,0),

# Demography table
"demographyTypeNames": ["date", "sex", "age", "natality", "mortality"],
"demographyTypeFormats": ["S8", "<i8",  "<i8",  "<f8", "<f8"],

# Where to save/load the checkpoints...
"checkpoint_AG": "checkpoint_AG.pkl",
"checkpoint_HH": "checkpoint_HH.pkl",
"checkpoint_WP": "checkpoint_WP.pkl",

}






