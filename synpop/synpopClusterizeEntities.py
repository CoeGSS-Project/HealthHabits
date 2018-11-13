import pickle, gzip
import scipy
from scipy.stats import norm
import geojson
import numpy as np
import datetime
import h5py
import sys
import os

import pandas as pd
import geopandas as gpd
import shapely

from scipy.spatial.distance import pdist, squareform
from haversine import haversine

import synpopStructures
import synpopUtils

import importlib

from synpopGenerator_tools import getEducationLevelCDF, getEmploymentProba,\
        agentSchoolEduEmploy, getEducationRate, filterL0codes


def clusterizeEntities(cfgname, additionalArgs={}):
    cfg_module = importlib.import_module(cfgname)
    cfg = cfg_module.cfg
    cfg.update(additionalArgs)

    print("Loading population data...")
# The reference gdf...
    geoDataFrame = pickle.load(gzip.open(cfg["geoDataFrame_file"], "rb"))
    geoDFid2nuts = pickle.load(open(cfg["geoDFid2nuts_file"], "rb"))
    geoDFnuts2id = pickle.load(open(cfg["geoDFnuts2id_file"], "rb"))
    reference_gdf_tot = geoDataFrame[-1]
    selectedNUTS = cfg["selectedNUTS"]
    reference_gdf = reference_gdf_tot[
            reference_gdf_tot["l0"].apply(
                lambda v: filterL0codes(v, selectedNUTS, geoDFid2nuts)
            )
        ].copy()

# popscale
    popScale = cfg["popScale"]

# The index position of the attributes...
# Agents
    agentIndex_id = cfg["agentIndex_id"]
    agentIndex_hhid = cfg["agentIndex_hhid"]
    agentIndex_role = cfg["agentIndex_role"]
    agentIndex_sex = cfg["agentIndex_sex"]
    agentIndex_age = cfg["agentIndex_age"]
    agentIndex_edu = cfg['agentIndex_edu']
    agentIndex_wpkind = cfg['agentIndex_wpkind']
    agentIndex_income = cfg['agentIndex_income']
    agentIndex_wpid = cfg['agentIndex_wpid']

# Households
    hhIndex_id = cfg["hhIndex_id"]
    hhIndex_kind = cfg["hhIndex_kind"]
    hhIndex_size = cfg["hhIndex_size"]
    hhIndex_lon = cfg["hhIndex_lon"]
    hhIndex_lat = cfg["hhIndex_lat"]
    hhIndex_geocode = cfg["hhIndex_geocode"]

# Workplaces
    wpIndex_id = cfg["wpIndex_id"]
    wpIndex_kind = cfg["wpIndex_kind"]
    wpIndex_size = cfg["wpIndex_size"]
    wpIndex_lon = cfg["wpIndex_lon"]
    wpIndex_lat = cfg["wpIndex_lat"]
    wpIndex_geocode = cfg["wpIndex_geocode"]

    generatedAgents_DF = pd.read_pickle(cfg["checkpoint_AG"])
    hhdf = pd.read_pickle(cfg["checkpoint_HH"])
    wpdf = pd.read_pickle(cfg["checkpoint_WP"])

# Create communities
    print("Creating communities...")
    levelsTargetSize = cfg["levelsTargetSize"]

    nHouseholds = hhdf.shape[0]
    nWorkplaces = wpdf.shape[0]

    locationsDF = hhdf.append(wpdf, ignore_index=True, verify_integrity=True)

    tic = datetime.datetime.now()

# Prepare the new column for the local code...
    nLevelsToAdd = len(levelsTargetSize)
    localCodeCols = ["lo_code_%d" % i for i in range(nLevelsToAdd)]
    for col in localCodeCols:
        locationsDF[col] = -1
    ini_idx, fin_idx = -1, -1
    i = 0
    nGroups = len(locationsDF["geocode"].unique())
    for tmp_geocode, tmp_group in locationsDF.groupby("geocode"):
        i += 1
        sys.stdout.write("\r%04d / %04d" % (i, nGroups))
        sys.stdout.flush()
        if i < ini_idx:
            continue

        xys = np.array(tmp_group[["lon", "lat"]])
        ids = tmp_group.index
        N = reference_gdf[reference_gdf["code"] == tmp_geocode]["POP"].values[0]*popScale
        nClustersPerLevel = [max(1, N/float(levelTargetSize))
                                for levelTargetSize in levelsTargetSize]
        if nClustersPerLevel[-1] > 1:
            tmp_res = synpopUtils.clusterPointsInLevelsBottomUp(
                    xys=xys, nClustersPerLevel=nClustersPerLevel,
                    n_init=4, sampleFrac=.1)
        else:
            # All in one cluster...
            tmp_res = np.zeros(shape=(xys.shape[0], nLevelsToAdd), dtype=int)
        locationsDF.loc[ids, localCodeCols] = tmp_res
        if i == fin_idx:
            break
    toc = datetime.datetime.now()
    print("\nCommunities generated in %f seconds..." % (toc-tic).total_seconds())

# ## Split back the dataframe and create the agents one

# Uncompress the nuts code
    lenGeoCode = len(locationsDF["geocode"].iloc[0])
    lenLocalCode = nLevelsToAdd

    for level in range(lenGeoCode+lenLocalCode):
        label = "l%d" % level
        locationsDF[label] = -1
        if level < lenGeoCode:
            locationsDF[label] = locationsDF["geocode"].apply(lambda v: v[level])
        else:
            locationsDF[label] = locationsDF["lo_code_%d" % (level-lenGeoCode)]

    generatedHouseholds_DF = locationsDF.loc[:nHouseholds].copy(deep=True)
    generatedHouseholds_DF.reset_index(inplace=True)
    generatedHouseholds_DF.rename(columns={"index": "id"}, inplace=True)

    generatedWorkplaces_DF = locationsDF.loc[nHouseholds:].copy(deep=True)
    generatedWorkplaces_DF.set_index(np.arange(generatedWorkplaces_DF.shape[0]), inplace=True)
    generatedWorkplaces_DF.reset_index(inplace=True)
    generatedWorkplaces_DF.rename(columns={"index": "id"}, inplace=True)

    del generatedWorkplaces_DF["geocode"]
    del generatedHouseholds_DF["geocode"]
    for col in localCodeCols:
        del generatedHouseholds_DF[col]
        del generatedWorkplaces_DF[col]

    populationFileName = cfg["populationFileName"]
    fout = h5py.File(populationFileName, mode="w")
    fout.clear()

# Agents table
    typeNames =   cfg["agentTypeNames"]
    typeFormats = cfg["agentTypeFormats"]
    agentType = np.dtype({"names": typeNames, "formats": typeFormats, })
    dataAsMatrix = generatedAgents_DF.as_matrix()
    dataAsMatrix = np.array([tuple(row) for row in dataAsMatrix], dtype=agentType)
    fout.create_dataset(name=cfg["agentDatasetName"], dtype=agentType, data=dataAsMatrix)
    del dataAsMatrix, generatedAgents_DF

# Household table
    nFieldsToKeep = 8 + nLevelsToAdd
    typeNames =   cfg["hhTypeNames"]
    typeFormats = cfg["hhTypeFormats"]
    householdType = np.dtype({
        "names": typeNames[:nFieldsToKeep],
        "formats": typeFormats[:nFieldsToKeep],
                              })
    dataAsMatrix = generatedHouseholds_DF.as_matrix()
    dataAsMatrix = np.array([tuple(row) for row in dataAsMatrix], dtype=householdType)
    fout.create_dataset(name=cfg["hhDatasetName"], dtype=householdType, data=dataAsMatrix)
    del dataAsMatrix, generatedHouseholds_DF

# Workplaces table
    typeNames =   cfg["wpTypeNames"]
    typeFormats = cfg["wpTypeFormats"]
    workplaceType = np.dtype({
        "names": typeNames[:nFieldsToKeep],
        "formats": typeFormats[:nFieldsToKeep],
                            })
    dataAsMatrix = generatedWorkplaces_DF.as_matrix()
    dataAsMatrix = np.array([tuple(row) for row in dataAsMatrix], dtype=workplaceType)
    fout.create_dataset(name=cfg["wpDatasetName"], dtype=workplaceType, data=dataAsMatrix)
    del dataAsMatrix, generatedWorkplaces_DF

# Define the types
    type_group = fout.create_group(cfg["typeDatasetName"])
    type_group[cfg["agentDatasetName"]] = agentType
    type_group[cfg["hhDatasetName"]] = householdType
    type_group[cfg["wpDatasetName"]] = workplaceType


# Demography
# We load the natality and mortality tables and we save the data for the NUTS codes
# in the synthetic population.
# Also, we convert the dates of the original dataframe to be in a given time range
# for simulations.

    natality_df  = pd.read_pickle(cfg["natality_df_file"])
    mortality_df = pd.read_pickle(cfg["mortality_df_file"])

    demographyLevel = cfg["demographyLevel"]
    dateFormat = cfg["dateFormat"]
    outData = []
    for tmp_nuts_code, tmp_nuts_id in geoDFnuts2id.iteritems():
        # The NUTS codes we have to fetch from the table...
        tmp_stat_code = tmp_nuts_code
        while tmp_stat_code not in natality_df.index:
            #print tmp_stat_code
            tmp_stat_code = tmp_stat_code[:-1]
        for year in natality_df.columns.get_level_values(0).unique():
            if year < 2002: continue
            for sex in [0, 1]:
                for age in range(1, 101):
                    tmp_row = [tmp_nuts_id, datetime.datetime(year, 1, 2)
                                              .strftime(dateFormat), sex, age]
                    tmp_column = tuple([year, sex, age])
                    outData.append(tmp_row
                        + [natality_df[tmp_column][tmp_stat_code],
                           mortality_df[tmp_column][tmp_stat_code]])

    demographyDF = pd.DataFrame(outData, columns=["l%d"%d for d in range(demographyLevel)] + ["date", "sex", "age", "natality", "mortality"])

# Create fake dates
    currentDates = demographyDF.date.unique()
    numberOfDates = len(currentDates)
    timeDeltaPerDate = datetime.timedelta(days=int(np.ceil(200./numberOfDates)))
    dateZero = datetime.datetime(2015,1,1,0,0,0)
    old2newDates = {oldDate: (dateZero+timeDeltaPerDate*iii).strftime(dateFormat)
                        for iii, oldDate in enumerate(currentDates)}

    demographyDF.replace(to_replace={"date": old2newDates}, inplace=True)
    demographyDF.date.unique()

# Demography table
    typeNames =   ["l%d" % d for d in range(demographyLevel)]\
                    + ["date", "sex", "age", "natality", "mortality"]
    typeFormats = ["<i8"]*demographyLevel + ["S8", "<i8",  "<i8",  "<f8", "<f8"]
    demographyType = np.dtype({"names": typeNames, "formats": typeFormats,}) 
    dataAsMatrix = demographyDF.as_matrix()
    dataAsMatrix = np.array([tuple(row) for row in dataAsMatrix], dtype=demographyType)
    fout.create_dataset(name="demography", dtype=demographyType, data=dataAsMatrix)

    fout.close()


if __name__ == "__main__":
    cfg_module = sys.argv[1].split(".")[0]
    clusterizeEntities(cfgname=cfg_module)
