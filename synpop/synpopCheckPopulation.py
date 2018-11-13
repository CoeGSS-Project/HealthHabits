import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scipy
import pickle
import gzip
import numpy as np
import datetime
import sys
import os

from scipy.spatial.distance import pdist, squareform
from haversine import haversine

import pandas as pd
import geopandas as gpd
import shapely

import h5py

import synpopStructures
import synpopUtils

import importlib

from synpopGenerator_tools import getEducationLevelCDF, getEmploymentProba,\
        agentSchoolEduEmploy, getEducationRate, universityBins, universityPDF,\
        age2schoolKind

#################################################
#################################################

def checkPopulation(cfgname, additionalArgs={}):
    cfg_module = importlib.import_module(cfgname)
    cfg = cfg_module.cfg
    cfg.update(additionalArgs)

    print("Loading configuration...")
# Scale of the population to save
# Fraction between 0 and 1
    popScale = cfg["popScale"]
    populationFileName = cfg["populationFileName"]
    referenceName = cfg["referenceName"]
    selectedNUTS = cfg["selectedNUTS"]
    countryCode = list(selectedNUTS)[0][:2]
    NUTS3code = list(selectedNUTS)[0][:3]
    minimumPopulationPerWPkind = cfg["minimumPopulationPerWPkind"]


    workplacesDict = {k: synpopStructures.workplace(k) for k in
            minimumPopulationPerWPkind}

    universityDistrib = {"BINS": universityBins,
                     "CDF": (np.diff(universityBins)*universityPDF).cumsum(),
                     "PDF": universityPDF,
                    }
    workplacesDict[3].set_sizePDF(universityDistrib)

#################################################
#################################################

    ageBySex_PDF = pd.read_pickle(cfg["ageBySex_PDF_file"])
    ageBySex_CDF = pd.read_pickle(cfg["ageBySex_CDF_file"])

# We also load the data for nuts3 levels (we use them to evaluate the number of
# agents to create)
    print("Loading Eurostat data...")
    popBroadAgeBySex_NUTS3 = pd.read_pickle(cfg["popBroadAgeBySex_NUTS3_file"])

# The df containing the distribution of household kind per NUTS2
    householdKind_PDF = pd.read_pickle(cfg["householdKind_PDF_file"])
    householdKind_CDF = pd.read_pickle(cfg["householdKind_CDF_file"])

# The df containing the size distribution for each kind of household
    householdSizeByKind_PDF = pd.read_pickle(cfg["householdSizeByKind_PDF_file"])
    householdSizeByKind_CDF = pd.read_pickle(cfg["householdSizeByKind_CDF_file"])

# The df containing the age distribution of the population per household type (i.e.
# which age have the components of households)
    ageByHHrole_PDF = pd.read_pickle(cfg["ageByHHrole_PDF_file"])
    ageByHHrole_CDF = pd.read_pickle(cfg["ageByHHrole_CDF_file"])
    ageByHHrole_RAW = pd.read_pickle(cfg["ageByHHrole_RAW_file"])

# ### Work and education

# The commuting probabilities for each NUTS3
    studyCommuting_df = pd.read_pickle(cfg["studyCommuting_df_file"])
    workCommuting_df = pd.read_pickle(cfg["workCommuting_df_file"])

# Education indicator...
    educationLevelByAge_PDF = pd.read_pickle(cfg["educationLevelByAge_PDF_file"])
    educationLevelByAge_CDF = pd.read_pickle(cfg["educationLevelByAge_CDF_file"])

# School and employment rate given education...
    schoolAttendanceRate_df = pd.read_pickle(cfg["schoolAttendanceRate_df_file"])
    employmentBySexAgeEdu_df = pd.read_pickle(cfg["employmentBySexAgeEdu_df_file"])

# Schools and wp size...
    schoolSize_df = pd.read_pickle(cfg["schoolSize_df_file"])
    workplSize_df = pd.read_pickle(cfg["workplSize_df_file"])

# Geodataframe
    print("Loading boundaries data...")
    geoDataFrame = pickle.load(gzip.open(cfg["geoDataFrame_file"], "rb"))
    geoDFid2nuts = pickle.load(open(cfg["geoDFid2nuts_file"], "rb"))
    geoDFnuts2id = pickle.load(open(cfg["geoDFnuts2id_file"], "rb"))
    reference_gdf_tot = geoDataFrame[-1]
    selectedNUTS = cfg["selectedNUTS"]
    def filterL0codes(l0, toKeep, id2code):
        tmp_code = id2code[l0]
        for keep in toKeep:
            if tmp_code.startswith(keep):
                return True
        return False


    reference_gdf = reference_gdf_tot[
            reference_gdf_tot["l0"].apply(
                lambda v: filterL0codes(v, selectedNUTS, geoDFid2nuts)
            )
        ].copy()


# Household labels: the ones for the age structure and the ones for the actual household kinds
    ageHouseholdLabels = set(ageByHHrole_RAW.columns.get_level_values(1).unique())
    ageHouseholdLabels.discard("TOTAL")

    householdLabels = set(householdKind_PDF.columns)
    householdLabels.discard("TOTAL")

# Couple without children
    CPL_NCH = synpopStructures.householdType(minMaxParents=(2,2), minMaxSons=(0,0), ageMinMaxParents=(18,100), sexParents="etero",
                                             agePDFparents=None, agePDFsons=None)

# Couple with young and/or old children
    CPL_WCH = synpopStructures.householdType(minMaxParents=(2,2), minMaxSons=(1,9), ageMinMaxParents=(18,100), ageMinMaxChildren=(0, 80),
                                             dMinMaxParSon=(18,50), sexParents="etero", sexSons="free",
                                             agePDFparents=None, agePDFsons=None,
                                           )

# Lone father/mother with young/old children
    M1_CH  = synpopStructures.householdType(minMaxParents=(1,1), minMaxSons=(1,10), ageMinMaxParents=(18,100), ageMinMaxChildren=(0,80),
                                             dMinMaxParSon=(18,50), sexParents="male", sexSons="free",
                                             agePDFparents=None, agePDFsons=None,
                                           )

    F1_CH  = synpopStructures.householdType(minMaxParents=(1,1), minMaxSons=(1,10), ageMinMaxParents=(18,100), ageMinMaxChildren=(0,80),
                                             dMinMaxParSon=(18,50), sexParents="female", sexSons="free",
                                             agePDFparents=None, agePDFsons=None,
                                           )

# Singles and multihouseholds share the same age distribution (but different size)
    A1_HH  = synpopStructures.householdType(minMaxParents=(1,1), minMaxSons=(0,0), ageMinMaxParents=(15, 100), sexParents="free",
                                             agePDFparents=None, agePDFsons=None,
                                           )
    MULTI_HH  = synpopStructures.householdType(minMaxParents=(2,11), minMaxSons=(0,0), ageMinMaxParents=(15, 100),
                                             sexParents="free", dMinMaxp1p2=(0, 40), dMinMaxParSon=(30, 100), fixedParentsSons=(False, True),
                                             agePDFparents=None, agePDFsons=None,
                                           )

# Save the households in an array and in a dictionary to remember their order.
# We also save the column from which they will inherit the parent and sons
# age PDF from the aggregation.
    houseHoldTypeDict = {
            "CPL_NCH":  {"obj": CPL_NCH,  "id": None,
                         'parentAgePDFName': "CPL_XCH",
                         'childsAgePDFName': None,
                        },
            "CPL_WCH":  {"obj": CPL_WCH,  "id": None,
                         'parentAgePDFName': "CPL_XCH",
                         'childsAgePDFName': "CH_PAR",
                        },
            "M1_CH":    {"obj": M1_CH,    "id": None,
                         'parentAgePDFName': "A1_XCH",
                         'childsAgePDFName': "CH_PAR",
                        },
            "F1_CH":    {"obj": F1_CH,    "id": None,
                         'parentAgePDFName': "A1_XCH",
                         'childsAgePDFName': "CH_PAR",
                        },
            "A1_HH":    {"obj": A1_HH,    "id": None,
                         'parentAgePDFName': "A1_HH",
                         'childsAgePDFName': None,
                        },
            "MULTI_HH": {"obj": MULTI_HH, "id": None,
                         'parentAgePDFName': "A1_HH",
                         'childsAgePDFName': None,
                        },
        }

    nHouseholdKinds = len(householdLabels)
    houseHoldTypeArray = [None]*nHouseholdKinds
    for idx, hhLabel in enumerate(householdLabels):
        tmp_householdEntry = houseHoldTypeDict[hhLabel]
        tmp_householdEntry["id"] = idx
        houseHoldTypeArray[idx] = tmp_householdEntry["obj"]
    houseHoldTypeArray = np.array(houseHoldTypeArray)

# Open the reference file and load the three tables
    f = h5py.File(populationFileName, "r")

    loaded_array = dict()
    agDSname = cfg["agentDatasetName"]
    hhDSname = cfg["hhDatasetName"]
    wpDSname = cfg["wpDatasetName"]
    for dataset_name in (agDSname, hhDSname, wpDSname):
        dataset = f[dataset_name]
        tmp_array = np.empty(shape=dataset.shape, dtype=dataset.dtype)
        dataset.read_direct(tmp_array)
        loaded_array[dataset_name] = tmp_array
    f.close()

    ags = loaded_array[agDSname]
    wps = loaded_array[wpDSname]
    hhs = loaded_array[hhDSname]

# Check the generated population

    print("Checking population...")

# ## Commuting
# 
# We check the distance distribution between home and workplace/school.
# 
# We compare our findings with the theoretical curve of
# 
# $P(d) \propto \frac{1}{(1+d/a)^b}$
# 
# with $a=3.8$ km and $b=2.32$.
# 
# Since we are generating only one region we do not reproduce the tail of the distribution as we are missing long distances travels. However it is clear that we are correctly reproducing the travel distances as the generated distribution closely follows the reference one in the $d\lesssim 50$ km.

# Compute the distance matrix...
    baricenters_LatLon = np.array(reference_gdf[["BARICENTER_Y", "BARICENTER_X"]])
    baricenters_distanceM = squareform(pdist(baricenters_LatLon, metric=haversine))

    plt.imshow(baricenters_distanceM)
    cbar = plt.colorbar(shrink=.85)
    cbar.set_label(r"Distance - $d_{ij}\; (km)$", size=22)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel(r"Destination - $j$", size=18)
    plt.ylabel(r"Origin - $i$", size=18)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout()
    plt.savefig("figures/synPop_%s_01_LAU2distanceMatrix.pdf" % (referenceName,), bbox_inches="tight")
    plt.close()

# Commuting distance
    commutingDistancesPerWPkind = {k: [] for k in workplacesDict}

    for wp_kind in np.unique(ags["employed"]):
        #, commuters in generatedAgents_DF.groupby("employed"):
        print wp_kind
        if wp_kind < 0:
            continue
        commuters = ags[ags["employed"] == wp_kind]
        tmp_dists = np.zeros(commuters.shape[0])
        iii = 0
        for commuter in commuters:
            hh_id = commuter["hh"]
            wp_id = commuter["wp"]
            tmp_hh = hhs[hh_id]
            tmp_wp = wps[wp_id]
            tmp_dists[iii] = haversine((tmp_hh["lat"], tmp_hh["lon"]),
                                       (tmp_wp["lat"], tmp_wp["lon"]))
            iii += 1
        commutingDistancesPerWPkind[wp_kind] = tmp_dists

    fig = plt.figure(figsize=(5,4))

    iii = 1
    for wp_kind, data in commutingDistancesPerWPkind.iteritems():
        f, b = np.histogram(data, bins=np.logspace(0, 3, 30), density=True)
        b = (b[1:] + b[:-1])/2.
        b = b[f>0]
        f = f[f>0]
        plt.loglog(b,f,label=wp_kind)

# Plot the reference one...
    b = np.array(list(b) + [max(100, b[-1]*1.5)])
    plt.loglog(b, .5*(1. + b/3.8)**-2.32, "--k", lw=2, label=r"$Thr$")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(5, b[-1])
    plt.xlabel(r"Distance $d$ (km)", size=18)
    plt.ylabel(r"$P(d)$", size=18)

    plt.xticks(size=16)
    plt.yticks(size=16)

    plt.legend(fontsize=14, ncol=3, loc=1, handlelength=.5, labelspacing=.7, columnspacing=.5)
    plt.tight_layout()
    plt.savefig("figures/synPop_%s_02_CommutingDistancesPDF.pdf" % (referenceName,),
            bbox_inches="tight")
    plt.close()


## Workplaces size distribution
# Since we generated only the 10% of the population we get a lot of small workplaces
# but the overall shape of the distribution is reproduced.

# Workplace kind...

    # The generated data...
    nWorkplaceKind = len(minimumPopulationPerWPkind.keys())

    nCols = 3
    nRows = nWorkplaceKind // nCols + 1
    plt.subplots(nRows, nCols, figsize=(4.5*nCols, 5*nRows))

    for subplot, wp_kind in enumerate(sorted(minimumPopulationPerWPkind)):
        if wp_kind < 10:
            if wp_kind < 3:
                reference_data = schoolSize_df.loc[countryCode]
            else:
                reference_data = universityDistrib
        else:
            reference_data = workplSize_df.loc[NUTS3code]
        plt.subplot(nRows, nCols, subplot+1)
        plt.title("WP kind = %d" % wp_kind)
        f, b = np.histogram(wps[wps["kind"] == wp_kind]["size"],
                         bins=reference_data["BINS"], density=True);
        if len(b) == len(f) + 1:
            b = (b[1:] + b[:-1])/2.
            b = b[f>1e-6]
            f = f[f>1e-6]
            plt.plot(f, "^-C1", label="Generated")
            #plt.plot((reference_data["BINS"][1:] +reference_data["BINS"][:-1])/2.,
            plt.plot(reference_data["PDF"], "o-C0", label="Actual data");
        plt.xscale("linear")
        plt.yscale("log")
        tmp_empiricalBNS = reference_data["BINS"]
        nBins = len(tmp_empiricalBNS)
        step = 1
        if nBins > 7:
            step = nBins/6
        locs = np.arange(0, nBins-1, step)
        labs = ["%d-%d" % (tmp_empiricalBNS[i], tmp_empiricalBNS[i+1])
                            for i in locs]
        plt.xticks(locs, labs, size=14, rotation=35, ha="right")
        plt.yticks(size=14)

        plt.xlabel(r"Size - $s$", size=18)
        plt.ylabel(r"$P(s)$", size=18)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("figures/synPop_%s_wpKindSize.pdf" % (referenceName,),
            bbox_inches="tight")
    plt.close()

# Occupation of agents...

# Compute the expected values...
# Procedure:
# - expected school = sum_age attendance_rate_age * pop_age
# - expected employ = sum_age edu_age * employ|edu_age * pop_age
# - expected unemply = N - expected school - expected employ
    NperWpKind = {k: .0 for k in minimumPopulationPerWPkind.keys()}
    home_nuts = [geoDFid2nuts[hhs[ag["hh"]]["l0"]] for ag in ags]
    for NUTS_code in selectedNUTS:
        tmp_nuts= NUTS_code
        while tmp_nuts not in educationLevelByAge_CDF.index:
            tmp_nuts = tmp_nuts[:-1]
        edu_levl_cdf = getEducationLevelCDF(
                educationLevelByAge_CDF["2011"].loc[tmp_nuts])
        edu_rate_age = getEducationRate(
                schoolAttendanceRate_df["2013"].loc[tmp_nuts])
        emp_rate_edu = getEmploymentProba(
                employmentBySexAgeEdu_df["2011"].loc[tmp_nuts])

        for agent, tmp_home_nuts in zip(ags, home_nuts):
            if not tmp_home_nuts.startswith(NUTS_code): continue
            age, sex = agent["age"], agent["sex"]
            # Going to school
            tmp_edu_rate = edu_rate_age[age, sex]
            if age < 25:
                NperWpKind[age2schoolKind(age)] += tmp_edu_rate
            # Work
            tmp_notedu_rate = 1. - tmp_edu_rate
            pdf = np.concatenate((np.array([edu_levl_cdf[age,sex][0]]), np.diff(edu_levl_cdf[age, sex], axis=0)))
            assert .0 <= tmp_edu_rate <= 1.
            assert .9995 < pdf.sum() < 1.0005
            for edu, frac in enumerate(pdf):
                NperWpKind[10] += frac*emp_rate_edu[age, sex, edu]*tmp_notedu_rate
    NperWpKind[-1] = ags.shape[0] - sum(NperWpKind.values())
    NperWpKind[4] = NperWpKind.pop(10)

    # the generated values...
    bins = np.array([-1.5, -.5, .5, 1.5, 2.5, 3.5, 4.5])
    vals = np.array(ags["employed"])
    vals[vals == 10] = 4
    plt.hist(vals, bins=bins, rwidth=.75, color="C1", lw=2, label="Generated")

    Xs = sorted(NperWpKind.keys())
    Ys = [NperWpKind[k] for k in Xs]
    plt.plot(Xs, Ys, "o-C0", label="Actual data", lw=2, ms=14)
    loc2label = {
            -1: "Unemployed", 0: "Kindergarten", 1: "Primary Sc.",
            2: "Secondary Sc.", 3: "University", 4: "Employed",
            }

    plt.ylabel("Number of agents", size=18)
    plt.xticks(sorted(loc2label.keys()), [v for k, v in sorted(loc2label.iteritems())],
                size=16, rotation=45, ha="right")
    plt.yticks(size=16)

    plt.legend(fontsize=16, loc="upper left", bbox_to_anchor=[.25,.95])
    plt.tight_layout()
    plt.savefig("figures/synPop_%s_03_wpKindEmployed.pdf"
                        % (referenceName,), bbox_inches="tight")
    plt.close()

##############
# Households #
##############

# Plot the location of households
    xs = hhs["lon"]
    ys = hhs["lat"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax.set_aspect('equal')
    res = plt.hexbin(xs, ys, cmap=plt.cm.Blues,
            mincnt=1, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar(shrink=.65)
    cbar.set_label("Number of households", size=22)
    cbar.ax.tick_params(labelsize=16)
    geoDataFrame[2].plot(ax=ax, color="none", edgecolor="green", linestyle="--", lw=.5, alpha=.8)
    geoDataFrame[1].plot(ax=ax, color="none", edgecolor="blue", linestyle="-", lw=.5, alpha=.7)
    geoDataFrame[0].plot(ax=ax, color="none", edgecolor="black")

    dx = xs.max() - xs.min()
    dy = ys.max() - ys.min()
    max_dd = max(dx, dy)
    plt.xlim(xs.min()-.25, xs.min()+max_dd+.25)
    plt.ylim(ys.min()-.25, ys.min()+max_dd+.25)
    plt.xlabel("lon", size=22)
    plt.ylabel("lat", size=22)
    plt.xticks(size=16)
    plt.yticks(size=16)

    plt.tight_layout()
    plt.savefig("figures/synPop_%s_04_hhSpatialDistribution.pdf" % (referenceName,),
                bbox_inches="tight")
    plt.close()

# Clustering
    from collections import Counter
    from shapely.geometry import Polygon
    levelsTargetSize = cfg["levelsTargetSize"]
    NlevelsTargetSize = len(levelsTargetSize)
    Ntot_level = NlevelsTargetSize
    levelSizes = {tmp_l:
            Counter([tuple(hh["l%d" %l] for l in xrange(3+tmp_l+1)) for hh in hhs])
            for tmp_l in range(Ntot_level)
            }
    codes = Counter([(h0, h1, h2) for h0, h1, h2 in zip(hhs["l0"], hhs["l1"], hhs["l2"])])
    most_common = codes.most_common(10)[0][0]
    most_common_shape = reference_gdf[reference_gdf.code == most_common].geometry
    print "Most common code:", most_common

    tmp_hhs = hhs[hhs["l0"] == most_common[0]]
    tmp_hhs = tmp_hhs[tmp_hhs["l1"] == most_common[1]]
    tmp_hhs = tmp_hhs[tmp_hhs["l2"] == most_common[2]]
    plt.figure(figsize=(13,4))
    minx, maxx = min(tmp_hhs["lon"]), max(tmp_hhs["lon"])
    miny, maxy = min(tmp_hhs["lat"]), max(tmp_hhs["lat"])
    ddx, ddy = maxx - minx, maxy - miny
    step_dx = 10.**np.floor(np.log10(ddx/4.))
    step_dy = 10.**np.floor(np.log10(ddx/4.))
    minx, maxx = minx - step_dx/5., maxx + step_dx/5.
    miny, maxy = miny - step_dy/5., maxy + step_dy/5.
    bounds_to_plot = reference_gdf[reference_gdf.intersects(
        Polygon([[minx,miny], [minx,maxy], [maxx,maxy], [maxx,miny], [minx,miny]]))]
    for l in range(NlevelsTargetSize):
        ax = plt.subplot(1,NlevelsTargetSize,l+1)
        ax.set_aspect('equal')
        plt.title("Level %d - size %d" % (3+l, levelsTargetSize[l]), size=15)
        plt.scatter(tmp_hhs["lon"], tmp_hhs["lat"], c=tmp_hhs["l%d" % (l+3)])
        bounds_to_plot.plot(ax=ax, color="none", edgecolor="black", linestyle="-", lw=.5, alpha=.8)
        plt.xticks(np.arange(minx-minx%step_dx, maxx, step_dx), size=12, rotation=45, ha="right")
        plt.yticks(np.arange(miny-miny%step_dy, maxy, step_dy), size=12)
        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)
        plt.xlabel("lon", size=14)
        plt.ylabel("lat", size=14)
    plt.tight_layout()
    plt.savefig("figures/synPop_%s_05_clusteringLocal.pdf" % (cfg["referenceName"]), bbox_inches="tight")
    plt.close()

    # Size per level...
    avgHHsize = hhs["size"].mean()
    plt.figure(figsize=(4.5*NlevelsTargetSize, 4))
    for levelID, levelSize in enumerate(levelsTargetSize):
        ax = plt.subplot(1, NlevelsTargetSize, levelID+1)
        ax.set_title("Level %d - Size %d" % (levelID+3, levelSize), size=18)

        Xs = np.array(levelSizes[levelID].values(), dtype=np.float64)
        Xs *= avgHHsize
        r = plt.hist(Xs, rwidth=.95, label="Generated", color="C1")
        ySpan = [0,max(r[0])*1.15]
        plt.plot([levelSize]*2, ySpan, "--C0", lw=6, label="Target")

        plt.xlabel(r"Cluster Size - $s$", size=18)
        plt.ylabel(r"$P(s)$", size=18)
        plt.xticks(size=14, rotation=45, ha="right")
        plt.ylim(ySpan)
    plt.legend(fontsize=14, loc="best")
    plt.tight_layout()
    plt.savefig("figures/synPop_%s_06_clusteringSizePerLeveL.pdf" % (cfg["referenceName"]),
            bbox_inches="tight")
    plt.close()

# ## Household structure
    houseHolds = hhs

# Save the relevant statistics for this area...
# Compute the statistics for the new area...

# The overall age distribution
    sexAgeCDF_array = np.column_stack((ageBySex_CDF[sex].loc[NUTS3code]
                                        for sex in ["male", "female", "total"]))
    sexAgePDF_array = np.column_stack((ageBySex_PDF[sex].loc[NUTS3code]
                                        for sex in ["male", "female", "total"]))

# The household type distribution
    houseHoldTypeCDF = np.array(householdKind_CDF.loc[NUTS3code])
    houseHoldTypePDF = np.array(householdKind_PDF.loc[NUTS3code])

# The size distribution for each household type distribution
    houseHoldType_sizeCDF = np.array([householdSizeByKind_CDF[k].loc[NUTS3code]
                                        for k in householdLabels])
    houseHoldType_sizePDF = np.array([householdSizeByKind_PDF[k].loc[NUTS3code]
                                        for k in householdLabels])

# The age distribution for male and female for parents and children of each household
# type
    agePDFparentSonHHtype = {}
    ageCDFparentSonHHtype = {}
    ageRAWparentSonHHtype = {}

    for hhKind in ageHouseholdLabels:
        agePDFparentSonHHtype[hhKind] = np.column_stack((ageByHHrole_PDF[("male",   hhKind)].loc[NUTS3code],
                                                         ageByHHrole_PDF[("female", hhKind)].loc[NUTS3code],
                                                         ageByHHrole_PDF[("total",  hhKind)].loc[NUTS3code],))

        ageCDFparentSonHHtype[hhKind] = np.column_stack((ageByHHrole_CDF[("male",   hhKind)].loc[NUTS3code],
                                                         ageByHHrole_CDF[("female", hhKind)].loc[NUTS3code],
                                                         ageByHHrole_CDF[("total",  hhKind)].loc[NUTS3code]))
        # The raw numbers
        ageRAWparentSonHHtype[hhKind] = np.column_stack((ageByHHrole_RAW[("male",   hhKind)].loc[NUTS3code],
                                                         ageByHHrole_RAW[("female", hhKind)].loc[NUTS3code]))
        # Put it in a row and divide by sum
        ageRAWparentSonHHtype[hhKind] = ageRAWparentSonHHtype[hhKind].flatten(order="C")
        ageRAWparentSonHHtype[hhKind] /= max(1., ageRAWparentSonHHtype[hhKind].sum())

# Plot the age distribution for the males and females given their role and household status.
# Check that we correctly translated the eurostat weights into probabilities.

    nToPlot = len(ageRAWparentSonHHtype)
    plt.figure(figsize=(4*nToPlot, 4))
    for iii, selectedHH in enumerate(ageRAWparentSonHHtype):
        plt.subplot(1,nToPlot,iii+1)
        plt.title(selectedHH)
        
        # Since in the raw data we are normalizing over male+female here we have to "de-normalize"
        # the PDF of the original distribution by the weights of the male/female part.
        maleWeight = ageRAWparentSonHHtype[selectedHH][::2].sum()
        femaleWeight = ageRAWparentSonHHtype[selectedHH][1::2].sum()
        
        plt.plot(np.arange(0,101), ageRAWparentSonHHtype[selectedHH][::2], "oC1", label="male RAW", lw=2)
        plt.plot(np.arange(0,101), ageRAWparentSonHHtype[selectedHH][1::2], "^C0", label="female RAW", lw=2)
        
        plt.plot(np.arange(0,101,1.), agePDFparentSonHHtype[selectedHH][:,0]*maleWeight, "--C3", label="male PDF", lw=2)
        plt.plot(np.arange(0,101,1.), agePDFparentSonHHtype[selectedHH][:,1]*femaleWeight, "--C9", label="female PDF", lw=2)
        plt.xlabel(r"Age - $a$", size=15)
        plt.ylabel(r"$P(a)$", size=15)
        plt.xticks(size=12)
        plt.yticks(size=12)

    plt.legend(fontsize=12)
    plt.tight_layout()
    #plt.savefig("figures/synPop_%s_rawVsDerivedAgePDF.pdf" % (referenceName,),
    #                bbox_inches="tight")
    plt.close()

# In[ ]:


# Household type frequency
    plt.figure(figsize=(5,4))
    bins = np.arange(-.5, 7.5, 1)
    plt.hist(hhs["kind"], bins=bins,
             density=True, rwidth=.75, label="Generated", color="C1")
    plt.plot(np.arange(len(houseHoldTypePDF)), houseHoldTypePDF,
             "o-C0", ms=14, lw=2, label="Actual data")

    locs = [h["id"] for k, h in sorted(houseHoldTypeDict.iteritems())]
    labs = [k       for k, h in sorted(houseHoldTypeDict.iteritems())]
    plt.xticks(locs, labs, size=16, rotation=45, ha="right")
    plt.yticks(size=16)

    plt.xlim(-.75, bins[-1])
    plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=[.775, .975])

    plt.xlabel(r"Household type - $h$", size=18);
    plt.ylabel(r"$P(h)$", size=18);

    plt.tight_layout()
    plt.savefig("figures/synPop_%s_07_hhKindDistribution.pdf" % (referenceName), bbox_inches="tight")
    plt.close()

# Household size frequency per hh type
    nHouseholdKinds = len(houseHoldTypeDict)

    nCols = 3
    nRows = nHouseholdKinds // nCols
    fig, ax = plt.subplots(nrows=nRows, ncols=nCols, sharex=True, figsize=(4*nCols,3*nCols))


    for i, hhName in enumerate(houseHoldTypeDict):
        plt.subplot(nRows,nCols, (i/nCols)*nCols +  i%nCols+1)
        plt.title(hhName, size=18)
        selectedHHtype = i
        plt.hist(hhs[hhs["kind"] == selectedHHtype]["size"],
                 bins=np.arange(-.5, 12.5, 1.), density=True, label="Generated",
                 rwidth=.75, color="C1")
        plt.plot(np.arange(len(houseHoldType_sizeCDF[selectedHHtype]))+1., houseHoldType_sizePDF[selectedHHtype],
                 "o-C0", label="Actual data", ms=14, lw=2)
        plt.xticks(range(0,12), size=16)
        plt.yticks(size=16)
        plt.xlim(.25, 11.75)
    plt.legend(fontsize=16, loc="best")

    fig.text(.5, -.02, r"Members - $m$", size=22, ha="center")
    fig.text(-.02, .5, r"$P(m)$", size=22, va="center", rotation="vertical")

    plt.tight_layout()
    plt.savefig("figures/synPop_%s_08_hhSizePerKind.pdf" % (referenceName), bbox_inches="tight")
    plt.close()

# Overall age in the whole population
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,4))

    binwidth = 2
    plotCount = 1
    for selectedSexes, sexName in zip([[0], [1], [0,1]], ["Male", "Female", "Total"]):
        plt.subplot(1,3,plotCount)
        plt.title(sexName, size=16)
        plt.hist(ags["age"], bins=np.arange(0,103,binwidth), normed=True,
                label="Generated", rwidth=.9, color="C1");
        plt.plot(np.arange(len(sexAgePDF_array[:,0])), sexAgePDF_array[:,0], "o-C0",
                label="Actual data", lw=3)

        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.xlabel(r"Age - $a$", size=16)
        if plotCount == 1:
            plt.ylabel(r"$P(a)$", size=16)
        plotCount += 1
    plt.legend(fontsize=16, loc="upper left", bbox_to_anchor=[.7, 1.])
    plt.tight_layout()
    plt.savefig("figures/synPop_%s_09_agePopulationPerSex.pdf" % (referenceName,), bbox_inches="tight")
    plt.close()

# Age distribution for household kind and role covered by person.
    dictRole = {0: "Children", 1: "Parent"}
    dictSex = {0: "male", 1: "female"}

###################
    ncols = len(houseHoldTypeDict)
    nrows = len(dictSex)*len(dictRole)
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows))

    iii = 0
    for selectedRole, roleLabel in dictRole.iteritems():
        for selectedSex, sexLabel in dictSex.iteritems():
            for householdType, houseHoldData in houseHoldTypeDict.iteritems():
                iii += 1
                plt.subplot(nrows, ncols, iii)
                plt.title(householdType, size=20)

                plt.xticks(size=16)
                plt.yticks(size=16)

                if iii % ncols == 1:
                    plt.ylabel(r"$P(a)$", size=18)
                if (iii-1) // ncols >= nrows -1:
                    plt.xlabel(r"Age - $a$", size=18)

                plt.xlim(-.5, 104)

                selectedHHkind = houseHoldData["id"]
                ages = np.array([ag["age"]
                        for ag in ags[
                                    (ags["sex"] == selectedSex) &
                                    (ags["role"] == selectedRole)
                                ]
                            if hhs[ag["hh"]]["kind"] == houseHoldData["id"]
                    ])
                if len(ages) < 2: continue

                lGen = plt.hist(ages, bins=np.arange(0, 102, 2), normed=True, rwidth=.9, label="Generated", color="C1");

                roleSexHH2label = {
                                (1,0,"M1_CH"): "A1_XCH",
                                (1,1,"F1_CH"): "A1_XCH",
                                (1,0,"MULTI_HH"): "A1_HH",
                                (1,1,"MULTI_HH"): "A1_HH",
                                (1,0,"CPL_WCH"): "CPL_XCH",
                                (1,1,"CPL_WCH"): "CPL_XCH",
                                (1,0,"A1_HH"): "A1_HH",
                                (1,1,"A1_HH"): "A1_HH",
                                (1,0,"CPL_NCH"): "CPL_XCH",
                                (1,1,"CPL_NCH"): "CPL_XCH",
                                (0,0,"M1_CH"): "CH_PAR",
                                (0,1,"M1_CH"): "CH_PAR",
                                (0,0,"F1_CH"): "CH_PAR",
                                (0,1,"F1_CH"): "CH_PAR",
                                (0,0,"MULTI_HH"): "CH_PAR",
                                (0,1,"MULTI_HH"): "CH_PAR",
                                (0,0,"CPL_WCH"): "CH_PAR",
                                (0,1,"CPL_WCH"): "CH_PAR",
                            }
                try:
                    lEmp = plt.plot(agePDFparentSonHHtype[roleSexHH2label[(selectedRole,selectedSex,householdType)]][:,selectedSex], "o-C0",
                         label="Actual data", lw=3)
                except:
                    pass

            fig.text(.5, .9995-selectedRole*.5-selectedSex*.25, roleLabel + " - " + sexLabel, size=22, ha="center")

    from matplotlib.patches import mlines
    empPatch = mlines.Line2D([], [], linestyle="", marker="s", markersize=10, color="C0", label="Actual data")
    genPatch = mlines.Line2D([], [], linestyle="", marker="s", markersize=10, color="C1", label="Generated")
    fig.legend(handles=[empPatch, genPatch], fontsize=20, loc="upper left", bbox_to_anchor=[.9, 1.065])
    plt.tight_layout(h_pad=4.)
    plt.savefig("figures/synPop_%s_10_agePerRole.pdf" % (referenceName,), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    cfg_module = sys.argv[1].split(".")[0]
    checkPopulation(cfgname=cfg_module)
