#  <p><div class="lev1 toc-item"><a href="#Overview" data-toc-modified-id="Overview-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Overview</a></div><div class="lev2 toc-item"><a href="#Input" data-toc-modified-id="Input-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Input</a></div><div class="lev3 toc-item"><a href="#Population-structure" data-toc-modified-id="Population-structure-111"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Population structure</a></div><div class="lev3 toc-item"><a href="#Work-and-education" data-toc-modified-id="Work-and-education-112"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Work and education</a></div><div class="lev3 toc-item"><a href="#Geodataframe" data-toc-modified-id="Geodataframe-113"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Geodataframe</a></div><div class="lev2 toc-item"><a href="#Create-the-households" data-toc-modified-id="Create-the-households-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Create the households</a></div><div class="lev2 toc-item"><a href="#Here-we-define-the-households" data-toc-modified-id="Here-we-define-the-households-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Here we define the households</a></div><div class="lev2 toc-item"><a href="#Education-and-work" data-toc-modified-id="Education-and-work-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Education and work</a></div><div class="lev2 toc-item"><a href="#Commuting" data-toc-modified-id="Commuting-15"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Commuting</a></div><div class="lev2 toc-item"><a href="#Create-the-population" data-toc-modified-id="Create-the-population-16"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Create the population</a></div><div class="lev2 toc-item"><a href="#Plot-the-location-of-households" data-toc-modified-id="Plot-the-location-of-households-17"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Plot the location of households</a></div><div class="lev2 toc-item"><a href="#Workplaces-creation" data-toc-modified-id="Workplaces-creation-18"><span class="toc-item-num">1.8&nbsp;&nbsp;</span>Workplaces creation</a></div><div class="lev2 toc-item"><a href="#Create-communities" data-toc-modified-id="Create-communities-19"><span class="toc-item-num">1.9&nbsp;&nbsp;</span>Create communities</a></div><div class="lev3 toc-item"><a href="#Demonstrate-how-it-will-work" data-toc-modified-id="Demonstrate-how-it-will-work-191"><span class="toc-item-num">1.9.1&nbsp;&nbsp;</span>Demonstrate how it will work</a></div><div class="lev3 toc-item"><a href="#Alternatively,-KMeans" data-toc-modified-id="Alternatively,-KMeans-192"><span class="toc-item-num">1.9.2&nbsp;&nbsp;</span>Alternatively, KMeans</a></div><div class="lev3 toc-item"><a href="#Comments" data-toc-modified-id="Comments-193"><span class="toc-item-num">1.9.3&nbsp;&nbsp;</span>Comments</a></div><div class="lev3 toc-item"><a href="#Compute-and-append-the-new-codes-to-the-old-ones" data-toc-modified-id="Compute-and-append-the-new-codes-to-the-old-ones-194"><span class="toc-item-num">1.9.4&nbsp;&nbsp;</span>Compute and append the new codes to the old ones</a></div><div class="lev2 toc-item"><a href="#Split-back-the-dataframe-and-create-the-agents-one" data-toc-modified-id="Split-back-the-dataframe-and-create-the-agents-one-110"><span class="toc-item-num">1.10&nbsp;&nbsp;</span>Split back the dataframe and create the agents one</a></div><div class="lev2 toc-item"><a href="#Save-to" data-toc-modified-id="Save-to-111"><span class="toc-item-num">1.11&nbsp;&nbsp;</span>Save to</a></div><div class="lev2 toc-item"><a href="#Define-the-types" data-toc-modified-id="Define-the-types-112"><span class="toc-item-num">1.12&nbsp;&nbsp;</span>Define the types</a></div><div class="lev2 toc-item"><a href="#Demography" data-toc-modified-id="Demography-113"><span class="toc-item-num">1.13&nbsp;&nbsp;</span>Demography</a></div><div class="lev1 toc-item"><a href="#Check-the-generated-population" data-toc-modified-id="Check-the-generated-population-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Check the generated population</a></div><div class="lev2 toc-item"><a href="#Workplaces-size-distribution" data-toc-modified-id="Workplaces-size-distribution-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Workplaces size distribution</a></div><div class="lev2 toc-item"><a href="#Household-structure" data-toc-modified-id="Household-structure-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Household structure</a></div><div class="lev2 toc-item"><a href="#Commuting" data-toc-modified-id="Commuting-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Commuting</a></div>
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pickle, gzip
import scipy
from scipy.stats import norm
import geojson
import numpy as np
import datetime
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
        agentSchoolEduEmploy, getEducationRate, filterL0codes, universityBins, universityPDF

# In this notebook we show how to generate the households of our synthetic population using the pre-processed Eurostat, ISTAT and geo data.
# 
# These steps are presented one by one and their only purpose is to explain what happens internally in the synpop generator.
# 
# We are going to load the tables containing the following information:
# 
# - regarding household structure:
#     - $P^{(NUTS)}(a|s)$ the age $a$ distribution for people living in a given $NUTS$ area given their sex $s$;
#     - $P^{(NUTS)}(h)$ the probability to have an household of kind $h$ for each $NUTS$ code;
#     - $P^{(NUTS)}(m|h)$ the probability, for each $NUTS$ code, for an household to have $m$ members given the household kind $h$;
#     - $P^{(NUTS)}(a|r,h)$ the probability, for each $NUTS$ code, for an household member to have age $a$ given that he is covering the role $r$ (i.e., parent/adult or children) in an household of kind $h$;
# - regarding education and employment:
#     - the size distribution of schools and workplaces;
#     - the probability to attend school at a given age by NUTS2;
#     - the probability to have a certain education level given the sex and age by NUTS2;
#     - the probability to be employed given sex, age and education level for NUTS2;
#     - the commuting probability to other municipalities at given NUTS3 level;
#     - we will put a hand-made income distribution as we did not find data on this;
# - regarding the geographic and administrative structure:
#     - the geodataframe containing the NUTS + LAU1/2 codes of the areas, their extent and their codes;
#     - a connection to the geodatabase;
#     
# ## Input
# 
# Here we set the location of the input files (pickled dataframes) and we set the global variables.
# Population structure
def generateEntities(cfgname, additionalArgs={}):
    cfg_module = importlib.import_module(cfgname)
    cfg = cfg_module.cfg
    cfg.update(additionalArgs)

    print("Loading demography data...")
# The dataframe containing the age distribution (PDF and CDF) of the population per sex for all the NUTS0-3
    ageBySex_PDF = pd.read_pickle(cfg["ageBySex_PDF_file"])
    ageBySex_CDF = pd.read_pickle(cfg["ageBySex_CDF_file"])

# We also load the data for nuts3 levels (we use them to evaluate the number of
# agents to create)
    popBroadAgeBySex_NUTS3 = pd.read_pickle(cfg["popBroadAgeBySex_NUTS3_file"])

    print("Loading Eurostat data...")
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

    print("Loading commuting data...")
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

    print("Loading boundaries data...")
# Geodataframe
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

# Create the households
# Here, for each boundary at the finest resolution level (municipalities and/or
# districts of large cities) we create the population residing there.
# First, we have to define generate the possible kind of households.
# Each household accept some constraint on the age of the parents (adults) and sons
# (children), the difference in age etc; moreover each household type accepts a cdf
# for the age of the adults and one for the age of the children.
#
# The household class and the relevant functions are in `synpopStructures`.
#
# We will also use utilities from the `synpopUtils` module and a connection to the
# geodatabase.
# The `synpopStructures` expects raw counters of houses to be passed as pdf.
# In this we alternate the counter of males and females having a given age and participating to a given household kind.
# For example:
#
# | $m_0$ | $f_0$ | $m_1$ | $f_1$ | $m_2$ | $f_2$ | ... | ... | $m_{100}$ | $f_{100}$ |
#
# The procedure is simple:
# - for each level5 code area within the geodataframe:
#     - fetch statistics of that area on the household structure and age distribution of components;
#     - create the age PDF and the household size/kind distribution accordingly;
#     - aggregate the population found in each SEDAC cell in the selected area;
#     - create households until population of cells is reproduced;
#     - put each household on a random (lon, lat) location within the area proportionally to population;
#     - append households to general list;
#
# Once we finish we have a list of households made like this:
# ```
# ((hhID, hhKind, hhSize), (lon,lat), (nuts3, LAU1, LAU2))
# ```
# and an agents list made like this:
# ```
# [(id, role, sex, age)_0, (id, role, sex, age)_1, ...]
# ```
# We will then assign an education/employment/income, workplace (school) and commuting based on age and sex of each agent accordingly to the given tables.

# Here we define the households
#
# Each household has a structure given by the number of parents, number of sons and
# age constraints between the parent/son.
# We will inform these structures with the age PDF for each NUTS code during the
# household creation.
# Household labels: the ones for the age structure and the ones for the actual
# household kinds
    print("Defining household structures...")
    ageHouseholdLabels = set(ageByHHrole_RAW.columns.get_level_values(1).unique())
    ageHouseholdLabels.discard("TOTAL")
    householdLabels = set(householdKind_PDF.columns)
    householdLabels.discard("TOTAL")

    referenceName = cfg["referenceName"]

# Couple without children
    CPL_NCH = synpopStructures.householdType(
            minMaxParents=(2,2), minMaxSons=(0,0), ageMinMaxParents=(18,100),
            sexParents="etero", agePDFparents=None, agePDFsons=None)

# Couple with young and/or old children
    CPL_WCH = synpopStructures.householdType(
            minMaxParents=(2,2), minMaxSons=(1,9),
            ageMinMaxParents=(18,100), ageMinMaxChildren=(0, 80),
            dMinMaxParSon=(18,50), sexParents="etero", sexSons="free",
            agePDFparents=None, agePDFsons=None)

# Lone father/mother with young/old children
    M1_CH  = synpopStructures.householdType(
            minMaxParents=(1,1), minMaxSons=(1,10),
            ageMinMaxParents=(18,100), ageMinMaxChildren=(0,80),
            dMinMaxParSon=(18,50), sexParents="male", sexSons="free",
            agePDFparents=None, agePDFsons=None)

    F1_CH  = synpopStructures.householdType(
            minMaxParents=(1,1), minMaxSons=(1,10), ageMinMaxParents=(18,100),
            ageMinMaxChildren=(0,80), dMinMaxParSon=(18,50), sexParents="female",
            sexSons="free", agePDFparents=None, agePDFsons=None)

# Singles and multihouseholds share the same age distribution (but different size)
    A1_HH  = synpopStructures.householdType(
            minMaxParents=(1,1), minMaxSons=(0,0),
            ageMinMaxParents=(15, 100), sexParents="free",
            agePDFparents=None, agePDFsons=None)

    MULTI_HH  = synpopStructures.householdType(
            minMaxParents=(2,11), minMaxSons=(0,0),
            ageMinMaxParents=(15, 100), sexParents="free",
            dMinMaxp1p2=(0, 40), dMinMaxParSon=(30, 100),
            fixedParentsSons=(False, True), agePDFparents=None, agePDFsons=None)

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

# ## Education and work
# Here, using the data on education and employment we define if each agent is attending school, working and which education level it has.
# We adopt the following assumptions:
# - we adopt 4 schools levels:
#     - 0: pre-primary (no primary school), [0, 5) years;
#     - 1: primary school [6, 10) years old;
#     - 2: secondary school [11, 18) years old;
#     - 3: tertiary (university) [18, 25) y.o..
# - we assume 3 education levels:
#     - 0: agent completed pre-primary and/or primary;
#     - 1: agent completed secondary school;
#     - 2: agent completed university.
# - agents younger than 18 years old that are not following school have education level equal to the previous one of their age (e.g. a 16 y.o. children not attending school has a primary education level);
# - workplaces and schools are treated as equal: schools are just workplaces with kind id from 0 to 9 (included), while workplaces have kind id >= 10.
# - we assume only one workplace id for real workplaces, id=10.
# - due to lack of data we assign an income to the non students employed as follows:
#     - if edu = 0 income = Normal(15000, 1000);
#     - if edu = 1 income = Normal(25000, 1500);
#     - if edu = 2 income = Normal(35000, 2000);
#     - if age > 65 and not employed assign an income as in edu = 0.
# 
# The procedure, for each agent, is as follows:
# - decide if agent goes to school or not:
# - if yes, set education to previous level and workplace kind to corresponding school level code (e.g. a 22 y.o. guy going to the University has education level 1=secondary school and workplace id 3=university).
# - if not:
#     - if age < 18 set education to either 0 or 1, depending on age;
#     - else if age >= 18 draw education from education table;
#     - then decide whether the agent is working or not accordingly to the employment rate per sex, age and education level.

# ## Commuting
# 
# Now for each LAU2 code we compute its baricenter (in terms of population) and compute the matrix of distances using the *haversine* distance.
# 
# We end up with $d_{ij}$ setting the distance (in km) between LAU2 code $i$ and $j$.
# 
# From this matrix and from the population $n_i$ of each LAU2 code we compute the origin destination matrix $C_{ij}$ setting the probability to commute from municipality $i$ to $j$. The matrix is derived from the gravity model:
# 
# $C_{ij} = \theta \frac{n_i^{\tau_f}n_j^{\tau_t}}{d_{ij}^\rho},\;\; \rm{if } i\neq j$
# 
# We use $\tau_f=0.28$, $\tau_t=0.66$ and $\rho=2.95$ as found in Ajelli et al. 2010.
# 
# The $\theta$ proportionality constant is set so that the overall commuting probability to other municipalities equals the fraction of people in the area moving outside of teh municipality for work/study.
# 
# In particular, from the commuting table we know the probability to commute to the same municipality $p_{sMun} = 1 - p_{dMun}$, so that:
# 
# $\sum_{i \neq j}C_{ij} = p_{dMun} = 1 - p_{sMun}$,
# 
# and
# 
# $C_{ii} = p_{sMun}$
# 
# We will define one matrix of commuting for each kind of school/workplace applying the following assumption as in Ajelli et al.:
# - for kindergarten, primary and secondary schools we have a maximum travelling distance of 100 km;
# - kindergarten and primary schools may be found in every municipality, whereas secondary schools are based in municipalities of at least 15000 people. Universities are found in cityes of at least $10^5$ people. If no cities like this are found we assign universities to the top populated area.

    print("Computing the commuting distances...")
    tic = datetime.datetime.now()
    minimumPopulationPerWPkind = cfg["minimumPopulationPerWPkind"]
    maximumDistancePerWPkind = cfg["maximumDistancePerWPkind"]

# Compute the distance matrix...
    baricenters_LatLon = np.array(reference_gdf[["BARICENTER_Y", "BARICENTER_X"]])
    baricenters_distanceM = squareform(pdist(baricenters_LatLon, metric=haversine))

# Gravity model:
    tau_f = cfg["tau_f"]
    tau_t = cfg["tau_t"]
    rho_exp = cfg["rho_exp"]

    populationI = np.array(reference_gdf["POP"])
    topPopulatedArea = np.argmax(populationI)

    fractionToSameMunicipalityStudy = studyCommuting_df["PDF", "sMun"]
    fractionToSameMunicipalityWork = workCommuting_df["PDF", "sMun"]

    Commuting_ij = {wp_k: None for wp_k in minimumPopulationPerWPkind.keys()}

    for wp_kind, _ in Commuting_ij.iteritems():
        sys.stdout.write("\rFor wp kind %d..." % wp_kind)
        sys.stdout.flush()
        tmp_commutingM = np.zeros(baricenters_distanceM.shape, dtype=float)
        minTargetPop = minimumPopulationPerWPkind[wp_kind]
        maxTargetDist = maximumDistancePerWPkind[wp_kind]
        for source in xrange(baricenters_distanceM.shape[0]):
            source_pop = populationI[source]
            if source_pop == 0:
                continue
            NUTS3_code = geoDFid2nuts[reference_gdf.iloc[source]["code"][0]]
            if wp_kind < 10:
                fraction_to_same = fractionToSameMunicipalityStudy[NUTS3_code]
            else:
                fraction_to_same = fractionToSameMunicipalityWork[NUTS3_code]

            for target in xrange(baricenters_distanceM.shape[1]):
                target_pop = populationI[target]
                distance_ij = baricenters_distanceM[source, target]
                if target_pop < minTargetPop or distance_ij > maxTargetDist:
                    continue
                if source == target:
                    tmp_commutingM[source, target] = fraction_to_same
                else:
                    tmp_commutingM[source, target] =\
                            (source_pop**tau_f) * (target_pop**tau_t)\
                            / max(1., distance_ij**rho_exp)
            # Normalize row...
            tmp_row_sum = tmp_commutingM[source, :].sum()
            if tmp_row_sum == .0:
                # No destinations for this wp kind, asssigning to top populated one...
                tmp_commutingM[source, topPopulatedArea] = 1.
            else:
                # Normalize and restore diagonal...
                # Fraction to same may be 0 because of municipality size (especially for
                # unis)!
                fraction_to_same = tmp_commutingM[source, source]
                amountToOthers = tmp_row_sum - fraction_to_same
                tmp_commutingM[source, :] *= (1. - fraction_to_same)/amountToOthers
                tmp_commutingM[source, source] = fraction_to_same

            assert np.isclose(tmp_commutingM[source,:].sum(), 1.),\
                    "Sum was %f code %r row %r" % (tmp_sum, NUTS3_code, source_pop)
        Commuting_ij[wp_kind] = tmp_commutingM

# Create the CDF for each wp kind and row...
    Commuting_ij_CDF = {k: v.cumsum(axis=1) for k,v in Commuting_ij.iteritems()}

# Origin dest matrix per wp kind
    for wpKindSelected in minimumPopulationPerWPkind.keys():
        plt.imshow(Commuting_ij[wpKindSelected], norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=1.),
                cmap=plt.cm.Blues)
        cbar = plt.colorbar(shrink=.85)
        cbar.set_label(r"Fraction of commuters - $C_{ij}$", size=22)
        cbar.ax.tick_params(labelsize=16)
        plt.xlabel(r"Destination - $j$", size=18)
        plt.ylabel(r"Origin - $i$", size=18)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.tight_layout()
        plt.savefig("figures/synPop_%s_00_doriginDestinationMatrix_wpkind%02d.pdf" %
                        (referenceName, wpKindSelected),
                        bbox_inches="tight")
        plt.close()

    if True:
        selectedWorkplaceKind = 10
        selectedODmatrix = Commuting_ij[selectedWorkplaceKind]
        print("\nSaving commuting data...")
        mostPopulatedIndex = np.argmax(reference_gdf["POP"].values)
        volumeToTarget = reference_gdf["POP"]*selectedODmatrix[:,mostPopulatedIndex]
        maxVolumeToTarget = max(volumeToTarget)

        import networkx as nx
        G = nx.Graph()
        edges = [(i, mostPopulatedIndex, {"weight": vol}) for i, vol in enumerate(volumeToTarget)]
        nodes = xrange(selectedODmatrix.shape[0])
        G.add_edges_from(edges)
        XXs = reference_gdf["BARICENTER_X"].values
        YYs = reference_gdf["BARICENTER_Y"].values
        pos = {i: (x, y) for i, (x, y) in enumerate(zip(XXs, YYs))}
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect("equal")
        reference_gdf_tot.plot(ax=ax, color="none", edgecolor="black", alpha=.8)
        nx.draw_networkx_nodes(G,pos, node_color='C0', node_size=300.*reference_gdf["POP"].values/max(reference_gdf["POP"]))
        nx.draw_networkx_edges(G,pos, node_color='C0', node_size=200, width=[100.*e[2]['weight']/maxVolumeToTarget for e in edges], edge_color='C1')
        minx, maxx = min(XXs), max(XXs)
        miny, maxy = min(YYs), max(YYs)
        dxx, dxy = maxx-minx, maxy-miny
        plt.xlim(minx-dxx/10., maxx+dxx/10.)
        plt.ylim(miny-dxy/10., maxy+dxy/10.)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig("figures/synPop_%s_00_network2mostPop_wpkind%02d.pdf" %
                        (referenceName, selectedWorkplaceKind),
                        bbox_inches="tight")
        plt.close()


        #commuting_dict = {"matrices": Commuting_ij, "lat_lon": reference_gdf[["BARICENTER_Y", "BARICENTER_X"]]}
        toc = datetime.datetime.now()
    print("\nCommuting done in %f seconds..." % (toc-tic).total_seconds())

# Create the population
    print("Creating population...")
    tic = datetime.datetime.now()
# Scale of the population to save
# Fraction between 0 and 1 
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

# Here we set the counters, the container of the generated households
    tot_pop, tot_generated_pop = 0, 0
    generatedHouseholds = []
    generatedAgents = []

# Since we have a lot of municipalities under the same
# NUTS3 we save the last seen NUTS level from which we
# computed the household stats to save time
    lastStatisticNUTS = None
    lastEducationNUTS = None

# While I am here I also save the set of agents working in
# each workplace kind for each LAU2 index...
    workersIdPerLAU = {lau:
                           {kind: set() for kind in minimumPopulationPerWPkind}
                       for lau in reference_gdf.index
                      }

# We have to keep track of the generated agents and households from one area to the
# other
    incremental_hh_id = 0
    incremental_ag_id = 0

    counter, total = 1,  reference_gdf.shape[0]
    LAU2_df_iIndex = -1
    for LAU2_index, LAU2_data in reference_gdf.iterrows():
        LAU2_df_iIndex += 1
        tmp_code = LAU2_data["code"]
        tmp_nuts3_id = tmp_code[0]
        tmp_nuts3_code = geoDFid2nuts[tmp_nuts3_id]

        # Look for the code that has statistics for this shape...
        # We need the hh size, kind and component age distributions.
        statisticsNuts = tmp_nuts3_code
        while statisticsNuts not in householdKind_CDF.index:
            statisticsNuts = statisticsNuts[:-1]
        if statisticsNuts != lastStatisticNUTS:
            lastStatisticNUTS = statisticsNuts
            # Compute the statistics for the new area...
            # The household type distribution
            houseHoldTypeCDF = np.array(householdKind_CDF.loc[statisticsNuts])
            houseHoldTypePDF = np.array(householdKind_PDF.loc[statisticsNuts])

            # The size distribution for each household type distribution
            houseHoldType_sizeCDF = np.array(
                    [householdSizeByKind_CDF[k].loc[statisticsNuts]
                        for k in householdLabels])
            houseHoldType_sizePDF = np.array(
                    [householdSizeByKind_PDF[k].loc[statisticsNuts]
                        for k in householdLabels])

            # The age distribution for male and female for parents and children of each
            # household type
            ageRAWparentSonHHtype = {}
            for hhKind in ageHouseholdLabels:
                # The raw numbers
                tmp_arra = np.column_stack(
                        (ageByHHrole_RAW[("male",   hhKind)].loc[statisticsNuts],
                          ageByHHrole_RAW[("female", hhKind)].loc[statisticsNuts]))
                # Put it in a row and divide by sum...
                tmp_arra = tmp_arra.flatten(order="C")
                ageRAWparentSonHHtype[hhKind] = tmp_arra/max(1., tmp_arra.sum())

            # Inform the household about the newly computed age pdf...
            for hhName, hhData in houseHoldTypeDict.iteritems():
                tmp_obj = hhData["obj"]
                par_name = hhData["parentAgePDFName"]
                if par_name:
                    tmp_obj.setAgePDFparents(ageRAWparentSonHHtype[par_name])
                son_name = hhData["childsAgePDFName"]
                if son_name:
                    tmp_obj.setAgePDFsons(ageRAWparentSonHHtype[son_name])
        educationNUTS = tmp_nuts3_code
        while educationNUTS not in schoolAttendanceRate_df["2013"].index:
            educationNUTS = educationNUTS[:-1]
        if educationNUTS != lastEducationNUTS:
            # Save info about education and employment
            eduRate = getEducationRate(
                        schoolAttendanceRate_df["2013"].loc[educationNUTS])
            eduLevelCDF = getEducationLevelCDF(
                    educationLevelByAge_CDF["2011"].loc[educationNUTS])
            employPerSexAgeEdu = getEmploymentProba(
                    employmentBySexAgeEdu_df["2011"].loc[educationNUTS])

        # Setting reference to the cells CDF and their shapes...
        cellsCDF = LAU2_data["CELLS_CDF"]
        if len(cellsCDF) < 1:
            print "\nNo cells found within code %s name %s" %\
                    (tmp_code, LAU2_data["name"])
            continue
        cellsIntersectionSHP = LAU2_data["CELLS_SHP"]
        popToGenerate = int(LAU2_data["POP"]*popScale)
        if popToGenerate == 0:
            continue
        # The households for the NUTS3. Since we do not have eurostat data at this level
        # we use the sedac cells for an estimation of the population...
        tmp_hhs, tmp_agents = synpopStructures.generatePopulation(
               houseHoldType=houseHoldTypeArray, houseHoldTypeCDF=houseHoldTypeCDF,
               houseHoldType_sizeCDF=houseHoldType_sizeCDF, size=popToGenerate,
               offset_agID=incremental_ag_id, offset_hhID=incremental_hh_id
            )
        incremental_hh_id += len(tmp_hhs)
        incremental_ag_id += len(tmp_agents)

        tmp_gen_pop = len(tmp_agents)
        while tmp_hhs:
            # Now assign each household to the corresponding cell proportionally to the
            # population density.
            tmp_hh = tmp_hhs.pop(0)
            point = synpopUtils.assignToCell(cellsCDF=cellsCDF,
                                             cellsIntersectionSHP=cellsIntersectionSHP)
            tmp_hh = tmp_hh + [point.x, point.y, tmp_code]
            generatedHouseholds.append(tmp_hh)

        # Compute edu, income and employment for current agents...
        # Then, if employed, assign to a destination based on the
        # origin destination matrix...
        tmp_agents = [agentSchoolEduEmploy(
                          agent=agent, eduRate=eduRate, eduLevel=eduLevelCDF,
                          employAgeSexEduRate=employPerSexAgeEdu)
                      for agent in tmp_agents]
        for agent in tmp_agents:
            tmp_wp_kind = agent[agentIndex_wpkind]
            if tmp_wp_kind < 0:
                agent.append(-1)
            else:
                selected_destination = np.argmax(
                          np.random.rand()
                          < Commuting_ij_CDF[tmp_wp_kind][LAU2_df_iIndex,:])
                selected_destination_index = reference_gdf.index[selected_destination]
                # Save the id in the corresponding set and append the workplace LAU id to
                # the agent (will be later substituted with the wpid)...
                workersIdPerLAU[selected_destination_index][tmp_wp_kind].add(
                                        agent[agentIndex_id])
                agent.append(selected_destination_index)
        generatedAgents.extend(tmp_agents)
        tot_pop += popToGenerate
        tot_generated_pop += tmp_gen_pop
        sys.stdout.write("\r%04d / %04d: stat NUTS %r - Code %r - Name %s"
            % (counter, total, statisticsNuts,  tmp_code, LAU2_data["name"]) + " "*30)
        sys.stdout.flush()
        counter += 1
    print "\nTOT", tot_pop, "GEN", tot_generated_pop
    print "\nDone!"
    toc = datetime.datetime.now()
    print("Population created in in %f seconds..." % (toc-tic).total_seconds())

# Workplaces creation
# Now for each LAU we collect the workers of that lad (for each workplace kind) and
# we generate a workplace
# Define the workplaces
# For universities (wp=3) we have a unique, empirical size distribution N(2500, 100).

    print("Creating workplaces...")
    tic = datetime.datetime.now()
    workplacesDict = {k: synpopStructures.workplace(k) for k in
            minimumPopulationPerWPkind}

    universityDistrib = {"BINS": universityBins,
                     "CDF": universityPDF.cumsum()/universityPDF.sum(),
                     "PDF": universityPDF,
                    }
    workplacesDict[3].set_sizePDF(universityDistrib)

# The last nuts from which I took stats
# the incremental counter of the wp id and
# the dict {kind: {lau2: [s0, s1, s2, ..., sn]}} of the size of
# the created workplaces...
    lastStatisticNUTSschool = None
    lastStatisticNUTSwork = None
    incremental_wp_id = 0
    plainGeneratedWorkplaces = {
            k: {LAU2_code: []
                    for LAU2_code in reference_gdf.index}
            for k in minimumPopulationPerWPkind
        }
    nWorkersToAssign = {
            lau2code:
                    {k: len(v) for k, v in lau2data.iteritems()}
            for lau2code, lau2data in workersIdPerLAU.iteritems()
        }

    # Compute once and for all the LAU2 <-> statistical NUTS mapping for speedup
    statisticalNUTSperLAU2 = {}

    for LAU2_code, LAU2_data in reference_gdf.iterrows():
        LAU2_geoCode = LAU2_data["code"]
        NUTS3_index = LAU2_geoCode[0]
        NUTS3_code = geoDFid2nuts[NUTS3_index]

        # Update information on size CDF of school and workplaces...
        statisticsNutsSchool = NUTS3_code
        while statisticsNutsSchool not in schoolSize_df.index:
            statisticsNutsSchool = statisticsNutsSchool[:-1]

        statisticsNutsWork = NUTS3_code
        while statisticsNutsWork not in workplSize_df.index:
            statisticsNutsWork = statisticsNutsWork[:-1]

        statisticalNUTSperLAU2[LAU2_code] = {\
                    "school": statisticsNutsSchool,
                    "workpl": statisticsNutsWork,
                }

    for wp_kind in minimumPopulationPerWPkind:
        tmp_generatedWPdict = plainGeneratedWorkplaces[wp_kind]
        for LAU2_code, LAU2_data in reference_gdf.iterrows():
            nLeftToAssign = nWorkersToAssign[LAU2_code][wp_kind]
            if nLeftToAssign <= 0: continue
            LAU2_geoCode = LAU2_data["code"]
            NUTS3_index = LAU2_geoCode[0]
            NUTS3_code = geoDFid2nuts[NUTS3_index]

            if wp_kind < 10:
                workplacesDict[wp_kind].set_sizePDF(
                        schoolSize_df.loc[
                            statisticalNUTSperLAU2[LAU2_code]["school"]
                        ][["BINS", "PDF"]]
                    )
            else:
                workplacesDict[wp_kind].set_sizePDF(
                        workplSize_df.loc[
                            statisticalNUTSperLAU2[LAU2_code]["workpl"]
                        ][["BINS", "PDF"]]
                    )

            while nLeftToAssign > 0:
                tmp_generatedWP = workplacesDict[wp_kind].makeOne(
                            incremental_wp_id, maxSize=nLeftToAssign)
                incremental_wp_id += 1
                tmp_generatedWPdict[LAU2_code].append(tmp_generatedWP)
                nLeftToAssign -= tmp_generatedWP[wpIndex_size]
        sys.stdout.write("Generated wps for wp kind %d\n" % wp_kind)
        sys.stdout.flush()

    # Now that we generated the wp we assign workers to them and arrange them in
    # space
    sys.stdout.write("Assigning workers to workplaces...\n")
    sys.stdout.flush()
    iii = 0
    generatedWorkplaces = []
    for LAU2_code, LAU2_data in reference_gdf.iterrows():
        # Retrieve cells CDF and shapes to place workplaces in space...
        LAU2_geoCode = LAU2_data["code"]
        cellsCDF = LAU2_data["CELLS_CDF"]
        cellsIntersectionSHP = LAU2_data["CELLS_SHP"]

        # Now create workplaces/schools...
        for wp_kind, workers_id in workersIdPerLAU[LAU2_code].iteritems():
            tmp_wp_obj = workplacesDict[wp_kind]
            # We cast the set of ids to a list, shuffle it once
            # and then pop from the list the desired number of agents ids...
            assigned, toAssign = 0, list(workers_id)
            np.random.shuffle(toAssign)
            nToAssign = len(toAssign)
            for tmp_wp in plainGeneratedWorkplaces[wp_kind][LAU2_code]:
                # Check how many workers are left to assign and update the
                # real size of the workplace...
                assigned_now = tmp_wp[wpIndex_size]
                for agent_id in toAssign[assigned:assigned+assigned_now]:
                    generatedAgents[agent_id][agentIndex_wpid] = tmp_wp[wpIndex_id]
                # Put the workplace in a cell...
                point = synpopUtils.assignToCell(cellsCDF=cellsCDF,
                                             cellsIntersectionSHP=cellsIntersectionSHP)
                tmp_wp = tmp_wp + [point.x, point.y, LAU2_geoCode]
                generatedWorkplaces.append(tmp_wp)
                assigned += assigned_now
        iii += 1
        sys.stdout.write("\rCode %r number %05d of %05d"
                            % (NUTS3_code, iii, reference_gdf.shape[0]))
        sys.stdout.flush()
    toc = datetime.datetime.now()
    print("\nWorkplaces created in in %f seconds..." % (toc-tic).total_seconds())


    hhdf = pd.DataFrame(generatedHouseholds,
            columns=["id", "kind", "size", "lon", "lat", "geocode"])
    hhdf.set_index("id", inplace=True)
    nHouseholds = hhdf.shape[0]

    wpdf = pd.DataFrame(generatedWorkplaces,
            columns=["id", "kind", "size", "lon", "lat", "geocode"])
    wpdf.set_index("id", inplace=True)
    nWorkplaces = wpdf.shape[0]

    generatedAgents_DF = pd.DataFrame(generatedAgents,
            columns=["id", "hh", "role", "sex", "age", "edu", "employed", "income", "wp"])

    generatedAgents_DF.to_pickle(cfg["checkpoint_AG"])
    hhdf.to_pickle(cfg["checkpoint_HH"])
    wpdf.to_pickle(cfg["checkpoint_WP"])

    pickle.dump(plainGeneratedWorkplaces, open("checkpoint_genWP.pkl", "wb"))
    pickle.dump(nWorkersToAssign, open("checkpoint_nWRK.pkl", "wb"))
    pickle.dump(workersIdPerLAU, open("checkpoint_WperLAU.pkl", "wb"))

    fOutName = "synpopGenerator_config_%s.py" % referenceName
    with open(fOutName, "wb") as fOut:
        fOut.write("cfg = %r" % cfg)
    return  fOutName, cfg

if __name__ == "__main__":
    cfg_module = sys.argv[1].split(".")[0]
    generateEntities(cfgname=cfg_module)
