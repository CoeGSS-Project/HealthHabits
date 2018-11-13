#This file is part of
# YASPoGen
#    Yet Another Synthetic Population Generator
# that is part of the CoeGSS project.

import geopandas as gpd
import pandas as pd
import numpy as np
from numpy import array, argmax, zeros
from numpy.random import choice, rand
import shapely
#import matplotlib.pyplot as plt


def assignHierarchyCodes(NUTS_LAU_hierarchy, codeColumnLabel="code", minOverlap=.7):
    '''
    Assigns the hierarchy codes to the hierarchical represntation of the
    boundaries.

    Parameters
    ----------

    NUTS_LAU_hierarchy  - list of geo-dataframes
        An ordered list whose indexes are the levels of the hierarchy (from coarser
        to finer) and whose elements are the geodatabases containing the shapes of
        that level. Every shape in a lower level MUST be the son of some parent in
        the upper level (parent will be assigned based on the most overlapping
        one).

    codeColumnLabel - str
        The column name of the column that will contain the hierarchical code
        tuple. At level `l` this is a tuple of length `l+1` with the 0-level code,
        1st level code etc.

    minOverlap - float
        The minimum of a son with its parent to be kept in the final dataframe.

    Returns
    -------

    NUTS_LAU_hierarchy - updated inplace
        A List whose index is the level of the hierarchy and whose values are the
        geodataframes of that hierarchical level.

        The geodataframes are decorated with additional columns, depending on the
        level:
        - for all: a `codeColumnLabel` column containing the tuple of teh
          hierarchical code;
        - `l0`, `l1`, ..., `lL+1` columns containing the index of the parent at
          the i-th level.

    '''
    nHierarchyLevels = len(NUTS_LAU_hierarchy)

    # Assign the default values to each level of the hierarchy.
    for tmp_level, tmp_gdf in enumerate(NUTS_LAU_hierarchy):
        tmp_gdf[codeColumnLabel] = [tuple() for i in range(tmp_gdf.shape[0])]

    for tmp_level, tmp_gdf in enumerate(NUTS_LAU_hierarchy):
        # Now, for each level in the hierarchy find the parent of each shape and
        # save the corresponding code.
        print "Doing level", tmp_level

        tmp_son_index = 0
        gdf_nRows = tmp_gdf.shape[0]
        while tmp_son_index < gdf_nRows:
            tmp_son_code = tmp_gdf.index[tmp_son_index]
            tmp_son_data  = tmp_gdf.loc[tmp_son_code]

            # Now for each son at the current level look for the most overlapping
            # shape in the previous hierarchical level...
            if tmp_level > 0:
                max_parent_overlap = .0
                max_parent_index = None
                max_parent_code = None

                # Here we cycle over the possible parents...
                if tmp_son_data["geometry"].area > .0:
                    for tmp_parent_index, tmp_parent_data\
                            in NUTS_LAU_hierarchy[tmp_level-1].iterrows():
                        tmp_overlap = tmp_son_data["geometry"].intersection(tmp_parent_data["geometry"]).area/\
                                tmp_son_data["geometry"].area

                        if tmp_overlap > max_parent_overlap:
                            max_parent_overlap = tmp_overlap
                            max_parent_index = tmp_parent_index
                            max_parent_code = tmp_parent_data[codeColumnLabel]

                if max_parent_overlap < minOverlap:
                    if max_parent_code is None:
                        print "\nMissing parent for child %s name %s id %s at level %s, deleting it..."\
                        % (tmp_son_code, tmp_son_data["name"], tmp_son_data["id"], tmp_level)
                    else:
                        print "\nChild %s name %s id %s at level %s had a parent overlap of %.03f, deleting it..."\
                        % (tmp_son_code, tmp_son_data["name"], tmp_son_data["id"], tmp_level, max_parent_overlap)
                    tmp_gdf.drop(tmp_son_code, inplace=True)
                    gdf_nRows = tmp_gdf.shape[0]
                    continue
                else:
                    # Save the tuple code of parent
                    tmp_gdf.at[tmp_son_code, codeColumnLabel] = max_parent_code

            # Append the dataframe index as the index the current hierarchy level...
            tmp_tuple = list(tmp_gdf.loc[tmp_son_code, codeColumnLabel])
            tmp_tuple.append(int(tmp_son_code))
            tmp_gdf.at[tmp_son_code, codeColumnLabel] = tuple(tmp_tuple)

            tmp_son_index += 1
            #print "\rDone son %06d out of %06d left..." % (tmp_son_index, gdf_nRows),

        # Now that we have the complete code unpack it to the `li` columns...
        for levelCode in range(tmp_level+1):
            tmp_gdf["l%d" % levelCode] = tmp_gdf[codeColumnLabel].apply(lambda v: v[levelCode])


def getSonsMap(sonsDF, parentCode, codeColumnName="code"):
    '''
    From the given dataframe returns A VIEW on the rows that have the code
    starting with the parent code.

    Parameters
    ----------

    sonsDF - pandas or geopandas DataFrame
        The dataframe containing the sons, must have a column with the tuple of
        the hierarchical code.

    parentCode - tuple
        The tuple of teh parent code.

    codeColumnName - str
        The dataframe column name where the code is stored. Defaults to `code`.

    Returns
    -------

    sonsDF VIEW
        A VIEW of the dataframe with the rows containing the prefic equal to the
        given parent code.
    '''

    parentLevel = len(parentCode)
    filterFoo = lambda v: v[:parentLevel]
    return sonsDF[sonsDF[codeColumnName].apply(filterFoo) == parentCode]

def plotHierarchyParent(NUTS_LAU_hierarchy, parentCode, ax, codeColumnName="code", levelStyles=None):
    '''
    Plots the hierarchical representation of the areas below a given aprent on a
    given axis.

    Parameters
    ----------

    NUTS_LAU_hierarchy - list
        The list containing the geographical hierarchy.

    parentCode - tuple
        The tuple of the parent code that we want to visualize.

    ax - matplotlib.axes
        The axes of the figure where to plot the hierarchy. Get them with `fig, ax
        = plt.subplots` or with `ax = plt.gca()`.

    levelStyles - dict
        A dictionary whose key are the hierarchical levels and whose values are
        line style to plot that level. Example and default value:
        `
        levelStyles = {
                    0: {"lw": 2, "ls": "-", "c": "black", "a": .6},
                    1: {"lw": 1, "ls": "-", "c": "blue", "a": .7},
                    2: {"lw": .75, "ls": "--", "c": "orange", "a": .9}
                  }
    '''
    ax.set_aspect("equal")
    levelStyles = levelStyles if levelStyles else {
                    0: {"lw": 2, "ls": "-", "c": "black", "a": .6},
                    1: {"lw": 1, "ls": "-", "c": "blue", "a": .7},
                    2: {"lw": .75, "ls": "--", "c": "orange", "a": .9}
                  }

    parentLevel = len(parentCode) - 1
    for level, level_gdf in enumerate(NUTS_LAU_hierarchy):
        if level < parentLevel:
            continue

        tmp_style = levelStyles[level]
        tmp_data = getSonsMap(level_gdf, parentCode, codeColumnName=codeColumnName)

        if tmp_data.shape[0]:
            tmp_data.plot(ax=ax, color="none", edgecolor=tmp_style["c"],
                          alpha=tmp_style["a"], linewidth=tmp_style["lw"],
                          linestyle=tmp_style["ls"])

            if level == len(NUTS_LAU_hierarchy)-1:
                # print also father...
                try:
                    #parentCode = tmp_data[codeColumnName][0][:-1]
                    parentCode = tmp_data[codeColumnName].iloc[0][:-1]
                except Exception as e:
                    #print str(e)
                    print "Wrong:", tmp_data[codeColumnName]
                    pass
                else:
                    print "Correct:", tmp_data[codeColumnName]
                    tmp_father = getSonsMap(NUTS_LAU_hierarchy[level-1], parentCode,
                            codeColumnName=codeColumnName)
                    tmp_father.plot(ax=ax, color="none", edgecolor="red",
                                    alpha=.9, linewidth=.8, linestyle="-")


def preProjectToFinestLevel(NUTS_LAU_hierarchy, minCoverage=.975, minSonFrac=.01):
    '''
    Projects areas to lower levels filling the holes if an area has childs covering it for
    less than `mincoverage`.

    If an area has no child we copy the area itself as its child in the next level.

    Parameters
    ----------

    NUTS_LAU_hierarchy - list
        The ordered list containing the hierarchical areas.

    minCoverage - float
        The minimum percentage coverage [0.0-1.0] for a parent area to be considered
        fully covered by the sons.

    Returns
    -------

    (NUTS_LAU_hierarchy, overlapCounter)
        The input hierarchy updated with the projection of the areas to the finer
        levels. `overlapCounter` has a list containing the overlap of the sons values
        found for each shape of a given level: for example you can see how much
        the shapes of level 0 were covered by shapes of level 1 by doing:

        ```
        plt.hist(overlapCounter[0])
        ```

        If an area gets projected to the lower level for lack of children we
        assume overlap = 1.
    '''

    # Do the projection for all the levels but the last one.
    overlapCounter = {}
    for level, level_data in enumerate(NUTS_LAU_hierarchy[:-1]):
        print "Doing level", level

        # We will look for sons at this level and save the sons overlap for each
        # in this list for later usage...
        son_level = level+1
        overlapCounter[level] = []

        for parent_code, parent_data in level_data.iterrows():
            # For each parent check the area of the sons covered
            # Fetch the sons with `getSonsMap`. The cumulative shape
            # resulting from the union of the sons will be cumulated in
            # `sons_cumulative`
            parent_shape = parent_data["geometry"]
            son_gdf = NUTS_LAU_hierarchy[son_level]
            intersection_indexes = son_gdf.intersects(parent_shape)
            intersection_df = son_gdf[intersection_indexes]
            sons_cumulative = intersection_df.unary_union
            diff = parent_shape.difference(sons_cumulative)
            percent_covered = 1. - diff.area/max(1e-8, parent_shape.area)

            if percent_covered < minCoverage:
                # Create a new area with new code as son for the uncovered area...
                print "\nCREATING A NEW CHILD AREA FOR %r covered by only %.02f percent..." % (parent_code, percent_covered)
                parent_row = parent_data.copy()
                if percent_covered < minSonFrac*2.:
                    # Paste parent as son area
                    parent_row = parent_data.copy()
                    tmp_son_gdf = NUTS_LAU_hierarchy[son_level]
                    NUTS_LAU_hierarchy[son_level] = tmp_son_gdf.append(parent_row, verify_integrity=True, ignore_index=True)
                elif diff.type == "MultiPolygon":
                    for subPoly in diff.geoms:
                        subPolyArea = subPoly.area
                        if subPolyArea > .0 and subPolyArea/parent_shape.area > minSonFrac:
                            # Add the new area as children to the reference dataframe
                            parent_row["geometry"] = subPoly
                            tmp_son_gdf = NUTS_LAU_hierarchy[son_level]
                            NUTS_LAU_hierarchy[son_level] = tmp_son_gdf.append(parent_row, verify_integrity=True, ignore_index=True)
                elif diff.type == "Polygon":
                    parent_row["geometry"] = diff
                    tmp_son_gdf = NUTS_LAU_hierarchy[son_level]
                    NUTS_LAU_hierarchy[son_level] = tmp_son_gdf.append(parent_row, verify_integrity=True, ignore_index=True)
                else:
                    raise KeyError, "Unknown shape type %s in boundary" % (diff.type, parent_code)

            overlapCounter[level].append(percent_covered)
        print "\nDone!"
    return NUTS_LAU_hierarchy, overlapCounter


def addSonToLevel(tmp_son_gdf, son_level, parent_data, codeColumnName="code"):
    parent_row = parent_data.copy()
    next_son_code = tmp_son_gdf.index.max() + 1
    parent_row["l%d" % son_level] = next_son_code
    tmp_tuple = list(parent_data[codeColumnName])
    tmp_tuple.append(next_son_code)
    parent_row[codeColumnName] = tuple(tmp_tuple)
    return tmp_son_gdf.append(parent_row, verify_integrity=True, ignore_index=True)

def projectToFinestLevel(NUTS_LAU_hierarchy, codeColumnName="code", minCoverage=.975,
        minSonFrac=.015):
    '''
    Projects areas to lower levels. If an area has childs covering it for
    more than `mincoverage` percent these child are used as the only child of the area.

    If an area has less area covered we add one artificial son accounting for this difference.

    If an area has no child we copy the area itself as its child in the next level.

    Parameters
    ----------

    NUTS_LAU_hierarchy - list
        The ordered list containing the hierarchical areas, already filled with
        the codes by `assignHierarchyCodes`.

    codeColumnName - str
        The name of the column containing the tuple codes.

    minCoverage - float
        The minimum percentage coverage [0.0-1.0] for a parent area to be considered
        fully covered by the sons.

    minSonFrac - float
        The minimum fraction of the parent area for a missing child area to be
        considered and used as a son.

    Returns
    -------

    (NUTS_LAU_hierarchy, overlapCounter)
        The input hierarchy updated with the projection of the areas to the finer
        levels. `overlapCounter` has a list containing the overlap of the sons valuies
        found for each shape of a given level: for example you can see how much
        the shapes of level 0 were covered by shapes of level 1 by doing:

        ```
        plt.hist(overlapCounter[0])
        ```
    '''

    # Do the projection for all the levels but the last one.
    overlapCounter = {}
    for level, level_data in enumerate(NUTS_LAU_hierarchy[:-1]):
        print "Doing level", level

        # We will look for sons at this level and save the sons overlap for each
        # in this list for later usage...
        son_level = level+1
        overlapCounter[level] = []

        for parent_code, parent_data in level_data.iterrows():
            # For each parent check the area of the sons covered
            # Fetch the sons with `getSonsMap`. The cumulative shape
            # resulting from the union of the sons will be cumulated in
            # `sons_cumulative`
            parent_shape = parent_data["geometry"]
            son_gdf = getSonsMap(NUTS_LAU_hierarchy[son_level], parent_data[codeColumnName])
            # We could use `unary_union` but sometimes it fails for no reason, better
            # safe than sorry.
            sons_cumulative = shapely.geometry.Polygon()
            for son_code, son_data in son_gdf.iterrows():
                sons_cumulative = sons_cumulative.union(son_data["geometry"])
            diff = parent_shape.difference(sons_cumulative)
            percent_covered = 1. - diff.area/parent_shape.area

            # Now, diff is the missing area in either the case of missing sons or in
            # the case of missing areas. If it is a `MultiPolygon` we use each
            # sub-polygon as a separate area, given that that area covers at least
            # the `minSonFrac` area of the parent area.

            parent_row = None
            if percent_covered < minCoverage:
                parent_row = parent_data.copy()
                print "\rParent %s name %s id %s at level %d was covered by %.02f percent by sons, adding children..." %\
                        (parent_code, parent_row["name"],
                         parent_row["id"], level, percent_covered), " "*30,
                if diff.type == "MultiPolygon":
                    for subPoly in diff.geoms:
                        if subPoly.area/parent_shape.area > minSonFrac:
                            # Add the new area as children to the reference dataframe
                            parent_row["geometry"] = subPoly
                            NUTS_LAU_hierarchy[son_level] =\
                                    addSonToLevel(
                                tmp_son_gdf=NUTS_LAU_hierarchy[son_level],
                                son_level=son_level, parent_data=parent_row)
                elif diff.type == "Polygon"\
                         and diff.area/parent_shape.area > minSonFrac:
                    parent_row["geometry"] = diff
                    NUTS_LAU_hierarchy[son_level] = addSonToLevel(
                            tmp_son_gdf=NUTS_LAU_hierarchy[son_level],
                            son_level=son_level, parent_data=parent_row)
                else:
                    raise KeyError,\
                        "Unknown shape type %s in boundary %r" %\
                                        (diff.type, parent_code)
            overlapCounter[level].append(percent_covered)
        print "\nDone!"
    return NUTS_LAU_hierarchy, overlapCounter

def cell2shape(cell):
    '''
    Function that given the cell shape returns the sapely object of the boundary.
    '''
    return shapely.geometry.Polygon(cell["geometry"]["coordinates"][0])

def shapeCell2intersection(boundSHP, cellSHP):
    '''
    Function that given the shape and the cell returns the sapely
    object of their intersection.
    '''
    return cellSHP.intersection(boundSHP)

def bound2shape(bound):
    '''
    Function that given the bound shape returns the sapely object of the boundary.
    '''
    tmp_geometry = bound["geometry"]["type"]
    if tmp_geometry == "Polygon":
        tmp_boundSHP = shapely.geometry.Polygon(bound["geometry"]["coordinates"][0])
    elif tmp_geometry == "MultiPolygon":
        tmp_boundSHP = shapely.geometry.MultiPolygon([(entry[0], entry[1:])
                                     for entry in bound["geometry"]["coordinates"]])
    else:
        raise RuntimeError, "Unknown geometry %s" % tmp_geometry
    return tmp_boundSHP


def assignToCell(cellsCDF, cellsIntersectionSHP):
    '''
    Given the CDF of the cells and the corresponding shape in the
    (shapely, xmin, ymin, dx, dy) format returns a point randomly
    placed within a shape extracted proportionally to CDF.
    '''

    selectedCell = argmax(rand() < cellsCDF)
    selectedIntersection = cellsIntersectionSHP[selectedCell][0]
    xmin, ymin, dx, dy = cellsIntersectionSHP[selectedCell][1:5]

    point = shapely.geometry.Point(-180, -180)
    while not selectedIntersection.contains(point):
        point = shapely.geometry.Point(rand()*dx + xmin,
                                       rand()*dy + ymin)
    return point

################
# Clustering   #
################

################
# KMeans       #
################

from sklearn.cluster import KMeans

def clusterPointsInLevelsTopDown(xys, levelsTargetSize):
    '''
    Returns and array with [(c0, c1, ..., cl), ] codes for each point...
    '''
    N = xys.shape[0]
    levelSize = levelsTargetSize[0]
    nClusters = int(round(N/float(levelSize)))

    model = KMeans(n_clusters=nClusters, init="random", n_jobs=-1)
    assignment = model.fit_predict(xys)
    codes = [[a] for a in assignment]
    codesUnique = set(assignment)

    if len(levelsTargetSize) > 1:
        for subclusterCode in codesUnique:
            tmp_points = np.where(assignment == subclusterCode)[0]
            suffix = clusterPointsInLevelsTopDown(xys[tmp_points], levelsTargetSize[1:])
            for index_res, index_old in enumerate(tmp_points):
                codes[index_old].extend(suffix[index_res])
    return codes

def clusterPointsInLevelsBottomUp(xys, nClustersPerLevel, sampleFrac=1.,
        maxSamples=15000, **kwargs):
    '''
    Returns and array with [(c0, c1, ..., cl), ] codes for each point.

    Parameters
    ----------

    xys - np.array
        An `n*d` array containing n samples in a d-dimensional space.

    nClustersPerLevel - list/array
        A list containing the number of clusters to be found at the first, second,
        ... levels, being the first level the coarser one.

    sampleFrac - float
        The fraction of points to be used in training in the finest refinement
        level to speed up the fitting procedure.

    maxSamples - int
        The maximum samples to fit the model on, we will truncate
        `xys.shape[0]*sampleFrac` to this maximum level. 15000 works well for a 16GB
        machine.

    **kwargs
        Will be passed to the KMeans fit method. For instance you can set the
        `init` and the `n_init` parameters.

    Returns
    -------

    (n*l) np.array where n rows corresponds to the n input points and where each
    row is composed by `l = len(nClustersPerLevel)` levels of the code.

    '''
    N = xys.shape[0]
    sampleFrac = max(.01, min(1., sampleFrac))
    Nsample = int(max(1, min(maxSamples, int(N*sampleFrac))))
    nClusters = int(min(Nsample, max(1, int(round(nClustersPerLevel[-1])))))

    model = KMeans(n_clusters=nClusters, n_jobs=-1, **kwargs)
    if Nsample < N:
        tmp_data = xys[choice(N, max(1, int(Nsample)), replace=False)]
        model.fit(tmp_data)
        assignment = model.predict(xys)
    else:
        assignment = model.fit_predict(xys)

    codes = np.column_stack((assignment,))

    if len(nClustersPerLevel) > 1:
        center_points = model.cluster_centers_
        prefix_labels = clusterPointsInLevelsBottomUp(center_points,
                nClustersPerLevel[:-1], sampleFrac=1., **kwargs)

        prefix = array([prefix_labels[suffix] for suffix in codes[:,0]])
        codes = np.column_stack((prefix, codes))
    return codes

def clusterPointsInLevelsBottomUpOld(xys, nClustersPerLevel, init="k-means++",
        n_init=10):
    '''
    Returns and array with [(c0, c1, ..., cl), ] codes for each point...
    '''
    N = xys.shape[0]
    nClusters = min(N, max(1, int(round(nClustersPerLevel[-1]))))

    model = KMeans(n_clusters=nClusters, n_jobs=-1, init=init, n_init=n_init)
    #print "level", len(nClustersPerLevel), "xys.shape", xys.shape, "nCluster", nClusters

    assignment = model.fit_predict(xys)
    codes = [[a] for a in assignment]
    if len(nClustersPerLevel) > 1:
        center_points = model.cluster_centers_.copy()
        prefix = clusterPointsInLevelsBottomUpOld(center_points, nClustersPerLevel[:-1])
        for cluster_index, cluster_prefix in enumerate(prefix):
            tmp_points = np.where(assignment == cluster_index)[0]
            for i in tmp_points:
                codes[i] = cluster_prefix + codes[i]
    return codes

def visualizeKMeansResults(res_arr, xys, fig):
    nLevels = res_arr.shape[1]
    fig.set_size_inches(5*nLevels,4)
    for l in range(nLevels):
        ax = fig.add_subplot(1, nLevels, l+1)
        ax.scatter(xys[:,0], xys[:,1], c=res_arr[:,:l+1].sum(axis=1))

def plotKMeansSizePerLevel(r_arr, levelsTargetSize, fig):
    # Size distribution for level
    from collections import Counter
    nLevels = len(levelsTargetSize)
    fig.set_size_inches(5*nLevels,4)
    for levelID, levelSize in enumerate(levelsTargetSize):
        ax = fig.add_subplot(1,nLevels,levelID+1)
        ax.set_title("Level %d" % levelID, size=18)

        tmp_results = Counter([tuple(c) for c in r_arr[:,:levelID+1]])
        Xs = [c[1] for c in tmp_results.most_common()]

        r = ax.hist(Xs)
        ySpan = [0,max(r[0])*1.15]
        ax.plot([levelSize]*2, ySpan, "--r", lw=2)

        ax.set_xlabel(r"Cluster Size - $s$", size=18)
        ax.set_ylabel(r"$P(s)$", size=18)
        ax.xaxis.set_tick_params(size=16)
        ax.yaxis.set_tick_params(size=16)
        ax.set_ylim(ySpan)

    fig.tight_layout()

################
# Hierarchical #
################

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster,\
                                    fclusterdata, ClusterNode, to_tree
from scipy.spatial.distance import cdist, pdist, squareform

def getLeafesFromNode(nodeRef):
    '''
    Wrapper around the recursive function.

    Retuns
    ------
    The set containing the leafes id under a given root.
    '''

    leafes = set()
    getSons(root=nodeRef, cumul=leafes)
    return leafes

def getSons(root, cumul):
    '''
    Starting from a node `root` and with a set `cumul` get all the leafes of this root node.
    '''

    leftSon = root.get_left()
    rightSon = root.get_right()

    for son in [leftSon, rightSon]:
        if son is None:
            # Arrived at the leaf
            cumul.add(root.id)
        else:
            # Keep digging
            getSons(son, cumul)

def getSonSizeAtLeast(root, size):
    '''
    Starting from root, finds all the children nodes in the three whose size is at least size.
    The id of the last nodes with a size gte than `size` before finding at both children with size
    smaller than `size` are returned.

    If a parent node `k` has a son `i` larger and one son `j` smaller than target size, the process
    keeps looking for candidates in the `i` branch while it marks `j` as candidate.
    '''

    if root.count <= size:
        # If I am starting with a node already below size I am done.
        return set([root.id])

    leftSon = root.left
    rightSon = root.right

    if max(leftSon.count, rightSon.count) < size:
        # Stop and mark parent as candidate...
        #print "child"
        return set([root.id])
    else:
        #print "dig"
        out = set()
        for son in [leftSon, rightSon]:
            if son.count >= size:
                out = out.union(getSonSizeAtLeast(son, size))
            else:
                out.add(son.id)
        return out

def hierarchicallyCluster(xys, levelsTargetSize, method="complete"):
    linkageMatrix = linkage(xys, method="complete")
    clusteringResults = findRootsLevels(linkageMatrix, levelsSize=levelsTargetSize)

    return clusteringResults, linkageMatrix

def findRootsLevels(linkageMatrix, levelsSize):
    '''
    Given a linkage matrix, the tree representation and the target level sizes
    returns a dictionary containing {level: [root0, root1, ..., rootN] for level in levels}

    We assume that we have only ONE level-0 cluster (i.e., pass a dendogram of size slightly
    larger than levelsSize[0])!

    Parameters
    ----------

    linkageMatrix - pdist-like matrix
        Condensed matrix representing the distances between points

    levelsSize - (ordered iterable)
        The list of index <-> target size of level
    '''

    root, treeRepresentation  = to_tree(linkageMatrix, rd=True)
    '''
    root - scipy.cluster.hierarchy.ClusterNode
        The root of the clustering tree, get it with `root = to_tree(linkageMatrix)`

    treeRepresentation - list of scipy.cluster.hierarchy.ClusterNode
        A list of ClusterNode objects - one per original entry in the linkage matrix plus entries
        for all clustering steps. Get it with `root, treeRepresentation = to_tree(linkageMatrix, rd=True)`
    '''

    results = {}
    tmp_target_size = levelsSize[0]
    N = root.count

    # Find first cluster whose size is
    startingNode = argmax(linkageMatrix[:,3] > tmp_target_size) + N
    if startingNode == N:
        # The root is already smaller than initial size
        startingNode = root.id

    startingNode = treeRepresentation[startingNode]

    results[0] = [((0,), startingNode)]

    for tmp_level in range(1, len(levelsSize)):
        tmp_target_size = levelsSize[tmp_level]

        results[tmp_level] = []
        for tmp_root_id, tmp_root in results[tmp_level-1]:
            ids = getSonSizeAtLeast(tmp_root, tmp_target_size)
            codes = range(len(ids))
            results[tmp_level].extend([(tuple(list(tmp_root_id)+[code]),
                                        treeRepresentation[idx])
                    for code, idx in zip(codes, ids)])

    return results

def percentClustered(linkageMatrix):
    # Check for the codes of the leaves and when they appear in the linkage...
    Nsteps = linkageMatrix.shape[0]
    Nnodes = float(Nsteps + 1)
    seenLeaves = set()
    percentPerStep = zeros(Nsteps)
    for iii, row in enumerate(linkageMatrix):
        seenLeaves.update([i for i in row[:2] if i <= Nsteps])
        percentPerStep[iii] = len(seenLeaves)/Nnodes
    return percentPerStep


def visualizeHierarchicalResults(clusteringResults, xys, fig):
    nLevels = len(clusteringResults)
    fig.set_size_inches(4*nLevels,4)

    xs, ys = xys[:,0], xys[:,1]
    for level, levelData in clusteringResults.iteritems():
        ax = fig.add_subplot(1, nLevels, level+1)

        for rootCode, root in levelData:
            points = getLeafesFromNode(root)
            tmp_x = xs[[i for i in points]]
            tmp_y = ys[[i for i in points]]
            ax.scatter(tmp_x, tmp_y, label="%s" % "-".join(["%d"%c for c in rootCode]), alpha=.3)
        ax.set_xlim(-.1,1.1)
        ax.set_ylim(-.1,1.1)
        #plt.legend(loc="upper left", bbox_to_anchor=[1,1], ncol=2*(level+1))
        fig.tight_layout()

def plotHierarchicalSizePerLevel(clusteringResults, levelsTargetSize, fig):
    # Plots Size distribution for level
    nLevels = len(levelsTargetSize)

    fig.set_size_inches(5*nLevels, 4)
    for levelID, levelSize in enumerate(levelsTargetSize):
        ax = fig.add_subplot(1, nLevels, levelID+1)
        ax.set_title("Level %d" % levelID, size=18)

        tmp_results = clusteringResults[levelID]
        Xs = [c[1].count for c in tmp_results]

        r = ax.hist(Xs)
        ySpan = [0,max(r[0])*1.15]
        ax.plot([levelSize]*2, ySpan, "--r", lw=2)

        ax.set_xlabel(r"Cluster Size - $s$", size=18)
        ax.set_ylabel(r"$P(s)$", size=18)

        ax.xaxis.set_tick_params(size=16)
        ax.yaxis.set_tick_params(size=16)

        ax.set_ylim(ySpan)

    fig.tight_layout()
