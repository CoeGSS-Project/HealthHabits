# This file is part of the geo-database implementation of the CoeGSS project

#import matplotlib.pyplot as plt
import pymongo
import pandas as pd
#import shapefile
import shapely
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import urllib
import getpass
import fiona
import sys
from collections import Counter
from bson.objectid import ObjectId

class geoMongoClient(object):

    def __init__(self, dbURL="map.citychrone.org", dbPort=5555, dbUser="coegss_guest", dbPassword=None, dbOptions="authSource=admin",\
                    boundariesDB="boundaries", boundariesColl="2013", cellDB="boundaries", cellColl="SEDAC_PC_UNA", simulationsColl="simulations"):

        if dbPassword is None:
            dbPassword = urllib.quote_plus(getpass.getpass("Enter password for mongo:"))
        else:
            dbPassword = urllib.quote_plus(dbPassword)
        self.__client = pymongo.MongoClient("mongodb://"+ urllib.quote_plus(dbUser) + ":" + dbPassword + "@" + dbURL + ":" + "%d"%dbPort + "/?" + dbOptions)
        del dbPassword

        self.__boundsDB = self.__client[boundariesDB]
        self.__boundsColl = self.__boundsDB[boundariesColl]

        self.__cellsDB = self.__client[cellDB]
        self.__cellsColl = self.__cellsDB[cellColl]
        self.__simulationsColl = self.__cellsDB[simulationsColl]

    def getParent(self, child, parentLevel=1):
        idToFind = None
        if isinstance(child, str) or isinstance(child, ObjectId):
            idToFind = child
        elif isinstance(child, dict) and "_id" in child:
            idToFind = child["_id"]
        else:
            raise RuntimeError("Unknown type in `getParent()` with no '_id' field, you passed me a ", type(child))

        # TODO here I am assuming that we keep the one-char per level format!
        return self.__boundsColl.find_one({"_id": idToFind[:-parentLevel]})


    def getAllChilds(self, parent, field="_id", offset=0, depth=100, limit=0, skip=0):
        '''
        Returns all the entries found starting by parent after descending `offset` levels of
        the hierarchy and stopping at the `depth`-th level. The field of the collection
        storing the hierarhical code is set in `field`.
        If `limit` > 0 returns only the first `limit` documents found; if `skip` > 0 the first
        `skip` documents are skipped.
        '''
        assert(isinstance(skip, int))
        assert(isinstance(limit, int))
        assert(isinstance(offset, int))
        assert(isinstance(depth, int))
        assert(offset >= 0)
        assert(offset <= depth)
        assert(isinstance(parent, str))

        regex = "^%s\w{%d,%d}$" % (parent, offset, depth)
        #sys.stdout.write("Looking for" + regex + "\n")
        return self.__boundsColl.find({field: {"$regex": regex}}, no_cursor_timeout=True, limit=limit, skip=skip)

    def getCountryLevel(self, countryCode="UK", level=1, limit=0, skip=0):
        '''
        Returns all the boundaries of level `level` of country `countryCode` found in `collection`.
        '''
        return self.getAllChilds(countryCode, offset=level, depth=level, limit=limit, skip=skip)

    def getCountryLevelCodes(self, countryCode="UK", level=1, field=["_id"]):
        '''
        Returns a counter containing all the codes found in the field `field` belonging to country `countryCode`
        and at the hierarchical level `level`.
        '''

        codes = []
        for e in self.getCountryLevel(countryCode=countryCode, level=level):
            tmp_val = e
            try:
                for f in field:
                    tmp_val = tmp_val[f]
                codes.append(tmp_val)
            except KeyError:
                continue

        return Counter(codes)

    def overlappingCells(self, referenceShape, shapeGeometryKey="geometry", collectionGeometryKey="geometry"):
        '''
        Generates  to the shapes overlapping with the `referenceShape` found in the `collection`
        collection. The geometry of the passed shape is found in `shape[shapeGeometryKey]` while the cells
        collection stores the coordinates of the point in `cell[collectionGeometryKey]`.

        Returns:

        cells, cellIsInside [bool : True if cell is strictly within `referenceShape` else False]

        '''

        # The ones strictly inside....
        cellsCodeInside = set([d["_id"] for d in self.__cellsColl.find({collectionGeometryKey:\
                                        {"$geoWithin": {"$geometry": referenceShape[shapeGeometryKey]}}})])
        # ... and the ones overlapping....
        cellsOverlap = self.__cellsColl.find({collectionGeometryKey:\
                                        {"$geoIntersects": {"$geometry": referenceShape[shapeGeometryKey]}}},\
                                        no_cursor_timeout=True)
        #sys.stdout.write("Found %d internal cells..." % len(cellsCodeInside))

        for cell in cellsOverlap:
            yield cell, cell["_id"] in cellsCodeInside

    def getBoundaryByID(self, boundID):
        return self.__boundsColl.find_one({"_id": boundID})

    def getCellByID(self, cellID):
        return self.__cellsColl.find_one({"_id": cellID})

    def overlapCell(self, shape, shapeGeometryKey="geometry", collectionGeometryKey="geometry"):
        '''
        Generator
        Returns:
        cell, fractionOverlap
        '''

        for cell, isInside in self.overlappingCells(shape, shapeGeometryKey=shapeGeometryKey, collectionGeometryKey=collectionGeometryKey):
            yield (cell, self.cellShapeOverlap(cell, shape, shapeGeometryKey=shapeGeometryKey, cellGeometryKey=collectionGeometryKey)) if not isInside\
                else (cell, 1.)

    def cellShapeOverlap(self, cell, shape, shapeGeometryKey="geometry", shapeGeoCoordKey="coordinates", cellGeometryKey="geometry", cellGeoCoordKey="coordinates"):
        tmp_cell = Polygon(cell[cellGeometryKey][cellGeoCoordKey][0])
        if shape[shapeGeometryKey]["type"] == "Polygon":
            tmp_shape = Polygon(shape[shapeGeometryKey][shapeGeoCoordKey][0],\
                                shape[shapeGeometryKey][shapeGeoCoordKey][1:])
        elif shape[shapeGeometryKey]["type"] == "MultiPolygon":
            tmp_shape = MultiPolygon([(entry[0], entry[1:]) for entry in shape[shapeGeometryKey][shapeGeoCoordKey]])

        return tmp_cell.intersection(tmp_shape).area/tmp_cell.area


    def sampleShapes(self, sampleSize=1):
        sample = [d for d in self.__boundsColl.aggregate([{"$sample": {"size": sampleSize}}])]
        if sampleSize == 1:
            return sample[0]
        return sample

    def sampleCells(self, sampleSize=1):
        sample = [d for d in self.__cellsColl.aggregate([{"$sample": {"size": sampleSize}}])]
        if sampleSize == 1:
            return sample[0]
        return sample

    def getSimulationsIDs(self):
        return self.performQuery({}, on="simulations")

    def performQuery(self, query, on="boundaries"):
        '''
        Performs a query either on the `cells` or `boundaries` collections
        #TODO expose also other collections and incorporate this custom query in other
        class functions.
        '''

        if on == "boundaries":
            return self.__boundsColl.find(query)
        elif on == "cells":
            return self.__cellsColl.find(query)
        elif on == "simulations":
            return self.__simulationsColl.find(query)
        else:
            sys.stdout.write("ERROR: `on` field %r not known in `performQuery`!" % on)
            return None

    def cellsReference(self, raster="SEDAC_PC_UNA_2015"):
        return self.__boundsDB["cellsReference"].find_one({"_id": raster})["properties"]

    def getCellByID(self, _id):
        return self.__cellsColl.find_one({"_id": _id})

    def getCellByIndex(self, index):
        return self.__cellsColl.find_one({"index": index})

    def updateSimulation(self, query="{}", update="{}", upsert=False):
        return self.__simulationsColl.update_one(query, update, upsert=upsert)

    def updateBoundaries(self, query="{}", update="{}", upsert=False):
        return self.__boundsColl.update_many(query, update, upsert=upsert)

    def updateBoundary(self, boundary, update="{}"):
        if isinstance(boundary, str) or isinstance(boundary, ObjectId) or isinstance(boundary, unicode):
            tmp_id = boundary
        elif isinstance(boundary, dict) and "_id" in boundary:
            tmp_id = boundary["_id"]
        else:
            raise RuntimeError("Unknown type in `updateBoundary()` with no '_id' field, you passed me a ", type(boundary))
        self.__boundsColl.find_one_and_update({"_id": tmp_id}, update)



    def insertFromPdDf(self, dataFrame, fieldsMap, keyDF=None, keyDB="_id", noValue=""):

        # Assert that we do not have duplicate keys...
        if keyDF is None:
            idxs = dataFrame.index
        else:
            idxs = dataFrame[keyDF]
            setIdxs = set(idxs)
            try:
                assert(len(idxs) == len(setIdxs))
            except AssertionError:
                print("Found a duplicate key in teh dataFrame key column, aborting!")
                return

        for index, (_, row) in zip(idxs, dataFrame.iterrows()):
            # Retrieve boundary...
            tmpLAD = self.__boundsColl.find_one({keyDB: index})

            if tmpLAD:
                updateCall = {"$set": {}}
                for dfField, dbField in fieldsMap.items():
                    updateCall["$set"][dbField] = row[dfField]

                #print(tmpLAD["properties"], updateCall)
                self.__boundsColl.find_one_and_update({"_id": tmpLAD["_id"]}, updateCall)
        return

    def bounds2df(self, query, useAsIndex, record2column):
        '''
        Returns a data frame with index and columns as given.
        '''

        tmpData = {}

        for doc in self.__boundsColl.find(query):
            try:
                tmp_idx = doc
                for key in useAsIndex.split("."):
                    tmp_idx = tmp_idx[key]
                    
                
                tmp_data = {}
                for record, column in record2column.items():
                    try:
                        tmp_val = doc
                        for key in record.split("."):
                            tmp_val = tmp_val[key]
                            tmp_data[column] = tmp_val
                    except KeyError:
                        print("Field %s not found in doc %s, skipping" % (key, doc["_id"]))
                        del tmp_data[column]
                        continue

                    tmpData[tmp_idx] = tmp_data
                    
            except KeyError:
                print("Document not found!!! : ", tmp_idx, key)
                continue

        return pd.DataFrame(tmpData).transpose()

    def insertRecord(self, document, collection="boundaries", update=False):
        tmp_coll = self.__boundsColl if collection == "boundaries" else\
                    self.__simulationsColl if collection == "simulations" else\
                     None

        if tmp_coll:
            # Check for existence...
            if "_id" in document:
                if tmp_coll.find_one({"_id": document["_id"]}):
                    if update:
                        tmp_coll.find_one_and_replace({"_id": document["_id"]},\
                                {k: v for k,v in document.items() if k != "_id"})
                    else:
                        sys.stdout.write("Document with '_id': %s already in collection and update set to False, skipping!\n" % (str(document["_id"])))
                else:
                    tmp_coll.insert_one(document)

            else:
                sys.stdout.write("no '_id' found in document in insertRecord!\n")
                return
        else:
            sys.stdout.write("Unknown collection %s in insertRecord!\n" % collection)
            return






    # Propagation and aggregations...

    # TODO write a decorator for all functions accepting a record/_id as first 
    # argument to check format and return right id?
    def propagateCountryLevelDown(self, countryCode, level, field, offset=0, depth=100, mode="copy", on=None, childsFilter=None):
        for parent in self.getCountryLevel(countryCode=countryCode, level=level):
            self.propagateParentDown(parent, field, offset=offset, depth=depth,\
                    mode=mode, on=on, childsFilter=childsFilter)
        return


    def propagateParentDown(self, doc, field, offset=0, depth=100, mode="copy", on=None, childsFilter=None):
        # TODO make more modes available, like a repartition based on a filed etc.

        parentID = None
        if isinstance(doc, str) or isinstance(doc, ObjectId):
            parentID = doc
        elif isinstance(doc, dict) and "_id" in doc:
            parentID = doc["_id"]
        else:
            raise RuntimeError("Unknown type in `updateBoundary()` with no '_id' field, you passed me a ", type(doc))

        if mode == "copy":
            # Just copy to all levels...
            # TODO make an even lower level of requests where we just get the query to be done
            # and not the iterator to speedup multiple updates.

            # TODO make sure to check that we are not overwriting data or clearly specify the
            # inner level of data to copy!

            try:
                tmp_val = self.__boundsColl.find_one(parentID)
                for key in field.split("."):
                    tmp_val = tmp_val[key]
            except KeyError:
                sys.stdout.write("Warning, key %s not found in record %s...\n" % (key, parentID))
                return

            for child in self.getAllChilds(str(parentID), offset=offset, depth=depth):
                self.__boundsColl.find_one_and_update({"_id": child["_id"]},\
                                    {"$set": {field: tmp_val}})

        return


    def aggregateCountryLevels(self, countryCode, field, levelStart, levelStop, noValue=.0, mode="sum", on=None, valueFilter=None, valfoo=None, denfoo=None, idsToSkip=[]):
        '''
        Parameters:
        -----------
        valfoo : function, optional
            A function to manipulate the read value before it is summed to the cumulative sum.
            For example if the stored data is an array you can sum the mean of it by setting `valfoo=np.mean`
        denfoo : function, optional
            A function to manipulate the denominator value before it is summed to the cumulative sum.
            For example if the stored data is an array you can sum the mean of it by setting `valfoo=np.mean`
        valueFilter : function, optional
            A function that takes the value being aggregated and the `on` value (if on is not passed
            just write a function that ignores the second argument ;) ).
            The function must evaluate to a `True`/`False` (or equivalent) and tells on which records
            we perform the aggregation (`True` for counting in `False` to leave it out).
            For example to cut out all the cells with the observable being aggregated equals to zero
            pass:
            `valueFilter=lambda x, y: x>.0`

            Default is always `True`

        idsToSkip : list/set, optional
            The list or set of documents featuring one of these `_id` to skip as children.


        '''
        assert(levelStart >= levelStop)

        valueFilter =  valueFilter if valueFilter else lambda x, y: True

        weightedMode = mode in ["wsum", "wmean"]
        meanMode = mode in ["mean", "wmean"]
        assert(not( (weightedMode) and (on is None)))

        cumSum = lambda curr, fact, cumul: curr*fact + cumul
        skippedCodes = set()
        for level in range(levelStart, levelStop-1, -1):
            for parent in self.getCountryLevel(countryCode=countryCode, level=level):
                cumNum = .0
                cumDen = .0
                for child in self.getAllChilds(str(parent["_id"]), depth=1, offset=1):
                    try:
                        tmp_val = child
                        for k in field.split("."):
                            tmp_val = tmp_val[k]

                        if weightedMode:
                            tmp_weight = child
                            for k in on.split("."):
                                tmp_weight = tmp_weight[k]
                        else:
                            tmp_weight = 1.

                    except KeyError:
                        skippedCodes.add(child["_id"])
                        continue
                    if valfoo:
                        tmp_val = valfoo(tmp_val)
                    if denfoo:
                        tmp_weight = denfoo(tmp_weight)

                    if (not valueFilter(tmp_val, tmp_weight)) or (child["_id"] in idsToSkip):
                        continue

                    cumNum = cumSum(tmp_val, tmp_weight, cumNum)
                    cumDen = cumSum(tmp_weight, 1., cumDen)

                if meanMode:
                    aggregatingVal = .0 if cumDen == .0 else cumNum/cumDen
                else:
                    aggregatingVal = cumNum

                self.__boundsColl.find_one_and_update({"_id": parent["_id"]},
                        {"$set": {field: aggregatingVal}})

        if skippedCodes:
            sys.stdout.write("Warning, %d docs skipped!\n" % (len(skippedCodes)))
        return skippedCodes










