# This file is part of the abm prototype implementation of the CoeGSS project

import h5py
import sys
import numpy as np
import datetime

from configuration import *
from entities import Agent, Household, Workplace

def getAttr2Id(datasetName):
    '''
    Function that reads the synthetic population in the dataset `datasetName` from
    the given file and computes the indexes for the default and custom traits of the
    given custom type.

    For example to get the attribute to id in column for the `agent` table you can do:

    ```
    id2attrAgent, attr2idAgent = getAttr2Id("agent")
    # Now you can say that this is the age of the agent
    arrayOfAttributes[attr2idAgent["age"]]
    ```

    This will be used to set and get the attributes of the agents/wps/schools.

    Parameters
    ----------

    datasetName - str
        The name of the dataset found in the data file.

    Returns
    -------

    id2field, field2id: {index: columnName}, {columnName: index}

        The maps telling the position of a column name in the tuple and vice-versa.
    '''
    #TODO inherit the configuration from the caller thread.
    synpopFile = h5py.File(datasetFile, mode="r")
    typeObj = synpopFile[datasetName]
    id2field = {i: k for i, k in enumerate(typeObj.dtype.names)}
    field2id = {k: i for i, k in id2field.iteritems()}
    synpopFile.close()

    return id2field, field2id


def synpopReadDataset(dataset):
    '''
    Generator: cycles over the rows of the given dataset and yields the tuple
    representation of the row accordingly to the type of the table.

    Parameters
    ----------

    dataset (str)
        The name of the dataset to read in the file

    Returns
    -------

    Returns a dictionary like {translation[columnName]: value} to initialize the
    entities of the Population class (agent, hh, wp and community).

    '''
    #TODO get the configuration from the caller.

    synpopFile = h5py.File(datasetFile, mode="r")

    # Here we fetch the handle to the collection and we solve once and for all
    # the mapping between the columnName -> index -> attributeName using the
    # `getAttr2Id` function that tells us which column name we found at
    # position i. We then translate the column names to the corresponding attribute
    # names of agents, workplaces etc. using both the `*AttributesTranslation` and
    # `additional*AttributesColumns` dictionaries.

    tmp_table = synpopFile[dataset]
    if dataset == agentsDatasetName:
        columnsToHold = agentsAttributesTranslation.keys()\
                        + additionalAgentsAttributesColumns.keys()
        attributesTranslation = agentsAttributesTranslation
        additionalTranslation = additionalAgentsAttributesColumns
    elif dataset == householdsDatasetName:
        columnsToHold = householdsAttributesTranslation.keys()\
                        + additionalHouseholdsAttributesColumns.keys()
        attributesTranslation = householdsAttributesTranslation
        additionalTranslation = additionalHouseholdsAttributesColumns
    elif dataset == workplacesDatasetName:
        columnsToHold = workplacesAttributesTranslation.keys()\
                        + additionalWorkplacesAttributesColumns.keys()
        attributesTranslation = workplacesAttributesTranslation
        additionalTranslation = additionalWorkplacesAttributesColumns
    else:
        raise KeyError, "Unknown dataset %s to load!" % dataset


    # Here we filter out the columns of the dataset that are not required in the
    # configuration. We end up with `cleaned_id2attr = {indexOfColumn: attributeName}`
    id2attr, attr2id = getAttr2Id(dataset)
    cleaned_id2attr = {}
    for iii, colName in id2attr.iteritems():
        if colName in columnsToHold:
            if colName in attributesTranslation:
                cleaned_id2attr[iii] = attributesTranslation[colName]
            elif colName in additionalTranslation:
                cleaned_id2attr[iii] = additionalTranslation[colName]
            else:
                raise KeyError, "Unkwown colname `%s` to load" % colName

    # If doing hh or wp also compute the position of hierarchical codes!
    # Turns codes name to a list of indexes corresponding to the position of the
    # hierarchy levels.
    codesIndexes = None
    if dataset in [householdsDatasetName, workplacesDatasetName]:
        #idList = [id2attr[k] for k in sorted(id2attr.keys())]
        #codesIndexes = [idList.index(l) for l in hierarchyColumns]
        codesIndexes = [attr2id[l] for l in hierarchyColumns]


    # Directly read the data in the `tmp_tableArr` array then yield each transformed
    # row of the dataset. The transform function is still the same for all the
    # datasets and is the readRow.
    tmp_tableArr = np.empty(shape=tmp_table.shape, dtype=tmp_table.dtype)
    tmp_table.read_direct(tmp_tableArr)
    transformFunction = readRow
    for tmp_vals in tmp_tableArr:
        yield transformFunction(tmp_vals, cleaned_id2attr, codesIndexes)
    synpopFile.close()

def synpopReadDemographyTable(tableName, targetObj):
    '''
    Returns the demography table in the following format:
    {datetime:
        {(l0,...,lD):
            {sex:
                {age:
                    {"d": val, "b": val}
                }
            }
        }
    }
    '''
    sys.stdout.write("Importing `%s` dataset...\n" % tableName)
    id2field, field2id = getAttr2Id(tableName)
    with h5py.File(datasetFile, mode="r") as synpopFile:
        tmp_table = synpopFile[tableName]

        codesIndexes = [field2id[l] for l in hierarchyColumns[:demographyGeocodeLevel]]
        tmp_tableArr = np.empty(shape=tmp_table.shape, dtype=tmp_table.dtype)
        tmp_table.read_direct(tmp_tableArr)
        date_id = field2id[demographyDateColumn]
        age_id = field2id[demographyAgeColumn]
        sex_id = field2id[demographySexColumn]
        death_id = field2id[demographyMortalityColumn]
        birth_id = field2id[demographyNatalityColumn]
        for vals in tmp_tableArr:
            tmp_date = datetime.datetime.strptime(vals[date_id], demographyDateFormat)
            tmp_geocode = tuple([vals[c] for c in codesIndexes])
            tmp_sex = vals[sex_id]
            tmp_age = vals[age_id]
            tmp_death = vals[death_id]
            tmp_birth = vals[birth_id]

            tmp_dict = targetObj
            if tmp_date not in tmp_dict:
                tmp_dict[tmp_date] = {}
            tmp_dict = tmp_dict[tmp_date]
            if tmp_geocode not in tmp_dict:
                tmp_dict[tmp_geocode]= {}
            tmp_dict = tmp_dict[tmp_geocode]
            if tmp_sex not in tmp_dict:
                tmp_dict[tmp_sex]= {}
            tmp_dict = tmp_dict[tmp_sex]
            if tmp_age not in tmp_dict:
                tmp_dict[tmp_age]= {}
            tmp_dict = tmp_dict[tmp_age]

            tmp_dict["d"] = tmp_death
            tmp_dict["b"] = tmp_birth
        synpopFile.close()

    sys.stdout.write("\n Done!\n")
    sys.stdout.flush()

def readRow(row, indx2attr, codesIndexes=None):
    '''
    Converts the tuple to a dict {attributeName: value} using the index to attribute
    name filtered in the `synpopReadDataset` function.

    If `codesIndexes` is not `None` we also save in the `hierarchyCodeAttributeName`
    the complete hierarchy code of the hh/wp.

    '''

    tmp_obj = {k: row[i] for i, k in indx2attr.iteritems()}
    if codesIndexes is not None:
        # Here we have to pack the location codes...
        tmp_obj[hierarchyCodeAttributeName] = row2geocode(row, codesIndexes)
    return tmp_obj

def row2geocode(row, codesIndexes):
    '''
    Function that given `row` and `codesIndexes` returns the code in the given
    number of hierarchical levels.

    Parameters
    ----------

    `row` (tuple)
        The tuple of the row

    `codesIndexes` (list-like)
        The indexes of the raw in decreasing hierarchical order.

    Examples
    --------

    For example if I have a dataset like this:

    |  a  |  b  | l1 | l0 | l2 |  c  | l3 |
    |  2  |  3  | 4  | 5  | 6  |  7  | 8 |

    where `l*` are the ordered levels of my hierarchy i would read with

    >>> row2geocode(row, [3,2,4,6])
    >>> (5, 4, 6, 8)
    '''
    return tuple([row[i] for i in codesIndexes])


