#This file is part of
# YASPoGen
#    Yet Another Synthetic Population Generator
# that is part of the CoeGSS project.

from numpy.random import rand, randint
from numpy import argmax, zeros
import numpy as np

houseHoldRoleEnumerator = {\
                                "son": 0,\
                                "parent": 1,\
                                "elderly": 2,\
                          } # Parents, children, elderly

sexEnumerator = {\
                    "male": 0,\
                    "female": 1,\
                }


implementedSexParents = ["free", "male", "female", "etero"]
implementedSexSons    = ["free", "male", "female"]

ageDistributionYearsRange = (0, 100) # Extremes included!

# Ages to produce
numberOfAgeYears = 1 + ageDistributionYearsRange[1] - ageDistributionYearsRange[0]

def generatePopulation(houseHoldType, houseHoldTypeCDF, houseHoldType_sizeCDF, size=10000, offset_hhID=0, offset_agID=0):
    '''
    Generates a population given households type and target size.

    Parameters
    ----------

    houseHoldType - iterable (list or array)
        The array containing the household class objects whose methods will be
        called to generate households.

    houseHoldTypeCDF - array
        The CDF (or cumulative PDF) of the household type. This is used to
        sample an household kind at each step.

    houseHoldType_sizeCDF - list of arrays
        For each household type reports the CDF of the hh size distribution in a
        index - probability way (i.e., CDF[0] is the cdf for household of size
        1, CDF[1] those for hh of size 2 etc.)

    size - int
        The size (number of agents) of the population to be generated.

    offset_hhID - int
        The hh unique identifiers will start from here and will be incremented
        by 1 for each generated hh (useful when calling this method for every
        subregion of a region and you don't want many household to share the
        same id

    offset_agID - int
        The same as for `offset_hhID` but for the agent table.
    '''
    created = 0

    hhID = offset_hhID
    agentID = offset_agID

    houseHolds = []
    agents = []
    while created < size:
        tmp_hh_kind = argmax(houseHoldTypeCDF > rand())
        tmp_hh_size = argmax(houseHoldType_sizeCDF[tmp_hh_kind] > rand()) + 1

        try:
            #print "doing household kind %d of size %d" % (tmp_hh_kind, tmp_hh_size)
            tmp_components = houseHoldType[tmp_hh_kind].makeHousehold(tmp_hh_size)
        except AssertionError as e:
            print tmp_hh_kind, tmp_hh_size, str(e)
            continue

        houseHolds.append([hhID, tmp_hh_kind, tmp_hh_size])
        agents.extend([[agentID+i, hhID] + tmp_component
                        for i, tmp_component in enumerate(tmp_components)])

        hhID += 1
        agentID += tmp_hh_size
        created += tmp_hh_size
    return houseHolds, agents

class householdType(object):
    '''
    YASPoGen

    Yet Another Synthetic Population Generator

    '''
    def __init__(self, minMaxParents=(2,2), minMaxSons=(1,9), dMinMaxParSon=(15,45), dMinMaxp1p2=(0,15),\
                ageMinMaxParents=(18,65), ageMinMaxChildren=(0,25), fixedParentsSons=(True, False),\
                sexParents=None, sexSons=None, agePDFparents=None, agePDFsons=None,\
            ):
        '''

        Parameters
        ----------

        `dMinMaxParSon` tuple(int, int)
            The minimum distance between the youngest parent and the oldest children and the
            maximum distance between the oldest parent and the youngest children.

        `sexParents` str
            The sex of the parents. If specified must be one of the following:
                - "free" the sex are drawn freely from the sex distribution (e.g., to be used in multi households);
                - "male" only males (e.g. for lone father with children);
                - "female" same as `male`;
                - "etero" one male and one female, alternatively (e.g, start from one sex and then continue alternatively);
            If None parents' sex is chosen alternatively (e.g., male, female, male, female, etc.)

        `sexSons` str
            The sex of the sons. If specified must be one of the following:
                - "free" the sex are drawn freely from the sex distribution (e.g., to be used in normal households);
                - "male" only males;
                - "female" same as `male`;
            If None sons' sex is chosen freely as in the `free` case.

        `fixedParentsSons` bool
            Whether to fix the number of parents (true by default) and sons (false by default).
            Since we want to make the class transparent to the creation process we
            require to have at least one of the two fixed. the fixed number(s) must coincide with
            a `minMaxParents` tuple containing two equal values.

        '''

        # chekcing stuff...
        assert dMinMaxp1p2[1] + dMinMaxParSon[0] < dMinMaxParSon[1],\
                ("Please increase the maximum distance between parents and"
                " sons or decrease the maximum distance between parents.")
        assert sexParents is None or sexParents in implementedSexParents, (
                "`sexParents` must be `None` or one of the following: [%s]" %
                 ", ".join(implementedSexParents))

        assert sexSons is None or sexSons in implementedSexSons,\
                "`sexSons` must be `None` or one of the following: [%s]" % ", ".join(implementedSexSons)

        assert (not fixedParentsSons[0] or (minMaxParents[0] == minMaxParents[1]) ) or\
               (not fixedParentsSons[1] or (minMaxSons[0] == minMaxSons[1]) )

        # We expect a row of cdf per sex and age m_0, f_0, m_1, f_1, ..., m_100, f_100
        assert agePDFparents is None or\
                agePDFparents.shape == (2*numberOfAgeYears,), "Please pass a valid agePDFparents"
        assert agePDFsons is None or\
                agePDFsons.shape == (2*numberOfAgeYears,), "Please pass a valid agePDFsons"

        # Begin of __init__
        super(householdType, self).__init__()
        # min and max number of parents
        self.__minParents, self.__maxParents = minMaxParents

        # min and max number of sons
        self.__minSons, self.__maxSons = minMaxSons

        # min and max age difference between parents and sons
        self.__dminParSon, self.__dmaxParSon = dMinMaxParSon

        # min and max age difference between parents
        self.__dminp1p2, self.__dmaxp1p2 = dMinMaxp1p2

        # Min and Maximum age of parents
        self.__minAgeParents, self.__maxAgeParents = ageMinMaxParents

        # min and max ageof children
        self.__minAgeChildren, self.__maxAgeChildren = ageMinMaxChildren

        self.__sexParents = "etero" if sexParents is None else sexParents
        self.__sexSons = "free" if sexSons is None else sexSons

        # Whether to fix parents or sons number (or both)
        self.__fixedParents, self.__fixedSons = fixedParentsSons

        # Adjusting the cdf: if we passed none we use a uniform in the range of the ages for
        # parents and children
        if agePDFparents is None:
            # 1/ageRange population, index is age, value is the cdf
            pdfs = zeros(ageDistributionYearsRange[1] - ageDistributionYearsRange[0] + 1)
            pdfs[self.__minAgeParents:self.__maxAgeParents+1] = 1.
            pdfs = np.column_stack((pdfs, pdfs)).flatten(order="C")
            pdfs /= pdfs.sum()
            self.__agePDFparents = pdfs.copy()
            del pdfs
        else:
            self.__agePDFparents = agePDFparents

        if agePDFsons is None:
            # 1/age population, index is age, value is the cdf
            pdfs = zeros(ageDistributionYearsRange[1] - ageDistributionYearsRange[0] + 1)
            pdfs[self.__minAgeChildren:self.__maxAgeChildren+1] = 1.
            pdfs = np.column_stack((pdfs, pdfs)).flatten(order="C")
            pdfs /= pdfs.sum()
            self.__agePDFsons = np.column_stack((pdfs, pdfs))
            del pdfs
        else:
            self.__agePDFsons = agePDFsons

        self._minSize = self.__minParents + self.__minSons
        self._maxSize = self.__maxParents + self.__maxSons


    def __drawSexAgeParents(self, minAge, maxAge, sex=None):
        '''
        Draws a sex and age for parents.

        Parameters
        ----------

        minAge: int
            the minimum age to draw;
        maxAge: int
            the maximum age to draw (included);
        sex: [optional] None
            force this person to be of this sex (cdf will be masked accordingly)
        '''

        #TODO every change here must be reflected in the sons method!
        if self.__sexParents in ["male", "female"]:
            sex = sexEnumerator[self.__sexParents]

        offset = 0 if sex is None else sex
        step = 1 if sex is None else 2

        iniIndex = minAge*len(sexEnumerator) + offset
        finIndex = min(len(sexEnumerator)*numberOfAgeYears, len(sexEnumerator)*(1 + maxAge))


        tmp_pdf = self.__agePDFparents[iniIndex:finIndex:step].copy()
        tmp_pdf /= tmp_pdf.sum()
        tmp_cdf = tmp_pdf.cumsum()
        extractedIndex = argmax(rand() < tmp_cdf)

        stopIndex = iniIndex + extractedIndex*step
        extractedAge = stopIndex // len(sexEnumerator)
        extractedSex = stopIndex %  len(sexEnumerator)
        if sex:
            assert extractedSex == sex, "extracted sex %d != passed sex %d !!!!!" % (extractedSex, sex)
        assert minAge <= extractedAge <= maxAge, "%d <= %d <= %d" % (minAge, extractedAge, maxAge)

        #print minAge, maxAge, extractedAge, iniIndex, finIndex, extractedIndex

        return (extractedSex, extractedAge)

    def __drawSexAgeSons(self, minAge, maxAge, sex=None):
        '''
        Draws a sex and age for sons.

        Parameters
        ----------

        minAge: int
            the minimum age to draw;
        maxAge: int
            the maximum age to draw (included);
        sex: [optional] None
            force this person to be of this sex (cdf will be masked accordingly)
        '''

        if self.__sexSons in ["male", "female"]:
            sex = sexEnumerator[self.__sexSons]

        offset = 0 if sex is None else sex
        step = 1 if sex is None else 2

        iniIndex = minAge*len(sexEnumerator) + offset
        finIndex = min(len(sexEnumerator)*numberOfAgeYears, len(sexEnumerator)*(1 + maxAge))

        tmp_pdf = self.__agePDFsons[iniIndex:finIndex:step].copy()
        tmp_pdf /= tmp_pdf.sum()
        tmp_cdf = tmp_pdf.cumsum()
        extractedIndex = argmax(rand() < tmp_cdf)

        stopIndex = iniIndex + extractedIndex*step
        extractedAge = stopIndex // len(sexEnumerator)
        extractedSex = stopIndex %  len(sexEnumerator)
        if sex:
            assert extractedSex == sex, "extracted sex %d != passed sex %d !!!!!" % (extractedSex, sex)
        assert minAge <= extractedAge <= maxAge, "%d <= %d <= %d" % (minAge, extractedAge, maxAge)
        return (extractedSex, extractedAge)

    def makeHousehold(self, size):

        assert self._minSize <= size <= self._maxSize,\
                    "Problems with the size of the household: minSize <= size <= maxSize? %d <= %d <= %d " % (self._minSize, size, self._maxSize)
        

        # Since at least one of the two is fixed we fix one with the other using the fixed flags
        nParents = self.__minParents if self.__fixedParents else size - self.__minSons
        nSons = self.__minSons if self.__fixedSons else size - self.__minParents

        householdComponents = []

        minAgeAcceptable, maxAgeAcceptable = self.__minAgeParents, self.__maxAgeParents
        minAgeParents, maxAgeParents = self.__maxAgeParents, self.__minAgeParents

        if self.__sexParents in ["male", "female"]:
            sexHead = sexEnumerator[self.__sexParents]
        else:
            sexHead = None

        for parent in range(nParents):
            if parent > 0 and (self.__sexParents in ["etero"]):
                sexHead = 0 if sexHead == 1 else 1
            if self.__sexParents in ["free"]:
                # Reset sex head to None
                sexHead = None

            sexHead, ageHead = self.__drawSexAgeParents(minAgeAcceptable, maxAgeAcceptable, sex=sexHead)
            if parent == 0:
                minAgeAcceptable = max(minAgeAcceptable, ageHead - self.__dmaxp1p2)
                maxAgeAcceptable = min(maxAgeAcceptable, ageHead + self.__dmaxp1p2)

            minAgeParents = min(ageHead, minAgeParents)
            maxAgeParents = max(ageHead, maxAgeParents)
            householdComponents.append([houseHoldRoleEnumerator["parent"], sexHead, ageHead])

        # Doing sons
        minAgeAcceptable = max(self.__minAgeChildren, maxAgeParents - self.__dmaxParSon)
        minAgeAcceptable = max(self.__minAgeChildren, minAgeParents - self.__dmaxParSon) # Goncalves version
        maxAgeAcceptable = minAgeParents - self.__dminParSon #min(self.__maxAgeChildren, minAgeParents - self.__dminParSon)

        if self.__sexSons in ["male", "female"]:
            tmp_sex = sexEnumerator[self.__sexSons]
        else:
            tmp_sex = None

        for children in range(nSons):
            if children > 0:
                # Adjust sex depending on scheme
                if self.__sexSons in ["free"]:
                    tmp_sex = None
            tmp_sex, tmp_age = self.__drawSexAgeSons(minAgeAcceptable, maxAgeAcceptable, sex=tmp_sex)
            householdComponents.append([houseHoldRoleEnumerator["son"], tmp_sex, tmp_age])

        return householdComponents

    def setAgePDFparents(self, pdf):
        assert pdf.shape == (2*numberOfAgeYears,),\
           "Wrong shape %r for parents pdf (expected %r)."\
           % (pdf.shape, 2*numberOfAgeYears)
        self.__agePDFparents = pdf.copy()

    def setAgePDFsons(self, pdf):
        assert pdf.shape == (2*numberOfAgeYears,),\
           "Wrong shape %r for sons pdf (expected %r)."\
           % (pdf.shape, 2*numberOfAgeYears)
        self.__agePDFsons = pdf.copy()



class workplace():
    def __init__(self, kindID, sizePDF=None):
        '''
        Given the large size of workplaces it expects a PDF like this:
        ```
        PDF = {
                "BINS": [b0, b1, ..., bN],
                "PDF":  [c0, c1, ..., cN-1]
              }
        ```

        Then we extract one bin `i` from the PDF and then uniformly sample 
        an int between `b[i]` and `b[i+1]` where `b` is `PDF["BINS"]`.
        '''
        self.__kindID = kindID
        self.set_sizePDF(sizePDF)

    def set_sizePDF(self, sizePDF):
        if sizePDF is not None:
            self.__sizePDF = sizePDF["PDF"]
            self.__sizeBINS = sizePDF["BINS"]
            self.__diffBINS = np.diff(self.__sizeBINS)
            self.__sizePMF = self.__sizePDF * self.__diffBINS
            self.__sizeCDF = (self.__diffBINS*self.__sizePDF).cumsum()
        else:
            self.__sizePDF = None
            self.__sizePMF = None
            self.__diffBINS = None
            self.__sizeCDF = None
            self.__sizeBINS = None
        self.__checkCDF()

    def __checkCDF(self):
        assert (self.__sizeCDF is None) or\
                (np.isclose(self.__sizeCDF[-1], 1.)
                 and len(self.__sizeCDF) == len(self.__sizeBINS) - 1
                )

    def makeOne(self, idx, maxSize):
        if maxSize <= self.__sizeBINS[0]:
            size = max(1, int(maxSize))
        else:
            if maxSize >= self.__sizeBINS[-1]:
                max_bin = len(self.__sizeBINS)
            else:
                max_bin = np.argmax(self.__sizeBINS > maxSize)
            tmp_cdf = self.__sizePMF[:max_bin]
            tmp_cdf = (tmp_cdf/tmp_cdf.sum()).cumsum()
            selectedSizeBin = argmax(tmp_cdf > rand())
            low = int(self.__sizeBINS[selectedSizeBin])
            high = max(low+1, min(self.__sizeBINS[selectedSizeBin+1], maxSize))
            width = int(high-low)
            tmp_wpSize = (np.ones(width)/float(width)).cumsum()
            #tmp_wpSize = 1./np.arange(low, high)**.0
            #tmp_wpSize = np.cumsum(tmp_wpSize/tmp_wpSize.sum())
            size = low + np.argmax(tmp_wpSize > rand())
        return [idx, self.__kindID, size]

