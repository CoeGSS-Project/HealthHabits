import numpy as np
from scipy.stats import norm
import datetime

#"age2schoolKind": age2schoolKind,
#"age2defaultEdu": age2defaultEdu,
#dateZero

universityBins = np.arange(1000, 4100, 100)
universityPDF = norm.pdf((universityBins[:-1] + universityBins[1:])/2.,
                            loc=2500, scale=200)

def filterL0codes(l0, toKeep, id2code):
    tmp_code = id2code[l0]
    for keep in toKeep:
        if tmp_code.startswith(keep):
            return True
    return False

def getEmploymentProba(original_data, maxYears=100, sexDict={"M": 0, "F": 1}):
    '''
    Returns the probability to be emplyed for given sex, age and education level.

    We return the data as 
    array(a,s,l) where we have a rows for each age and s columns for each sex.
    Then, the l levels of education are given.
    '''
    nSexes = len(sexDict)
    nLevelsOfEdu = len(original_data.index.get_level_values(-1).unique())
    employmentRateByAgeSexEdu = np.zeros(shape=(maxYears+1, nSexes, nLevelsOfEdu))

    for col in original_data.index.get_level_values(0).unique():
        age_from, age_to = col.split("-")
        age_from = int(age_from)
        age_to = int(age_to) + 1

        for age in range(age_from, age_to):
            for sex_label, sex_id in sexDict.iteritems():
                employmentRateByAgeSexEdu[age, sex_id, :] =\
                        original_data[col, sex_label]
    return employmentRateByAgeSexEdu

def getEducationLevelCDF(original_data, maxYears=100, sexDict={"M": 0, "F": 1}):
    '''
    Returns the CDF of the education for the given data.
    We return the data as array(a,s,l) where we have a rows for each age
    and s columns for each sex.

    Then, the l levels of education are given as CDF.
    '''
    nSexes = len(sexDict)
    nLevelsOfEdu = len(original_data.index.get_level_values(-1).unique())
    educationLevelByAgeSex = np.zeros(shape=(maxYears+1, nSexes, nLevelsOfEdu))

    agesDone = set()
    for col in original_data.index.get_level_values(0).unique():
        age_from, age_to = col.split("-")
        age_from = int(age_from)
        age_to = int(age_to) + 1

        for age in range(age_from, age_to):
            agesDone.add(age)
            for sex_label, sex_id in sexDict.iteritems():
                educationLevelByAgeSex[age, sex_id, :] = original_data[col, sex_label]

    for age in range(0, 18):
        if age in agesDone:
            # If it was in data skip over...
            continue
        for sex_label, sex_id in sexDict.iteritems():
            if age < 14:
                # Always primary
                educationLevelByAgeSex[age, sex_id, :] = np.ones(nLevelsOfEdu, dtype=float)
            else:
                # Secondary
                educationLevelByAgeSex[age, sex_id, :] = np.array([0, 1., 1.], dtype=float)
    return educationLevelByAgeSex

def age2schoolKind(age):
    if age < 5:
        return 0
    elif age < 10:
        return 1
    elif age < 18:
        return 2
    elif age < 25:
        return 3
    else:
        raise RuntimeError, "Age %d >= 25!" % age

def age2defaultEdu(age):
    if age < 14:
        return 0
    elif age < 20:
        return 1
    elif age < 30:
        return 2
    else:
        raise RuntimeError, "Age %d >= 25!" % age

def agentSchoolEduEmploy(agent, eduRate, eduLevel, employAgeSexEduRate, agePos=4, sexPos=3):
    '''
    Given an agent in the `[id, hhid, role, sex, age]` format it returns a tuple of
    (eduLevel, wpKind, income) for the agent.
    
    wpID is -1 if unemployed/not attending school.
    '''
    
    age, sex = agent[agePos], agent[sexPos]
    
    if np.random.rand() < eduRate[age, sex]:
        tmp_wpKind = age2schoolKind(age)
        tmp_eduLevel = age2defaultEdu(age)
        tmp_income = .0
    else:
        if age < 18:
            tmp_eduLevel = age2defaultEdu(age)
        else:
            tmp_eduLevel = np.argmax(np.random.rand() < eduLevel[age, sex])
            
        if np.random.rand() < employAgeSexEduRate[age, sex, tmp_eduLevel]:
            tmp_wpKind = 10
        else:
            tmp_wpKind = -1
            
        if tmp_wpKind >= 10 or age > 65:
            tmp_income = np.random.normal(15000*(tmp_eduLevel+1), 1000*(tmp_eduLevel+1))
        else:
            tmp_income = .0
            
    return agent + [tmp_eduLevel, tmp_wpKind, tmp_income]

def getEducationRate(original_data, maxYears=100):
    # Create the attendance rate for zero-primary, secondary
    # and tertiary schools for each age...
    attendanceRate = np.zeros(maxYears+1, dtype=float)
   
    for col, val in original_data.iteritems():
        age_from, age_to = col.split("-")
        age_from = int(age_from)
        age_to = int(age_to) + 1
        
        for age in range(age_from, age_to):
            attendanceRate[age] = original_data[col]       
    # Return for male and females
    attendanceRate = np.column_stack((attendanceRate, attendanceRate))
    return attendanceRate

