#MODEL: RANDOM FOREST CLASSIFIER

#IMPORTS
import pandas as pd;
import numpy as np;
import tools.dataMath as dm
from sklearn.ensemble import RandomForestClassifier

#GET/CLEAN DATA
trainFile = pd.read_csv(".\\data\\train.csv")
testFile = pd.read_csv(".\\data\\test.csv")

tr_averageAge = dm.average(trainFile, 'Age')
tr_averageSex = dm.average(trainFile, 'Sex')

te_averageAge = dm.average(testFile, 'Age')
te_averageSex = dm.average(testFile, 'Sex')

if(dm.hasNan(trainFile, 'Age')):
    trainFile = dm.correctNan(trainFile, 'Age')
    
if(dm.hasNan(trainFile, 'Sex')):
    trainFile = dm.correctNan(trainFile, 'Sex')
    
if(dm.hasNan(testFile, 'Age')):
    testFile = dm.correctNan(testFile, 'Age')

if(dm.hasNan(testFile, 'Sex')):
    testFile = dm.correctNan(testFile, 'Sex')

trainFile = dm.hasDuplicate(trainFile)
testFile = dm.hasDuplicate(testFile)

print(dm.getProportion(trainFile, 'Age'))

