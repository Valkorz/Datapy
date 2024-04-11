#1 - read a .csv training file, calculate statistic values (median, average, deviation) and clean up.
#2 - create a dataframe for the .csv
#3 - train a model using the cleaned up file.

import pandas as pd #Import pandas library as "pd"
import dataReading as dr #Import data reading functions
#import numpy as num #Import numpy as "num"

trainFile = ".\\data\\train.csv"

dataFrame = pd.read_csv(trainFile) #Load training .csv file as dataframe

average = dr.average(dataFrame, 'Age')
dataFrame = dr.correctNan(dataFrame, 'Age')
print(dr.stdDeviation(dataFrame, 'Age'))
print("average: ",average)
print(dataFrame)

