#MODEL: RandomForestClassifier


#1 - read a .csv training file, calculate statistic values (median, average, deviation) and clean up.
#2 - create a dataframe for the .csv
#3 - train a model using the cleaned up file.

import pandas as pd #Import pandas library as "pd"
import dataReading as dr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score;
import numpy as np

trainFile = ".\\data\\train.csv"
testFile = ".\\data\\test.csv"

dataFrame = pd.read_csv(trainFile) #Load training .csv file as dataframe
testData = pd.read_csv(testFile) #Load test .csv file 

#Clear dataframe

average = dr.average(dataFrame, 'Age')
dataFrame = dr.correctNan(dataFrame, 'Age')
testData = dr.correctNan(testData, 'Age')

#Get reference statistics (male survivors, female survivors, age average, total survivors)

men = dataFrame.loc[dataFrame.Sex == 'male']["Survived"]
menPercentage = (sum(men) / len(men)) * 100

women = dataFrame.loc[dataFrame.Sex == 'female']["Survived"]
womenPercentage = (sum(women) / len(women)) * 100

survivors = dataFrame["Survived"].sum()
survivorsPercentage = (survivors / len(dataFrame)) * 100

print(dr.valProb(dataFrame, 'Age', 10))

referenceValues = {"menPercentage": menPercentage, "womenPercentage": womenPercentage, "survivors": survivorsPercentage, "average": average}

#Train AI (code based on the Titanic Tutorial example at https://www.kaggle.com/code/alexisbcook/titanic-tutorial)

y = dataFrame["Survived"]
features = ["Age", "Sex", "Pclass"]
x = pd.get_dummies(dataFrame[features])
X_test = pd.get_dummies(testData[features])

model = RandomForestClassifier(n_estimators = 300,max_depth = 12,random_state = 42)
model.fit(x,y)
predictions = model.predict(X_test)

featureImportances = model.feature_importances_
importanceDf = pd.DataFrame({'Feature': x.columns, 'Importance': featureImportances})
importanceDf = importanceDf.sort_values(by='Importance', ascending=False)
print(importanceDf)

output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': predictions})
output.to_csv(".\\generated\\submission.csv", index = False)
print("Saved submission successfully.")

#Get statistics from output file (male survivors, female survivors, age average, total survivors)

men = []
women = []
passengers = []
for i in range(len(testData)):  
    passengers.append(output.iloc[i,1])
    if testData.iloc[i,3] == 'male':
        men.append(output.iloc[i,1])
    elif testData.iloc[i,3] == 'female':
        women.append(output.iloc[i,1])

menPercentage = (sum(men) / len(men)) * 100
print("male count: ", len(men), "male survivors: ", sum(men))
womenPercentage = (sum(women) / len(women)) * 100
print("female count: ", len(women), "female survivors: ", sum(women))
survivorsPercentage = (sum(passengers) / len(passengers)) * 100
print("passenger count: ", len(passengers), "survivors: ", sum(passengers))
average = dr.average(testData, 'Age')

testValues = {"menPercentage": menPercentage, "womenPercentage": womenPercentage, "survivors": survivorsPercentage, "average": average}

#Compare reference and output statistics

print("Reference male survivors: ", referenceValues["menPercentage"], "| Output male survivors: ", testValues["menPercentage"])
print("Reference female survivors: ", referenceValues["womenPercentage"], "| Output female survivors: ", testValues["womenPercentage"])
print("Reference survivors: ", referenceValues["survivors"], "| Output survivors: ", testValues["survivors"])
print("Reference age average: ", referenceValues["average"], "| Output age average: ", testValues["average"])





