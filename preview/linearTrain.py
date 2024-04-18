#MODEL: Linear

import numpy as np
import pandas as pd
import dataReading as dr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Load and polish data

trainFile = ".\\data\\train.csv"
testFile = ".\\data\\test.csv"

dataFrame = pd.read_csv(trainFile) #Load training .csv file as dataframe
testData = pd.read_csv(testFile) #Load test .csv file 

#Get reference statistics (male survivors, female survivors, age average, total survivors)

dataFrame = dr.correctNan(dataFrame, 'Age')
average = dr.average(dataFrame, 'Age')
men = dataFrame.loc[dataFrame.Sex == 'male']["Survived"]
menPercentage = (sum(men) / len(men)) * 100

women = dataFrame.loc[dataFrame.Sex == 'female']["Survived"]
womenPercentage = (sum(women) / len(women)) * 100

survivors = dataFrame["Survived"].sum()
survivorsPercentage = (survivors / len(dataFrame)) * 100

print(dr.valProb(dataFrame, 'Age', 10))

referenceValues = {"menPercentage": menPercentage, "womenPercentage": womenPercentage, "survivors": survivorsPercentage, "average": average}

#Train AI using logistic regression

y = dataFrame["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
x = pd.get_dummies(dataFrame[features])
X_test = pd.get_dummies(testData[features])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(X_test)

output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': y_pred})
output.to_csv(".\\generated\\submission.csv", index = False)

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




