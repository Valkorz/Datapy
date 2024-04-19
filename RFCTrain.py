#MODEL: RANDOM FOREST CLASSIFIER

#IMPORTS
import pickle as pk
import pandas as pd;
import numpy as np;
import os 
import tools.dataMath as dm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

#GET/CLEAN DATA
trainFile = pd.read_csv(".\\data\\train.csv")
testFile = pd.read_csv(".\\data\\test.csv")
modelFile = '.\\models\\trained_rfc.pkl'

tr_averageAge = dm.average(trainFile, 'Age')
tr_averageSex = dm.average(trainFile, 'Sex')
tr_averagePclass = dm.average(trainFile, 'Pclass')

te_averageAge = dm.average(testFile, 'Age')
te_averageSex = dm.average(testFile, 'Sex')
te_averagePclass = dm.average(testFile, 'Pclass')

if(dm.hasNan(trainFile, 'Age')):
    trainFile = dm.correctNan(trainFile, 'Age')
    
if(dm.hasNan(trainFile, 'Sex')):
    trainFile = dm.correctNan(trainFile, 'Sex')
    
if(dm.hasNan(trainFile, 'Pclass')):
    trainFile = dm.correctNan(trainFile, 'Pclass')
    
if(dm.hasNan(testFile, 'Age')):
    testFile = dm.correctNan(testFile, 'Age')

if(dm.hasNan(testFile, 'Sex')):
    testFile = dm.correctNan(testFile, 'Sex')
    
if(dm.hasNan(testFile, 'Pclass')):
    testFile = dm.correctNan(testFile, 'Pclass')

trainFile = dm.hasDuplicate(trainFile)
testFile = dm.hasDuplicate(testFile)


#PRINT RECEIVED VALUES

print("\n \n Proportion of age (%): \n \n trainFile: ", dm.getProportion(trainFile, 'Age'), " \n \n testFile: ", dm.getProportion(testFile, 'Age'))
print("\n \n Proportion of sex (%): \n \n trainFile: ", dm.getProportion(trainFile, 'Sex'), " \n \n testFile: ", dm.getProportion(testFile, 'Sex'))
print("\n \n Proportion of class (%): \n \n trainFile: ", dm.getProportion(trainFile, 'Pclass'), " \n \n testFile: ", dm.getProportion(testFile, 'Pclass'))

print("Average age: \n \n trainFile: ", tr_averageAge, "\n \n testFile: ", te_averageAge, "\n ########################")
print("Average sex: \n \n trainFile: ", tr_averageSex, "\n \n testFile: ", te_averageSex, "\n ########################")
print("Average class: \n \n trainFile: ", tr_averagePclass, "\n \n testFile: ", te_averagePclass, "\n ########################")

#ADD FAMILY CLASSIFIER COLUMN (WIP)

#LOAD OR TRAIN MODEL

model_exists = os.path.isfile(modelFile)

train_data, validation_data = train_test_split(trainFile, test_size=0.2)
targetColumn = train_data['Survived']
targetFeatures = ['Age', 'Sex', 'Pclass']
train_features = pd.get_dummies(train_data[targetFeatures])
validation_features = pd.get_dummies(validation_data[targetFeatures])

if not model_exists:
    model = RandomForestClassifier(n_estimators = 300,max_depth = 12,random_state = 42)
    model.fit(train_features, train_data['Survived'])

else:
    with open('.\\models\\trained_rfc.pkl', 'rb') as file:
        model = pk.load(file)

predictions = model.predict(validation_features)

#PRINT FEATURE IMPORTANCE

featureImportances = model.feature_importances_
importanceDf = pd.DataFrame({'Feature': train_features.columns, 'Importance': featureImportances})
importanceDf = importanceDf.sort_values(by='Importance', ascending=False)

print("\n \n ############## \n \n Value importance: ", importanceDf, " \n ################### \n")

#CONVERT PREDICTION TO CSV, MERGE PREDICTION WITH TEST

predictionFile = pd.DataFrame({'PassengerId': validation_data.PassengerId, 'Survived': predictions})
predictionFile.to_csv(".\\generated\\submission.csv", index = False)
new_testFile = validation_data.copy()
new_testFile = new_testFile.assign(Survived=predictions)
print("Model finished. \n \n")

#TEST ACCURACY

print("\n ###################### \n Analyzing statistics: \n")

model_accuracy = (predictions == validation_data['Survived']).mean()
tr_survivorProportions = dm.getProportion(trainFile, 'Survived')
tr_survivorAverage = dm.average(trainFile, 'Survived')

te_survivorProportions = dm.getProportion(new_testFile, 'Survived')
te_survivorAverage = dm.average(new_testFile, 'Survived')

print("\n -Model accuracy: ", model_accuracy)
print("\n -Survival proportions: ", te_survivorProportions ," to ", tr_survivorProportions)
print("\n -Survival average: ", tr_survivorAverage ,"to", te_survivorAverage)

#SAVE MODEL

with open('.\\models\\trained_rfc.pkl', 'wb') as file:
    pk.dump(model, file)
    
print("\n Build saved to trained_rfc.pkl successfully.")

