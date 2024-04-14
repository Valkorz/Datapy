#1 - read a .csv training file, calculate statistic values (median, average, deviation) and clean up.
#2 - create a dataframe for the .csv
#3 - train a model using the cleaned up file.

import pandas as pd #Import pandas library as "pd"
import dataReading as dr #Import data reading functions
from sklearn.ensemble import RandomForestClassifier

trainFile = ".\\data\\train.csv"
testFile = ".\\data\\test.csv"

dataFrame = pd.read_csv(trainFile) #Load training .csv file as dataframe
testData = pd.read_csv(testFile) #Load test .csv file 

average = dr.average(dataFrame, 'Age')
dataFrame = dr.correctNan(dataFrame, 'Age')

#Print values on reference file as an accuracy standard
print(dataFrame)

men = dataFrame.loc[dataFrame.Sex == 'male']["Survived"]
menPercentage = (sum(men) / len(men) * 100)
print("percentage of men who survived: ", menPercentage, "%" )

women = dataFrame.loc[dataFrame.Sex == 'female']["Survived"]
womenPercentage = (sum(women) / len(women) * 100)
print("percentage of women who survived: ", womenPercentage, "%")

survivors = dataFrame["Survived"].sum()
survivorsPercentage = (survivors / len(dataFrame) * 100)
print("Percentage of survivors: ", (survivors / len(dataFrame) * 100), "%")

print("Average age: ", average)

print(dr.valProb(dataFrame, 'Age', 10))

#Train AI (code based on the Titanic Tutorial example at https://www.kaggle.com/code/alexisbcook/titanic-tutorial)

y = dataFrame["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
x = pd.get_dummies(dataFrame[features])
X_test = pd.get_dummies(testData[features])

model = RandomForestClassifier(n_estimators = 100,max_depth = 5,random_state = 1)
model.fit(x,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': predictions})
output.to_csv(".\\generated\\submission.csv", index = False)
print("Saved submission successfully.")

#Compare reference and result





