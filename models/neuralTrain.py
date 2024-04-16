#MODEL: Neural Network

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import dataReading as dr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Load and polish data

trainFile = ".\\data\\train.csv"
testFile = ".\\data\\test.csv"

dataFrame = pd.read_csv(trainFile) #Load training .csv file as dataframe
testData = pd.read_csv(testFile) #Load test .csv file 

#Get reference statistics (male survivors, female survivors, age average, total survivors)

dataFrame = dr.correctNan(dataFrame, 'Age')
testData = dr.correctNan(testData, 'Age')
average = dr.average(dataFrame, 'Age')
men = dataFrame.loc[dataFrame.Sex == 'male']["Survived"]
menPercentage = (sum(men) / len(men)) * 100

women = dataFrame.loc[dataFrame.Sex == 'female']["Survived"]
womenPercentage = (sum(women) / len(women)) * 100

survivors = dataFrame["Survived"].sum()
survivorsPercentage = (survivors / len(dataFrame)) * 100

print(dr.valProb(dataFrame, 'Age', 10))

referenceValues = {"menPercentage": menPercentage, "womenPercentage": womenPercentage, "survivors": survivorsPercentage, "average": average}

#Train AI 

y = dataFrame["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
x = pd.get_dummies(dataFrame[features])
X_test = pd.get_dummies(testData[features])

print("y = ", y)

x_tensor = torch.tensor(x.values.astype(np.float32))
y_tensor = torch.tensor(y.values.astype(np.float32))

x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=x_train.shape[1], out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = torch.relu(torch.tensor(x, dtype=torch.float32))
        x = torch.relu(torch.tensor(x, dtype=torch.float32))
        x = torch.sigmoid(torch.tensor(x, dtype=torch.float32))
        return x

model = Net()

criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    
    num_classes = 5 
    y_train_one_hot = torch.zeros(len(y_train), num_classes).scatter_(1, y_train.view(-1, 1).long(), 1)
    y_train_one_hot.detach()
    
    loss = criterion(output, y_train_one_hot)
    loss.requires_grad = True
    loss.backward()
    optimizer.step()
    
    #print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

print("Xtest: ", X_test)
model.eval()
with torch.no_grad():
    output = model(torch.tensor(X_test.values.astype(np.float32)))
    predictions = (output > 0.5).float().cpu().numpy().flatten()

num_passengers = len(testData)
num_columns = len(predictions) // num_passengers
predictions_reshaped = np.reshape(predictions, (num_passengers, num_columns))

# Create DataFrame with reshaped predictions
output = pd.DataFrame(predictions_reshaped, columns=['Prediction_1', 'Prediction_2', ..., 'Prediction_419'])
output['PassengerId'] = testData['PassengerId']

output = output[['PassengerId'] + ['Prediction_' + str(i) for i in range(1, num_columns + 1)]]

output.to_csv(".\\generated\\submission.csv", index=False)

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