#DATAMATH: MATHEMATICAL FUNCTIONS FOR .CSV FILES

#IMPORT
import pandas as pd
import numpy as np
from tools.elements import enumeratedElement
from typing import Any
import sys

#DEFINITIONS

#RETURNS THE AVERAGE OF THE PROVIDED DATATYPE IN THE DATAFRAME
def average(data : pd.DataFrame, type : str) -> Any:
    
    if not isinstance(data, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Must be average(DataFrame,str)")
        return -1
    
    if type not in data.columns:
        print(f"Column '{type}' not found in DataFrame")
        return -1
    
    avg = 0
    nans = 0
    
   
    if isinstance(data.iloc[0][type], int) or isinstance(data.iloc[0][type], float):                        
        for i in range(len(data)):
            currentNum = data.iloc[i][type]
                
            if np.isnan(currentNum):
                nans += 1
                continue
                
            avg += data.iloc[i][type]    
        avg = avg / (len(data) - nans) 
        
    else:
        element = enumeratedElement(5)
        nan_vals = data[type].isna()
        for i in range(len(data)):
            currentNum = data.iloc[i][type]
                
            if nan_vals.iloc[i]:
                nans += 1
                continue              
            element.add(currentNum)
            
        avg = element.mostCommonElement()
  
    
    return avg


#CHECKS IF ANY COLUMN HAS NANs
def hasNan(data : pd.DataFrame, type : str) -> bool:
    if not isinstance(data, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Must be hasNan(DataFrame,str)")
        return -1
    
    if type not in data.columns:
        print(f"Column '{type}' not found in DataFrame")
        return -1
    
    nan_vals = data[type].isna()
    for i in range(len(nan_vals)):
        if np.isnan(nan_vals[i]):
            return True
    
    return False

#CORRECTS NANs
def correctNan(data : pd.DataFrame, type : str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Must be correctNan(DataFrame,str)")
        return -1
    newData = data.copy()
    
    avg = average(data, type)
    if np.issubdtype(data[type].dtype, np.number):
        for i in range(len(data)):
            if np.isnan(data.iloc[i][type]):
                newData.iloc[i][type] = avg
    else:
        missing_data = data[type].fillna(avg)
            
    return newData
    
#CHECKS IF HAS DUPLICATE ROWS, IF TRUE, REMOVES.
def hasDuplicate(data : pd.DataFrame) -> pd.DataFrame:
    dps = data.duplicated()
    for i in range(len(dps)):
         if dps[i] == True:
            data = data.drop(i)
    
    return data

#GETS PERCENTAGE VALUES FOR EACH ELEMENT
def getProportion(data : pd.DataFrame, type : str) -> dict:
    if not isinstance(data, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Must be getProportion(DataFrame,str)")
        return -1

    elements = enumeratedElement(len(data))
    if np.issubdtype(data[type].dtype, np.number):
        for i in range(len(data)):
            if np.isnan(data.iloc[i][type]): break
            else:
                elements.add(data.iloc[i][type])
    else:
        nan_vals = data[type].isna()
        for i in range(len(data)):
            if nan_vals[i] == True: break
            else:
                elements.add(data.iloc[i][type])
    
    #print(elements.length())
    return elements.proportions()
            
    
    