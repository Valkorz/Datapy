import pandas as pd
import sys
import numpy as np
import math as m

#Calculate average of an element inside sample. It's a method that takes two strings and returns a float.
def average(dataFrame : pd.DataFrame, type : str) -> float:
    
    if not isinstance(dataFrame, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Must be average(DataFrame,str)")
        return -1
    
    avg = 0
    nans = 0
    
    try:
        if isinstance(type, str):              
                  
            for i in range(len(dataFrame)):
                currentNum = dataFrame.iloc[i][type]
                
                if np.isnan(currentNum):
                    nans += 1
                    continue
                
                avg += dataFrame.iloc[i][type]
        
            avg = avg / (len(dataFrame) - nans)
            
        else :
            print("datatype ",type, "not found.")
    except:
        print("Error:",sys.exc_info()[0], "occured.")
    
    return avg


#Returns the standard deviation from the data
def stdDeviation(dataFrame : pd.DataFrame, type : str) -> float:
    if not isinstance(dataFrame, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Must be average(DataFrame,str)")
        return -1
    
    avg = average(dataFrame, type)
    sum = 0
    num = len(dataFrame) - 1
    for i in range(len(dataFrame)):
        
        currentNum = dataFrame.iloc[i][type]
        
        if not np.isnan(currentNum):
            sum += currentNum - avg
            
        
    print("Deviation: +=", m.sqrt(sum / num))
    return m.sqrt(sum / num)
    

#Replace NaN values by for the average value. Takes two strings and returns a pd.DataFrame class.
def correctNan(dataFrame : pd.DataFrame, type : str) -> pd.DataFrame:
      
    if not isinstance(dataFrame, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Must be correctNan(DataFrame,str).")
        return -1

    avg = average(dataFrame, type)
    newDataFrame = dataFrame
    
    for i in range(len(dataFrame)):
        currentNum = dataFrame.iloc[i][type]
        
        if np.isnan(currentNum):
            newDataFrame.iloc[i][type] = avg
            
    return newDataFrame
    
    
            
        
