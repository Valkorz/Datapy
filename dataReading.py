import pandas as pd
import sys
import numpy as np
import math as m
import eelem as em

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
        print("Incorrect parameters. Correct syntax is: correctNan(DataFrame,str).")
        return -1

    avg = average(dataFrame, type)
    newDataFrame = dataFrame
    
    for i in range(len(dataFrame)):
        currentNum = dataFrame.iloc[i][type]
        
        if np.isnan(currentNum):
            newDataFrame.iloc[i][type] = avg
            
    return newDataFrame


#Returns smallest value from dataframe
def min(dataFrame : pd.DataFrame, type : str) -> int:  
    if not isinstance(dataFrame, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Correct syntax is: min(DataFrame,str).")
        return -1
    
    min = 999
    
    for i in range(len(dataFrame)):
        if not np.isnan(dataFrame.iloc[i][type]) and dataFrame.iloc[i][type] < min:
            min = dataFrame.iloc[i][type]       
            
    return min
            
#Returns largest value from dataframe
def max(dataFrame : pd.DataFrame, type : str) -> int:
    if not isinstance(dataFrame, pd.DataFrame) or not isinstance(type, str):
        print("Incorrect parameters. Correct syntax is: max(DataFrame,str).")
        return -1
    
    max = 0
    
    for i in range(len(dataFrame)):
        if not np.isnan(dataFrame.iloc[i][type]) and dataFrame.iloc[i][type] > max:
            max = dataFrame.iloc[i][type]
    
    return max
 
#Returns a dataframe with the percentage of each interval   
def valProb(dataFrame : pd.DataFrame, type : str, interval : int) -> pd.DataFrame:
    
    if not isinstance(dataFrame, pd.DataFrame) or not isinstance(type, str) or not isinstance(interval, int):
        print("Incorrect parameters. Correct syntax is: valProb(DataFrame,str,float).")
        return -1
    
    avg = average(dataFrame, type)
    minVal = min(dataFrame, type)
    maxVal = max(dataFrame, type)
    data = pd.DataFrame()
    print("Min: ", minVal)
    print("Max: ", maxVal)
    
    intervalLength = (int)(maxVal / interval)
    intervals = np.zeros(intervalLength)
    elements = em.enumeratedElement(intervalLength)
    
    for i in range(len(intervals)):
        intervals[i] += intervalLength * i
        
    for i in range(len(dataFrame)):
        for x in range(len(intervals)):
            val = dataFrame.iloc[i][type]
            if not np.isnan(val):
                if not (x - 1) < 0 and val >= intervals[x - 1] and val < intervals[x]:
                    elements.add(intervals[x])
                elif (x - 1) < 0: 
                    elements.add(intervals[0])
    
    #elements.percentageOf(intervals[1])
    return elements.toDataFrame()

        
    
    
    
    
    
            
        
