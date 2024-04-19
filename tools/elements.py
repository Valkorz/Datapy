import numpy as np
import pandas as pd
from typing import TypeVar, Any

#Class to analyze different elements and their count
class enumeratedElement:
    elements = np.full(1, None) #Array that stores elements. Cannot have repeating elements
    elementCount = np.zeros(1) #Array stores how many times has each element repeated while attempting to add.
    totalElements = 0
    

    def __init__(self, size : int):
        self.elements = np.full(size, None)
        self.elementCount = np.zeros(size)
        
    #Adds element to "elements" if it doesn't already exists on the first NaN index. If it does exist, increment elementCount by one.
    def add(self, element):
        
        contains = 0
        full = 1
        for i in range(len(self.elements)):
            if self.elements[i] == element:
                contains = 1
                self.totalElements += 1
                self.elementCount[i] += 1
        
        if contains == 0:
            for i in range(len(self.elements)):
                if self.elements[i] is None:
                    self.elements[i] = element
                    self.totalElements += 1
                    self.elementCount[i] += 1
                    full = 0
                    break
        
       
    
    
    #Resizes collection
    def resize(self):
        newElements = np.full(len(self.elements) + 100, None)
        newCounts = np.full(len(self.elementCount) + 100, None)
        
        for i in range(len(self.elements)):
            newElements[i] = self.elements[i]
        
        for i in range(len(self.elementCount)):
            newCounts[i] = self.elementCount[i]
        
    
    #Returns number of times an element has been included
    def count(self, element) -> int:
        for i in range(len(self.elements)):
            if self.elements[i] == element:
                return self.elementCount[i]
        return 0    
    
    #Returns collection length
    def length(self) -> dict:
        elementLength = {}
        for i in range(len(self.elements)):
            elementLength[self.elements[i]] = self.elementCount
        return elementLength
            
            
    
    #Removes an x amount of an y element.
    def remove(self, element, amount : int):
        for i in range(len(self.elements)):
            if self.elements[i] == element and (self.elementCount[i] - amount) > 0:
                self.totalElements -= amount
                self.elementCount[i] -= amount
                return 0
            elif self.elements[i] == element and (self.elementCount[i] - amount) <= 0:
                self.totalElements -= amount
                self.elementCount[i] = 0
                self.elements[i] = None
                return 0
            else:
                print("None of", element, " found.")
                return -1
            
    #Returns most common element
    def mostCommonElement(self) -> Any:
        freq = {"Highest":0}
        num = 0
        for i in range(len(self.elementCount)):
            if self.elementCount[i] > num:
                num = self.elementCount[i]
                freq["Highest"] = i
        
        return self.elements[freq["Highest"]]
    
    #Calculates the percentage of an element over the total number of elements.
    def percentageOf(self, element) -> float:       
        for i in range(len(self.elements)):
            if self.elements[i] == element:
                percentage = (self.elementCount / self.totalElements) * 100
                print("Percentage of ", element," is ", percentage)
                return percentage
        return 0
    
    #Converts list into dataframe of values
    def toDataFrame(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for i in range(len(self.elements)):
            data.add({f"Interval to: {self.elements[i]}": self.elementCount[i]})
        return data

    #Prints all values out
    def dumpInfo(self):
        for i in range(len(self.elements)):
            print("Element ", self.elements[i], "of index", i ,"has count of: ", self.elementCount[i])
            
    def proportions(self) -> dict:
        propDict = {}
    
        for i in range(len(self.elements)):
            if self.elements[i] is None: continue
            else: propDict[self.elements[i]] = (self.elementCount[i] / self.totalElements) * 100
        
        return propDict
            
        
            
            
    